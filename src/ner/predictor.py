import justsdk
import torch
import numpy as np

from .model import NerModel
from .config import NerConfig
from .helper import NerHelper
from configs._constants import MODEL_DIR
from transformers import BertTokenizerFast
from typing import Union
from dataclasses import dataclass


MODEL_NER_BEST_PATH = MODEL_DIR / "ner" / "model.pt"


@dataclass
class NerEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


class NerPredictor:
    def __init__(self) -> None:
        if not MODEL_NER_BEST_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_NER_BEST_PATH}")

        justsdk.print_info(f"Loading NER model from {MODEL_NER_BEST_PATH}")
        self.config = NerConfig()
        self.uni_labels, _ = NerHelper.get_uni_label_map()
        self.id_to_label = {i: label for i, label in enumerate(self.uni_labels)}

        self.model = NerModel(num_labels=len(self.uni_labels))
        # Load the best model weights
        state_dict = torch.load(
            MODEL_NER_BEST_PATH, map_location=self.config.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.config.device)
        self.model.eval()  # NOTE: Eval mode to prevent dropout during inference
        justsdk.print_success("Loaded NER model")

        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
            self.config.base_model_name
        )

        self.return_confidence: bool = False
        self.aggregate_subtokens: bool = True

    def predict(
        self,
        texts: Union[str, list[str]],
    ) -> list:
        # Handle single string
        single_str: bool = isinstance(texts, str)

        if single_str:
            texts = [texts]

        # Process in batches
        all_ent: list = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]
            batch_ents = self._predict_batch(batch_texts)
            all_ent.extend(batch_ents)

        return all_ent[0] if single_str else all_ent

    def _predict_batch(
        self,
        texts: list[str],
    ) -> list:
        tokens = [text.split() for text in texts]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Keep original encoding for word_ids, create separate dict for model input
        model_inputs = {k: v.to(self.config.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**model_inputs)
            logits = outputs["logits"]

            # Get predictions
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            confidences = torch.max(probs, dim=-1).values

        preds = preds.cpu().numpy()
        confidences = confidences.cpu().numpy()

        batch_ents: list = []
        for batch_i in range(len(texts)):
            entities = self._extract_entities(
                tokens=tokens[batch_i],
                preds=preds[batch_i],
                confidences=confidences[batch_i],
                word_ids=encoding.word_ids(batch_index=batch_i),
            )
            batch_ents.append(entities)
        return batch_ents

    def _extract_entities(
        self,
        tokens: list[str],
        preds: np.ndarray,
        confidences: np.ndarray,
        word_ids: list,
    ) -> list:
        entities: list = []
        current_ent = None
        processed_word_ids = set()

        for _, (word_id, pred_id, conf) in enumerate(zip(word_ids, preds, confidences)):
            if word_id is None:
                continue
            if self.aggregate_subtokens and word_id in processed_word_ids:
                # NOTE: Skip processing subtokens
                continue

            processed_word_ids.add(word_id)
            label: str = self.id_to_label[pred_id]

            if conf < self.config.confidence_threshold:
                if current_ent:
                    # End current entity if confidence is low
                    entities.append(current_ent)
                    current_ent = None
                continue

            # End current entity if "O"
            if label == "O":
                if current_ent:
                    entities.append(current_ent)
                    current_ent = None

            elif label.startswith("B-"):
                if current_ent:
                    entities.append(current_ent)

                current_ent = NerEntity(
                    text=tokens[word_id],
                    label=label[2:],  # Remove "B-" prefix
                    start=word_id,
                    end=word_id,
                    # NOTE: Default to 1.0 if return_confidence is False
                    confidence=float(conf) if self.return_confidence else 1.0,
                )

            elif (
                label.startswith("I-")
                and current_ent
                and label[2:] == current_ent.label
            ):
                current_ent.text += " " + tokens[word_id]
                current_ent.end = word_id

                if self.return_confidence:
                    n_tokens = (
                        current_ent.end - current_ent.start + 1
                    )  # XXX: Does this always equal to zero?
                    current_ent.confidence = (
                        current_ent.confidence * (n_tokens - 1) + float(conf)
                    ) / n_tokens

            # End current entity if label mismatch
            else:
                if current_ent:
                    entities.append(current_ent)
                    current_ent = None

        # Include last entity if exists
        if current_ent:
            entities.append(current_ent)

        return entities
