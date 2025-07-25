import justsdk
import torch
import numpy as np
import torch.nn.functional as F
import re

from .model import NerModel
from .config import NerConfig
from .helper import NerHelper
from configs._constants import MODEL_DIR
from transformers import BertTokenizerFast
from typing import Union
from dataclasses import dataclass
from src.summary.extractive import ExtSumModel, ExtSumConfig


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
    def __init__(self, config: NerConfig) -> None:
        if not MODEL_NER_BEST_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_NER_BEST_PATH}")

        self.config = config
        self.cp = justsdk.ColorPrinter(quiet=self.config.quite)

        self.cp.info("Loading NER...")
        self.uni_labels, _ = NerHelper.get_uni_label_map()
        self.id_to_label = {i: label for i, label in enumerate(self.uni_labels)}

        self.model = NerModel(config=self.config, num_labels=len(self.uni_labels))
        # Load the best model weights
        state_dict = torch.load(
            MODEL_NER_BEST_PATH, map_location=self.config.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.config.device)
        self.model.eval()  # NOTE: Eval mode to prevent dropout during inference
        self.cp.success("Loaded NER")

        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
            self.config.base_model_name
        )

        # For deduplication
        ext_sum_config = ExtSumConfig(quiet=True)
        self.ext_sum_model = ExtSumModel(config=ext_sum_config)
        self.ext_sum_model.to(self.config.device)
        self.ext_sum_model.eval()

    def predict(self, texts: Union[str, list[str]]) -> list:
        single_str = isinstance(texts, str)
        texts = [texts] if single_str else texts

        all_entities = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            all_entities.extend(self._predict_batch(batch))

        if self.config.deduplicate_entities:
            all_entities = [self._deduplicate_entities(ents) for ents in all_entities]

        return all_entities[0] if single_str else all_entities

    def _predict_batch(self, texts: list[str]) -> list:
        tokens = [text.split() for text in texts]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        model_inputs = {k: v.to(self.config.device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self.model(**model_inputs)["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            confidences = torch.max(probs, dim=-1).values.cpu().numpy()

        return [
            self._extract_entities(
                tokens[i], preds[i], confidences[i], encoding.word_ids(i)
            )
            for i in range(len(texts))
        ]

    def _extract_entities(
        self,
        tokens: list[str],
        preds: np.ndarray,
        confidences: np.ndarray,
        word_ids: list,
    ) -> list:
        """Extract entities from model predictions"""
        entities = []
        current_entity = None
        processed_words = set()

        def _finalize_entity(entity):
            if entity:
                entity.text = self._normalize_text(entity.text)
                entities.append(entity)

        for word_id, pred_id, conf in zip(word_ids, preds, confidences):
            # NOTE: Skip special tokens and processed subtokens
            if word_id is None or (
                self.config.aggregate_subtokens and word_id in processed_words
            ):
                continue

            processed_words.add(word_id)
            label = self.id_to_label[pred_id]

            if conf < self.config.confidence_threshold:
                _finalize_entity(current_entity)
                current_entity = None
                continue

            if label == "O":
                _finalize_entity(current_entity)
                current_entity = None

            elif label.startswith("B-"):
                _finalize_entity(current_entity)
                current_entity = NerEntity(
                    text=tokens[word_id],
                    label=label[2:],
                    start=word_id,
                    end=word_id,
                    confidence=float(conf),
                )

            elif (
                label.startswith("I-")
                and current_entity
                and label[2:] == current_entity.label
            ):
                # Continue current entity
                current_entity.text += " " + tokens[word_id]
                current_entity.end = word_id

                if self.config.return_confidence:
                    n = current_entity.end - current_entity.start + 1
                    current_entity.confidence = (
                        current_entity.confidence * (n - 1) + float(conf)
                    ) / n
            else:
                # Label mismatch
                _finalize_entity(current_entity)
                current_entity = None

        _finalize_entity(current_entity)
        return entities

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        inputs = self.tokenizer(
            [text1, text2],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.ext_sum_model.bert(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] tokens

        similarity = F.cosine_similarity(embeddings[0:1], embeddings[1:2])
        return similarity.item()

    def _deduplicate_entities(self, entities: list[NerEntity]) -> list[NerEntity]:
        if not entities:
            return entities

        text_groups = {}
        for i, entity in enumerate(entities):
            key = self._normalize_text(entity.text)
            text_groups.setdefault(key, []).append(i)

        deduplicated = []
        processed = set()

        for indices in text_groups.values():
            if all(i in processed for i in indices):
                continue

            label_groups = {}
            for idx in indices:
                if idx not in processed:
                    label_groups.setdefault(entities[idx].label, []).append(idx)

            if label_groups:
                best_label = max(
                    label_groups,
                    key=lambda l: sum(entities[i].confidence for i in label_groups[l]),  # noqa: E741
                )
                best_idx = max(
                    label_groups[best_label], key=lambda i: entities[i].confidence
                )
                deduplicated.append(entities[best_idx])
                processed.update(indices)

        remaining = [i for i in range(len(entities)) if i not in processed]

        for i in remaining:
            if i in processed:
                continue

            similar = [i]
            base_text = self._normalize_text(entities[i].text)

            for j in remaining[i + 1 :]:
                if j not in processed:
                    comp_text = self._normalize_text(entities[j].text)
                    if (
                        self._calculate_similarity(base_text, comp_text)
                        >= self.config.similarity_threshold
                    ):
                        similar.append(j)

            if len(similar) > 1:
                label_groups = {}
                for idx in similar:
                    label_groups.setdefault(entities[idx].label, []).append(idx)

                best_label = max(
                    label_groups,
                    key=lambda l: (  # noqa: E741
                        len(label_groups[l]),
                        max(entities[i].confidence for i in label_groups[l]),
                    ),
                )
                best_idx = max(
                    label_groups[best_label], key=lambda i: entities[i].confidence
                )
            else:
                best_idx = i

            deduplicated.append(entities[best_idx])
            processed.update(similar)

        return deduplicated

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"[^\w\s]", "", text)
        return " ".join(text.split()).lower()
