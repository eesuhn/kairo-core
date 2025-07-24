import justsdk
import torch

from .config import ExtSumConfig
from .model import ExtSumModel
from configs._constants import MODEL_DIR
from transformers import AutoTokenizer
from typing import Union


MODEL_EXT_SUM_BEST_PATH = MODEL_DIR / "summary" / "extractive" / "model.pt"


class ExtSumPredictor:
    def __init__(self, config: ExtSumConfig) -> None:
        self.config = config
        self.cp = justsdk.ColorPrinter(quiet=self.config.quiet)

        self.cp.info("Loading Extractive Summarizer...")
        self.model = ExtSumModel(config=self.config)

        state_dict = torch.load(
            MODEL_EXT_SUM_BEST_PATH, map_location=self.config.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.config.device)
        self.model.eval()
        self.cp.success("Loaded Extractive Summarizer")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)

    def predict(
        self,
        text: Union[str, list],
        return_scores: bool = False,
        min_sentences_per_type: int = None,
        max_sentences_per_type: int = None,
    ) -> dict:
        if isinstance(text, str):
            sentences = self._split_into_sentences(text)
        else:
            sentences = text

        if not sentences:
            empty_result = {"challenge": [], "approach": [], "outcome": []}
            return (empty_result, empty_result) if return_scores else empty_result

        predictions, scores = self._get_predictions(sentences)

        if min_sentences_per_type is None:
            min_sentences_per_type = self.config.min_sentences_per_type

        results = {}
        result_scores = {}

        for label_type in ["challenge", "approach", "outcome"]:
            selected_indices = predictions[label_type]
            selected_scores = scores[label_type]

            if len(selected_indices) < min_sentences_per_type:
                top_k_indices = torch.argsort(
                    torch.tensor(selected_scores), descending=True
                )[:min_sentences_per_type]
                selected_indices = sorted(top_k_indices.tolist())

            if (
                max_sentences_per_type
                and len(selected_indices) > max_sentences_per_type
            ):
                sorted_pairs = sorted(
                    zip(
                        selected_indices, [selected_scores[i] for i in selected_indices]
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )
                selected_indices = sorted(
                    [pair[0] for pair in sorted_pairs[:max_sentences_per_type]]
                )

            results[label_type] = [sentences[i] for i in selected_indices]
            if return_scores:
                result_scores[label_type] = [
                    selected_scores[i] for i in selected_indices
                ]

        return (results, result_scores) if return_scores else results

    def predict_batch(
        self, texts: Union[str, list], return_scores: bool = False, **kwargs
    ) -> list:
        results = []
        for text in texts:
            result = self.predict(text, return_scores=return_scores, **kwargs)
            results.append(result)
        return results

    def _get_predictions(self, sentences: list) -> tuple:
        encoding = self._tokenize_sentences(sentences)

        if encoding is None:
            empty_preds = {"challenge": [], "approach": [], "outcome": []}
            empty_scores = {
                "challenge": [0.0] * len(sentences),
                "approach": [0.0] * len(sentences),
                "outcome": [0.0] * len(sentences),
            }
            return empty_preds, empty_scores

        input_ids = encoding["input_ids"].unsqueeze(0).to(self.config.device)
        attention_mask = encoding["attention_mask"].unsqueeze(0).to(self.config.device)
        sentence_masks = encoding["sentence_masks"].unsqueeze(0).to(self.config.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentence_masks=sentence_masks,
            )

        challenge_probs = torch.sigmoid(outputs["challenge_logits"]).squeeze(0)
        approach_probs = torch.sigmoid(outputs["approach_logits"]).squeeze(0)
        outcome_probs = torch.sigmoid(outputs["outcome_logits"]).squeeze(0)

        predictions = {
            "challenge": (challenge_probs > self.config.challenge_threshold)
            .nonzero(as_tuple=True)[0]
            .tolist(),
            "approach": (approach_probs > self.config.approach_threshold)
            .nonzero(as_tuple=True)[0]
            .tolist(),
            "outcome": (outcome_probs > self.config.outcome_threshold)
            .nonzero(as_tuple=True)[0]
            .tolist(),
        }

        scores = {
            "challenge": challenge_probs.cpu().tolist(),
            "approach": approach_probs.cpu().tolist(),
            "outcome": outcome_probs.cpu().tolist(),
        }

        return predictions, scores

    def _tokenize_sentences(self, sentences: list) -> dict:
        if len(sentences) > self.config.max_sentences:
            sentences = sentences[: self.config.max_sentences]

        text = f" {self.tokenizer.sep_token} ".join(sentences)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"][0]
        sentence_masks = self._create_sentence_masks(input_ids, len(sentences))

        if sentence_masks is None:
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"][0],
            "sentence_masks": sentence_masks,
        }

    def _create_sentence_masks(
        self, input_ids: torch.Tensor, num_sentences: int
    ) -> torch.Tensor:
        sep_token_id = self.tokenizer.sep_token_id
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]

        positions = [0] + sep_positions.tolist() + [len(input_ids)]

        masks = torch.zeros((num_sentences, len(input_ids)))

        for i in range(min(num_sentences, len(positions) - 1)):
            start = positions[i] + (1 if i > 0 else 0)
            end = positions[i + 1]
            masks[i, start:end] = 1

        return masks

    def _split_into_sentences(self, text: str) -> list:
        sentences = []
        current = []

        for word in text.split():
            current.append(word)

            if any(word.endswith(punct) for punct in [".", "!", "?"]):
                sentence = " ".join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []

        if current:
            sentence = " ".join(current).strip()
            if sentence:
                sentences.append(sentence)

        return sentences

    def extract_combined_summary(
        self,
        text: Union[str, list],
        max_sentences: int = 10,
        balanced: bool = True,
    ) -> str:
        """
        Extract a combined summary from all categories.
        """
        results, scores = self.predict(text, return_scores=True)

        all_sentences = []

        for label_type in ["challenge", "approach", "outcome"]:
            for sent, score in zip(results[label_type], scores[label_type]):
                all_sentences.append(
                    {
                        "sentence": sent,
                        "score": score,
                        "type": label_type,
                    }
                )

        all_sentences.sort(key=lambda x: x["score"], reverse=True)

        if balanced:
            selected = []
            type_counts = {"challenge": 0, "approach": 0, "outcome": 0}
            max_per_type = max_sentences // 3 + 1

            for item in all_sentences:
                if len(selected) >= max_sentences:
                    break
                if type_counts[item["type"]] < max_per_type:
                    selected.append(item["sentence"])
                    type_counts[item["type"]] += 1

            for item in all_sentences:
                if len(selected) >= max_sentences:
                    break
                if item["sentence"] not in selected:
                    selected.append(item["sentence"])
        else:
            selected = [item["sentence"] for item in all_sentences[:max_sentences]]

        return " ".join(selected)
