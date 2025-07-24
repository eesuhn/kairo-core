import justsdk
import torch

from .config import AbsSumConfig
from .model import AbsSumModel
from configs._constants import MODEL_DIR
from transformers import T5TokenizerFast
from typing import Union


MODEL_ABS_SUM_BEST_PATH = MODEL_DIR / "summary" / "abstractive" / "model.pt"


class AbsSumPredictor:
    COMPLETE_SYMBOLS = (".", "!", "?", '."', '!"', '?"', ".'", "!'", "?'")

    def __init__(self, config: AbsSumConfig) -> None:
        self.config = config
        self.cp = justsdk.ColorPrinter(quiet=self.config.quiet)

        self.cp.info("Loading Abstractive Summarizer...")
        self.model = AbsSumModel(config=self.config)
        state_dict = torch.load(
            MODEL_ABS_SUM_BEST_PATH, map_location=self.config.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.config.device)
        self.model.eval()
        self.cp.success("Loaded Abstractive Summarizer")

        self.tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(
            self.config.base_model_name
        )

        self.gen_conf = {
            "max_length": self.config.gen_max_length,
            "min_length": self.config.gen_min_length,
            "num_beams": self.config.gen_num_beams,
            "length_penalty": self.config.length_penalty,
            "repetition_penalty": self.config.repetition_penalty,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            "early_stopping": self.config.early_stopping,
        }

    def predict(
        self,
        texts: Union[str, list[str]],
        preprocess_text: bool = True,
        ensure_capitalized: bool = True,
        ensure_complete_sentence: bool = True,
    ) -> list:
        single_str = isinstance(texts, str)
        if single_str:
            texts = [texts]

        if preprocess_text:
            texts = [self._preprocess_text(text) for text in texts]

        summaries: list = []
        for i in range(0, len(texts), self.config.gen_batch_size):
            batch_texts: list = texts[i : i + self.config.gen_batch_size]

            batch_inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=self.config.gen_max_length,
                return_tensors="pt",
            )

            input_ids = batch_inputs["input_ids"].to(self.config.device)
            attention_mask = batch_inputs["attention_mask"].to(self.config.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **self.gen_conf,
                )

            # Decode outputs
            for out in range(outputs.shape[0]):  # NOTE: Batch size
                summary = self.tokenizer.decode(
                    outputs[out], skip_special_tokens=True
                ).strip()
                summaries.append(summary)

        if ensure_capitalized:
            summaries = [self._ensure_capitalized(s) for s in summaries]

        if ensure_complete_sentence:
            summaries = [self._ensure_complete_sentence(s) for s in summaries]

        return summaries[0] if single_str else summaries

    def _preprocess_text(self, text: str) -> str:
        if not text:
            return text

        lines = text.split("\n")
        processed_lines: list = []
        current_sentence: list = []

        for i, line in enumerate(lines):
            line = line.strip()

            if not line:
                if current_sentence:
                    processed_lines.append(" ".join(current_sentence))
                    current_sentence = []
                if i > 0 and i < len(lines) - 1:
                    processed_lines.append("")
                continue

            if self._is_complete_sentence(line):
                if current_sentence:
                    processed_lines.append(" ".join(current_sentence))
                    current_sentence = []
                processed_lines.append(line)
            else:
                current_sentence.append(line)
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if (
                        next_line
                        and next_line[0].isupper()
                        and line.endswith((",", ":"))
                    ):
                        processed_lines.append(" ".join(current_sentence))
                        current_sentence = []

        if current_sentence:
            processed_lines.append(" ".join(current_sentence))

        result = []
        for i, line in enumerate(processed_lines):
            if line:
                result.append(line)
            elif i > 0 and i < len(processed_lines) - 1:
                result.append("\n")

        return "\n".join(result)

    def _is_complete_sentence(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False

        if text.endswith(self.COMPLETE_SYMBOLS):
            return True

        if len(text.split()) <= 5 and text[0].isupper():
            continuation_words = [
                "and",
                "or",
                "but",
                "the",
                "a",
                "an",
                "with",
                "of",
                "in",
                "is",
                "are",
                "was",
                "were",
                "to",
                "for",
                "by",
                "from",
                "at",
            ]
            words = text.split()
            if len(words) > 1:
                if words[-1].lower() in continuation_words:
                    return False
            return True

        if text.endswith((",", ":", ";", "-")):
            return False

        return False

    def _ensure_complete_sentence(self, text: str) -> str:
        text = text.strip()
        if text and not text.endswith(self.COMPLETE_SYMBOLS):
            text += "."
        return text

    def _ensure_capitalized(self, text: str) -> str:
        if not text:
            return text

        if text[0].islower():
            text = text[0].upper() + text[1:]

        for symbol in self.COMPLETE_SYMBOLS:
            parts = text.split(symbol)
            if len(parts) > 1:
                result = parts[0]
                for i in range(1, len(parts)):
                    part = parts[i].lstrip()
                    if part and part[0].islower():
                        part = part[0].upper() + part[1:]
                    whitespace_prefix = parts[i][
                        : len(parts[i]) - len(parts[i].lstrip())
                    ]
                    result += symbol + whitespace_prefix + part
                text = result

        return text
