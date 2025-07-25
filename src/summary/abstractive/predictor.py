import justsdk
import torch

from .config import AbsSumConfig
from .model import AbsSumModel
from configs._constants import MODEL_DIR
from transformers import T5TokenizerFast
from typing import Union
from src.utils import Utils


MODEL_ABS_SUM_BEST_PATH = MODEL_DIR / "summary" / "abstractive" / "model.pt"


class AbsSumPredictor:
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
        dynamic_length: bool = True,
    ) -> Union[str, list[str]]:
        single_str = isinstance(texts, str)
        if single_str:
            texts = [texts]

        if preprocess_text:
            texts = [Utils.preprocess_text(text) for text in texts]

        summaries: list = []
        for i in range(0, len(texts), self.config.gen_batch_size):
            batch_texts: list = texts[i : i + self.config.gen_batch_size]

            batch_gen_configs = []
            if dynamic_length:
                for text in batch_texts:
                    gen_config = self._calculate_dynamic_gen_config(text)
                    batch_gen_configs.append(gen_config)
            else:
                batch_gen_configs = [self.gen_conf] * len(batch_texts)

            batch_inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=self.config.data_max_length,
                return_tensors="pt",
            )

            input_ids = batch_inputs["input_ids"].to(self.config.device)
            attention_mask = batch_inputs["attention_mask"].to(self.config.device)

            batch_outputs = []
            for idx in range(len(batch_texts)):
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids[idx : idx + 1],
                        attention_mask=attention_mask[idx : idx + 1],
                        **batch_gen_configs[idx],
                    )
                batch_outputs.append(outputs)

            for out in batch_outputs:
                summary = self.tokenizer.decode(
                    out[0], skip_special_tokens=True
                ).strip()
                summaries.append(summary)

        if ensure_capitalized:
            summaries = [Utils.ensure_capitalized(s) for s in summaries]

        if ensure_complete_sentence:
            summaries = [Utils.ensure_complete_sentence(s) for s in summaries]

        return summaries[0] if single_str else summaries

    def _calculate_dynamic_gen_config(self, text: str) -> dict:
        word_count = len(text.split())

        # TODO: Abstract these to `AbsSumConfig`
        THRESHOLDS = [32, 128, 512]
        BASE_MIN_RATIO = 0.4
        BASE_MAX_RATIO = 0.6

        tier = 0
        for threshold in THRESHOLDS:
            if word_count >= threshold:
                tier += 1
            else:
                break

        min_ratio = BASE_MIN_RATIO / (2**tier)
        max_ratio = BASE_MAX_RATIO / (2**tier)

        min_words = max(2, int(word_count * min_ratio))
        max_words = max(8, int(word_count * max_ratio))

        min_length = int(min_words * 1.2)
        max_length = int(max_words * 1.2)

        min_length = max(2, min(min_length, 200))
        max_length = max(8, min(max_length, 400))

        if min_length >= max_length:
            max_length = min_length + 8

        dynamic_config = self.gen_conf.copy()
        dynamic_config.update(
            {
                "min_length": min_length,
                "max_length": max_length,
                "num_beams": 2 if word_count < 50 else self.config.gen_num_beams,
                "repetition_penalty": 2.0
                if word_count < 50
                else self.config.repetition_penalty,
            }
        )

        return dynamic_config
