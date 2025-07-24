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
    ) -> list:
        single_str = isinstance(texts, str)
        if single_str:
            texts = [texts]

        if preprocess_text:
            texts = [Utils.preprocess_text(text) for text in texts]

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
            summaries = [Utils.ensure_capitalized(s) for s in summaries]

        if ensure_complete_sentence:
            summaries = [Utils.ensure_complete_sentence(s) for s in summaries]

        return summaries[0] if single_str else summaries
