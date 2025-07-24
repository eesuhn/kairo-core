import justsdk
import torch

from .config import AbsSumConfig
from torch import nn
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from typing import Optional


class AbsSumModel(nn.Module):
    def __init__(self, config: AbsSumConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = config
        self.cp = justsdk.ColorPrinter(quiet=self.config.quite)

        self.model: T5ForConditionalGeneration = (
            T5ForConditionalGeneration.from_pretrained(self.config.base_model_name)
        )
        self.tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(
            self.config.base_model_name,
        )

        self.model.to(self.config.device)

        if self.config.freeze_t5_encoder:
            self._freeze_t5_encoder()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return outputs

    def _freeze_t5_encoder(self) -> None:
        self.cp.info("Freezing T5 encoder params...")
        for param in self.model.encoder.parameters():
            param.requires_grad_(False)
