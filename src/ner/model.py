import torch
import justsdk

from torch import nn
from .config import NerConfig
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Optional


class NerModel(nn.Module):
    def __init__(self, num_labels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_labels = num_labels
        justsdk.print_debug(f"num_labels: {num_labels}")
        model_name = NerConfig.base_model_name

        # Init BERT model
        justsdk.print_info(f"Loading pre-trained: {model_name}")
        self.bert_config = BertConfig.from_pretrained(model_name)
        self.bert_config.output_hidden_states = True
        self.bert_model = BertModel.from_pretrained(model_name, config=self.bert_config)
        self.bert_model.to(NerConfig.device)  # XXX: CUDA here?

        # Classification head
        self.classifier = nn.Linear(
            in_features=self.bert_config.hidden_size,
            out_features=num_labels,
        )
        self.dropout = nn.Dropout(NerConfig.dropout_rate)

        if NerConfig.freeze_bert:
            self._freeze_bert()
        elif NerConfig.freeze_bert_encoder:
            self._freeze_bert(encoder=True)

        self._init_weights()

    def _freeze_bert(self, encoder: bool = False) -> None:
        if encoder:
            justsdk.print_info("Freezing BERT encoder")
            for param in self.bert_model.encoder.parameters():
                param.requires_grad_(False)
        else:
            justsdk.print_info("Freezing whole BERT")
            # XXX: Is this for real freezing the whole BERT???
            for param in self.bert_model.parameters():
                param.requires_grad_(False)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        outputs: BaseModelOutputWithPooling = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        res = {
            "logits": logits,
            "hidden_states": sequence_output,
        }

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=NerConfig.ignore_index)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            res["loss"] = loss

        return res
