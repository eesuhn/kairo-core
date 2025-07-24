import justsdk
import torch
import torch.nn as nn

from .config import ExtSumConfig
from transformers import AutoModel, AutoTokenizer
from typing import Optional


class ExtSumModel(nn.Module):
    def __init__(self, config: ExtSumConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = config
        self.cp = justsdk.ColorPrinter(quiet=self.config.quiet)

        self.cp.info(f"Loading {self.config.base_model_name}...")
        self.bert = AutoModel.from_pretrained(self.config.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)

        self.dropout = nn.Dropout(self.config.classifier_dropout)

        # Separate classifiers for each label type
        self.challenge_classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.approach_classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.outcome_classifier = nn.Linear(self.bert.config.hidden_size, 1)

        self._init_weights()
        self.to(self.config.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_masks: Optional[torch.Tensor] = None,
        labels: Optional[dict] = None,
    ) -> dict:
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        sequence_output = outputs.last_hidden_state

        if sentence_masks is not None:
            sentence_embeddings = self._aggregate_sentence_embeddings(
                sequence_output, sentence_masks, attention_mask
            )
        else:
            # Use [CLS] token for sequence classification
            sentence_embeddings = sequence_output[:, 0, :].unsqueeze(1)

        sentence_embeddings = self.dropout(sentence_embeddings)

        challenge_logits = self.challenge_classifier(sentence_embeddings).squeeze(-1)
        approach_logits = self.approach_classifier(sentence_embeddings).squeeze(-1)
        outcome_logits = self.outcome_classifier(sentence_embeddings).squeeze(-1)

        outputs = {
            "challenge_logits": challenge_logits,
            "approach_logits": approach_logits,
            "outcome_logits": outcome_logits,
        }

        if labels is not None:
            loss = self._calculate_loss(
                challenge_logits=challenge_logits,
                approach_logits=approach_logits,
                outcome_logits=outcome_logits,
                labels=labels,
            )
            outputs["loss"] = loss

        return outputs

    def _aggregate_sentence_embeddings(
        self,
        sequence_output: torch.Tensor,
        sentence_masks: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate token embeddings to sentence-level embeddings.
        """
        batch_size, num_sentences, seq_length = sentence_masks.shape
        hidden_size = sequence_output.shape[-1]

        attention_mask_expanded = attention_mask.unsqueeze(1).expand(
            batch_size, num_sentences, seq_length
        )

        combined_mask = sentence_masks * attention_mask_expanded

        sequence_output_expanded = sequence_output.unsqueeze(1).expand(
            batch_size, num_sentences, seq_length, hidden_size
        )

        masked_embeddings = sequence_output_expanded * combined_mask.unsqueeze(-1)
        summed_embeddings = masked_embeddings.sum(dim=2)

        valid_tokens = combined_mask.sum(dim=2, keepdim=True).clamp(min=1)
        sentence_embeddings = summed_embeddings / valid_tokens

        return sentence_embeddings

    def _calculate_loss(
        self,
        challenge_logits: torch.Tensor,
        approach_logits: torch.Tensor,
        outcome_logits: torch.Tensor,
        labels: dict,
    ) -> torch.Tensor:
        challenge_logits_flat = challenge_logits.view(-1)
        approach_logits_flat = approach_logits.view(-1)
        outcome_logits_flat = outcome_logits.view(-1)

        challenge_labels_flat = labels["challenge"].view(-1)
        approach_labels_flat = labels["approach"].view(-1)
        outcome_labels_flat = labels["outcome"].view(-1)

        valid_mask = challenge_labels_flat != -100

        if valid_mask.sum() > 0:
            if self.config.class_weights:
                challenge_loss_fct = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([self.config.class_weights[0]]).to(
                        self.config.device
                    )
                )
                approach_loss_fct = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([self.config.class_weights[1]]).to(
                        self.config.device
                    )
                )
                outcome_loss_fct = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([self.config.class_weights[2]]).to(
                        self.config.device
                    )
                )
            else:
                challenge_loss_fct = nn.BCEWithLogitsLoss()
                approach_loss_fct = nn.BCEWithLogitsLoss()
                outcome_loss_fct = nn.BCEWithLogitsLoss()

            challenge_loss = challenge_loss_fct(
                challenge_logits_flat[valid_mask],
                challenge_labels_flat[valid_mask].float(),
            )
            approach_loss = approach_loss_fct(
                approach_logits_flat[valid_mask],
                approach_labels_flat[valid_mask].float(),
            )
            outcome_loss = outcome_loss_fct(
                outcome_logits_flat[valid_mask], outcome_labels_flat[valid_mask].float()
            )

            total_loss = (challenge_loss + approach_loss + outcome_loss) / 3.0
        else:
            total_loss = torch.tensor(
                0.0, device=self.config.device, requires_grad=True
            )

        return total_loss

    def _init_weights(self) -> None:
        for classifier in [
            self.challenge_classifier,
            self.approach_classifier,
            self.outcome_classifier,
        ]:
            nn.init.xavier_uniform_(classifier.weight)
            nn.init.constant_(classifier.bias, 0)
