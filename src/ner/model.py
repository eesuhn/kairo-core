import justsdk
import torch

from torch import nn
from span_marker import SpanMarkerModel
from config._constants import MODEL_DIR, REPORTS_DIR, CONFIG_DIR
from dataclasses import dataclass, field
from .helper import NerHelper
from typing import Optional
from transformers import PreTrainedModel
from pathlib import Path


@dataclass
class NerModelConfig:
    # Base model
    model_name: str = "tomaarsen/span-marker-roberta-large-ontonotes5"
    cache_dir = str(MODEL_DIR / model_name)

    # Training params
    freeze_encoder: bool = True
    xavier_uniform_gain: float = 0.1

    # Decision layer
    hidden_size: int = 512
    dropout: float = 0.1

    # Labels
    base_labels: dict = field(default_factory=dict)
    dataset_labels: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        base_model_config = justsdk.read_file(
            REPORTS_DIR / "model" / "ner" / "base-model-config.json"
        )
        model_config = justsdk.read_file(CONFIG_DIR / "ner" / "model.yml")

        self.base_labels = base_model_config.get("label2id", {})

        raw_dataset_labels = model_config.get("dataset_labels", {})
        self.dataset_labels = {
            name: labels
            for dataset in raw_dataset_labels
            for name, labels in dataset.items()
        }


class NerModel(nn.Module):
    save_config: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nmc = NerModelConfig()
        self.base_model = self._init_ner_model()
        self.encoder: PreTrainedModel = self.base_model.encoder

        # Encoder
        self._freeze_encoder(show_info=True)
        self.encoder_feature_dimension = self._get_encoder_feature_dimension()

        # Labels
        self._create_uni_ds_labels()
        self._create_label_mapping()

        self.decision_layer = NerDecisionLayer(
            encoder_feature_dimension=self.encoder_feature_dimension,
            num_uni_ds_enc_labels=len(self.uni_ds_enc_labels),
        )

        self._init_projection_matrix()

        if self.save_config:
            NerHelper._save_ner_base_model_config(self.base_model)
            NerHelper._save_ner_base_model_encoder_config(self.base_model)

    def _init_ner_model(self) -> SpanMarkerModel:
        try:
            base_model = SpanMarkerModel.from_pretrained(
                pretrained_model_name_or_path=self.nmc.model_name,
                cache_dir=self.nmc.cache_dir,
            )
            justsdk.print_success(f"Init {self.nmc.model_name}", newline_before=True)
            return base_model
        except Exception as e:
            raise RuntimeError(f"Failed to init {self.nmc.model_name}: {e}")

    def _freeze_encoder(self, show_info: bool = False) -> None:
        """
        Freeze the encoder parameters of the base model

        Args:
            show_info: Whether to print info about frozen parameters
        """
        if not self.nmc.freeze_encoder:
            justsdk.print_warning("Encoder freezing disabled")
            return
        frozen_params: int = 0
        justsdk.print_info("Freezing encoder")
        for param in self.encoder.parameters():
            param.requires_grad_(False)
            frozen_params += param.numel()
        if show_info:
            justsdk.print_info(f"Frozen {frozen_params:,} params in encoder")

    def _get_encoder_feature_dimension(self) -> int:
        """
        Get the feature dimension of the base model's encoder
        """
        try:
            encoder_hidden_size = self.base_model.encoder.config.hidden_size
            justsdk.print_info(f"Encoder hidden size: {encoder_hidden_size}")
            return encoder_hidden_size
        except Exception as e:
            raise RuntimeError(f"Failed to get feature dimension: {e}")

    def _create_uni_ds_labels(self) -> None:
        """
        Create unified dataset labels for both encoder and base model

        Add labels in BIO format for the encoder
        """
        uni_enc_labels = {"O"}
        uni_labels = {"O"}

        for labels in self.nmc.dataset_labels.values():
            for label in labels:
                if label == "O":
                    continue
                entity_type = (
                    label[2:].upper()
                    if label.startswith(("B-", "I-"))
                    else label.upper()
                )
                uni_enc_labels.update(
                    [f"B-{entity_type}", f"I-{entity_type}"]
                )  # Add BIO format for encoder
                uni_labels.add(entity_type)

        # NOTE: Outside label will be placed at the largest index once sorted

        # Encoder
        self.uni_ds_enc_labels = sorted(uni_enc_labels)
        self.uni_ds_enc_label2id = {
            label: idx for idx, label in enumerate(self.uni_ds_enc_labels)
        }
        self.uni_ds_enc_id2label = {
            idx: label for label, idx in self.uni_ds_enc_label2id.items()
        }

        # Base
        self.uni_ds_labels = sorted(uni_labels)
        self.uni_ds_label2id = {
            label: idx for idx, label in enumerate(self.uni_ds_labels)
        }
        self.uni_ds_id2label = {
            idx: label for label, idx in self.uni_ds_label2id.items()
        }

    def _create_label_mapping(self) -> None:
        self.unified_ds_to_base_map = NerHelper.map_unified_ds_to_base_labels(
            self.uni_ds_enc_labels, self.nmc.base_labels
        )
        self.base_to_dataset_map = NerHelper.map_base_to_dataset(
            self.nmc.base_labels, self.nmc.dataset_labels
        )

    def _init_projection_matrix(self) -> None:
        projection = torch.zeros(len(self.nmc.base_labels), len(self.uni_ds_enc_labels))
        for base_id, uni_ids in self.unified_ds_to_base_map.items():
            if uni_ids:
                weight = 1.0 / len(uni_ids)
                for uid in uni_ids:
                    projection[base_id, uid] = weight

        # Update the decision layer's projection matrix
        with torch.no_grad():
            self.decision_layer.label_projection.data = projection
        justsdk.print_info("Init projection matrix for decision layer")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_base_predictions: bool = False,
        **kwargs,
    ) -> nn.Module:
        encoder_out: PreTrainedModel = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        seq_out = encoder_out.last_hidden_state
        out: nn.Module = self.decision_layer(
            seq_out, return_base_predictions=return_base_predictions
        )
        return out


class NerDecisionLayer(nn.Module):
    def __init__(
        self,
        encoder_feature_dimension: int,
        num_uni_ds_enc_labels: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.nmc = NerModelConfig()
        num_base_labels = len(self.nmc.base_labels)

        self.base_classifier = nn.Sequential(
            nn.Linear(encoder_feature_dimension, self.nmc.hidden_size),
            nn.LayerNorm(self.nmc.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.nmc.dropout),
            nn.Linear(self.nmc.hidden_size, num_base_labels),
        )

        refine_dim = encoder_feature_dimension + num_base_labels
        self.refine_layer = nn.Sequential(
            nn.Linear(refine_dim, self.nmc.hidden_size),
            nn.LayerNorm(self.nmc.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.nmc.dropout),
            nn.Linear(self.nmc.hidden_size, num_uni_ds_enc_labels),
        )

        self.label_projection = nn.Parameter(
            torch.zeros(num_base_labels, num_uni_ds_enc_labels)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.base_classifier, self.refine_layer]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(
            self.label_projection, gain=self.nmc.xavier_uniform_gain
        )

    def forward(
        self,
        encoder_out: torch.Tensor,
        return_base_predictions: bool = False,
    ) -> torch.Tensor:
        base_logits: torch.Tensor = self.base_classifier(encoder_out)
        base_probs = torch.softmax(base_logits, dim=-1)
        combined_feat = torch.cat([encoder_out, base_probs], dim=-1)

        uni_logits = self.refine_layer(combined_feat)
        projected_base = torch.matmul(base_probs, self.label_projection)
        uni_logits += projected_base

        if return_base_predictions:
            return uni_logits, base_logits
        return uni_logits


class NerModelSaver:
    @staticmethod
    def save_model_checkpoint(
        model: "NerModel",
        cache_dir: Path,
        epoch: Optional[int] = None,
        optimizer_state: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        cache_path = MODEL_DIR / "ner-span-marker" / cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "model_name": model.nmc.model_name,
                "hidden_size": model.nmc.hidden_size,
                "dropout": model.nmc.dropout,
                "freeze_encoder": model.nmc.freeze_encoder,
            },
            "label_mappings": {
                "unified_labels": model.uni_ds_enc_labels,
                "encoder_label2id": model.uni_ds_enc_label2id,
                "encoder_id2label": model.uni_ds_enc_id2label,
                "base_labels": model.nmc.base_labels,
                "dataset_labels": model.nmc.dataset_labels,
                "unified_to_base_map": model.unified_ds_to_base_map,
                "base_to_dataset_map": model.base_to_dataset_map,
            },
        }

        if epoch is not None:
            checkpoint["epoch"] = epoch
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
        if metrics is not None:
            checkpoint["metrics"] = metrics

        checkpoint_path = cache_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        justsdk.print_success(f"Checkpoint saved to {checkpoint_path}")

        config_path = cache_path / "config.json"
        justsdk.write_file(
            data=checkpoint,
            file_path=config_path,
            use_orjson=True,
            atomic=True,
        )
        justsdk.print_success(f"Model config saved to {config_path}")

    @staticmethod
    def load_model_checkpoint(
        cache_dir: Path,
    ) -> tuple:
        cache_path = MODEL_DIR / "ner-span-marker" / cache_dir
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(cache_path / "checkpoint.pt", map_location=device)

        nm = NerModel()
        nm.load_state_dict(checkpoint["model_state_dict"])
        nm.to(device)

        justsdk.print_success(f"Model loaded from {cache_path}")
        return nm, checkpoint


def main() -> None:
    nm = NerModel()

    justsdk.print_info(f"Base model: {nm.nmc.model_name}")
    justsdk.print_info(f"Encoder dimension: {nm.encoder_feature_dimension}")

    justsdk.print_info(
        f"No. of base labels: {len(nm.nmc.base_labels)}", newline_before=True
    )
    justsdk.print_data(nm.nmc.base_labels)

    justsdk.print_info(
        f"No. of unified dataset encoder labels: {len(nm.uni_ds_enc_labels)}",
        newline_before=True,
    )
    justsdk.print_data(nm.uni_ds_enc_labels)

    justsdk.print_info("Testing forward pass...", newline_before=True)
    batch_size, seq_length = 2, 128
    dummy_input = torch.randint(0, 1000, (batch_size, seq_length))
    dummy_attention = torch.ones(batch_size, seq_length)

    with torch.no_grad():
        uni_logits = nm(
            input_ids=dummy_input,
            attention_mask=dummy_attention,
        )
        justsdk.print_info(f"Unified logits shape: {uni_logits.shape}")

        uni_logits, base_logits = nm(
            input_ids=dummy_input,
            attention_mask=dummy_attention,
            return_base_predictions=True,
        )
        justsdk.print_info(f"Unified logits shape: {uni_logits.shape}")
        justsdk.print_info(f"Base logits shape: {base_logits.shape}")


if __name__ == "__main__":
    main()
