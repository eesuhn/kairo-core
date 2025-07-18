import justsdk

from torch import nn
from span_marker import SpanMarkerModel
from config._constants import MODEL_DIR, REPORTS_DIR, CONFIG_DIR
from dataclasses import dataclass, field
from .helper import NerHelper


REPORTS_MODEL_DIR = REPORTS_DIR / "model"


@dataclass
class NerModelConfig:
    # Base model
    model_name: str = "tomaarsen/span-marker-roberta-large-ontonotes5"
    cache_dir = str(MODEL_DIR / model_name)

    # Training params
    freeze_encoder: bool = True

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
        self.encoder = self.base_model.encoder

        # Encoder
        self._freeze_encoder(show_info=True)
        self.encoder_feature_dimension = self._get_encoder_feature_dimension()

        # Labels
        self._create_uni_ds_labels()
        self._create_label_mapping()

        if self.save_config:
            NerModelHelper._save_ner_base_model_config(self.base_model)
            NerModelHelper._save_ner_base_model_encoder_config(self.base_model)

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
        justsdk.print_debug("uni_ds_enc_id2label")
        justsdk.print_data(self.uni_ds_enc_id2label)
        justsdk.print_debug("unified_ds_to_base_map")
        justsdk.print_data(self.unified_ds_to_base_map)
        self.base_to_dataset_map = NerHelper.map_base_to_dataset(
            self.nmc.base_labels, self.nmc.dataset_labels
        )
        justsdk.print_debug("base_to_dataset_map")
        justsdk.print_data(self.base_to_dataset_map)


class NerModelHelper:
    @staticmethod
    def _save_ner_base_model_config(base_model: SpanMarkerModel) -> None:
        output_path = REPORTS_MODEL_DIR / "ner" / "base-model-config.json"
        justsdk.write_file(
            base_model.config.to_dict(),
            file_path=output_path,
            use_orjson=True,
            atomic=True,
        )
        justsdk.print_info(f"Config written to {output_path}")

    @staticmethod
    def _save_ner_base_model_encoder_config(base_model: SpanMarkerModel) -> None:
        output_path = REPORTS_MODEL_DIR / "ner" / "base-model-encoder-config.json"
        justsdk.write_file(
            base_model.encoder.config.to_dict(),
            file_path=output_path,
            use_orjson=True,
            atomic=True,
        )
        justsdk.print_info(f"Config written to {output_path}")


def main() -> None:
    NerModel()


if __name__ == "__main__":
    main()
