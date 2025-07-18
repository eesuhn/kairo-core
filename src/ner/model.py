import justsdk

from torch import nn
from span_marker import SpanMarkerModel
from config._constants import MODEL_DIR, REPORTS_DIR, CONFIG_DIR
from dataclasses import dataclass, field


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
        self.unified_labels = self._create_unified_labels()

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

    def _create_unified_labels(self) -> list:
        """
        Create a unified set of labels from dataset labels
        """
        unified_labels = set()
        unified_labels.add("O")  # Outside label
        for _, labels in self.nmc.dataset_labels.items():
            for label in labels:
                label: str
                if label == "O":
                    continue
                if label.startswith(("B-", "I-")):
                    entity_type = label[2:]
                    unified_labels.add(f"B-{entity_type}")
                    unified_labels.add(f"I-{entity_type}")
                else:
                    unified_labels.add(f"B-{label.upper()}")
                    unified_labels.add(f"I-{label.upper()}")
        unified_labels = sorted(unified_labels)

        self.label2id = {label: idx for idx, label in enumerate(unified_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        return unified_labels


class NerDecisionLayer(nn.Module):
    def __init__(self, hidden_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


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
    pass


if __name__ == "__main__":
    main()
