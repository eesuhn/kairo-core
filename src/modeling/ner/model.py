import justsdk

from torch import nn
from span_marker import SpanMarkerModel
from config._constants import MODEL_DIR, REPORTS_DIR, CONFIG_DIR
from dataclasses import dataclass, field


REPORTS_MODEL_DIR = REPORTS_DIR / "model"


@dataclass
class NerModelConfig:
    model_name: str = "tomaarsen/span-marker-roberta-large-ontonotes5"
    cache_dir = str(MODEL_DIR / model_name)
    hidden_size: int = 512
    dropout: float = 0.1

    base_labels: list = field(default_factory=list)
    new_labels: list = field(default_factory=list)
    all_labels: list = field(default_factory=list)

    def __post_init__(self) -> None:
        local_base_model_config = justsdk.read_file(
            REPORTS_DIR / "model" / "ner" / "base-model-config.json"
        )
        id2label: dict = local_base_model_config.get("id2label", {})
        self.base_labels = list(id2label.values())

        model_config = justsdk.read_file(CONFIG_DIR / "ner" / "model-config.yml")
        self.new_labels.extend(model_config.get("new_academic_labels", []))

        self.all_labels = self.base_labels + self.new_labels


class NerModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nmc = NerModelConfig()
        self.base_model = self._init_ner_model()
        self._freeze_base_model()

        # NerModelHelper._save_ner_base_model_config(self.base_model)
        # NerModelHelper._save_ner_base_model_encoder_config(self.base_model)

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

    def _freeze_base_model(self) -> None:
        justsdk.print_info(f"Freezing params: {self.nmc.model_name}")
        for p in self.base_model.parameters():
            p.requires_grad_(False)

    def _get_ner_feature_projection(self) -> nn.Sequential:
        try:
            encoder_hidden_size = self.base_model.encoder.config.hidden_size
            target_hidden_size = self.nmc.hidden_size
            seq = nn.Sequential(
                nn.Linear(encoder_hidden_size, target_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.nmc.dropout),
                nn.Linear(target_hidden_size, target_hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.nmc.dropout),
            )
            justsdk.print_success("Created NER feature projection")
            return seq
        except Exception as e:
            raise RuntimeError(f"Failed to get NER feature projection: {e}")


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
        justsdk.print_info(f"Config written to: {output_path}")

    @staticmethod
    def _save_ner_base_model_encoder_config(base_model: SpanMarkerModel) -> None:
        output_path = REPORTS_MODEL_DIR / "ner" / "base-model-encoder-config.json"
        justsdk.write_file(
            base_model.encoder.config.to_dict(),
            file_path=output_path,
            use_orjson=True,
            atomic=True,
        )
        justsdk.print_info(f"Config written to: {output_path}")


if __name__ == "__main__":
    nm = NerModel()
