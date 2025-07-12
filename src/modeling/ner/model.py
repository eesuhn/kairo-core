from torch import nn
from span_marker import SpanMarkerModel
from configs._constants import MODEL_DIR


class NerModel(nn.Module):
    NER_MODEL_NAME = "tomaarsen/span-marker-roberta-large-ontonotes5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = SpanMarkerModel.from_pretrained(
            pretrained_model_name_or_path=self.NER_MODEL_NAME,
            cache_dir=str(MODEL_DIR / self.NER_MODEL_NAME),
        )


if __name__ == "__main__":
    nm = NerModel()
