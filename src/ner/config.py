import torch

from dataclasses import dataclass
from configs._constants import MODEL_DIR
from pathlib import Path


@dataclass
class NerConfig:
    base_model_name: str = "google-bert/bert-base-uncased"
    freeze_bert: bool = False
    freeze_bert_encoder: bool = False

    # Training
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    dropout_rate: float = 0.1
    ignore_index: int = -100
    num_workers: int = 4  # NOTE: Prob. 8 if your device is a chad
    weight_decay: float = 0.01
    warmup_steps: int = 1000  # NOTE: Should set to 10% of total training steps

    # Input
    max_length: int = 128

    # Output
    model_dir: Path = MODEL_DIR / "ner"
    checkpoint_dir: Path = model_dir / "checkpoints"

    # Others
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
