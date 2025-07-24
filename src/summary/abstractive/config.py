import torch

from dataclasses import dataclass
from configs._constants import MODEL_DIR
from pathlib import Path


@dataclass
class AbsSumConfig:
    base_model_name: str = "google-t5/t5-base"
    freeze_t5_encoder: bool = True

    # Training
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 5e-5
    dropout_rate: float = 0.1
    num_workers: int = 8
    weight_decay: float = 0.01
    warmup_steps: int = 500  # NOTE: Should set to 10% of total training steps
    max_grad_norm: float = 1.0  # Default for gradient clipping

    # Evaluation
    logging_steps: int = 2
    eval_steps: int = 20
    save_steps: int = 1000

    # Early stopping
    early_stopping_delta: float = 0.001  # Early stopping threshold
    early_stopping_patience: int = 3

    # Data processing
    max_length: int = 128

    # Output
    model_dir: Path = MODEL_DIR / "summary" / "abstractive"

    # Others
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    quite: bool = False
