import torch

from dataclasses import dataclass
from configs._constants import MODEL_DIR
from pathlib import Path


@dataclass
class ExtSumConfig:
    base_model_name: str = "allenai/scibert_scivocab_uncased"
    num_labels: int = 3  # NOTE: challenge, approach, outcome
    hidden_dropout_prob: float = 0.1
    classifier_dropout: float = 0.2

    # Training
    batch_size: int = 16
    epochs: int = 15
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Evaluation
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    eval_batch_size: int = 32

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001

    # Data processing
    max_length: int = 512  # Maximum sequence length
    max_sentences: int = 100  # Maximum sentences per document
    stride: int = 128  # Stride for sliding window if needed

    # Loss weights for imbalanced classes
    class_weights: list = None  # Will be computed from data

    # Output
    model_dir: Path = MODEL_DIR / "summary" / "extractive"

    # Thresholds for prediction
    challenge_threshold: float = 0.5
    approach_threshold: float = 0.5
    outcome_threshold: float = 0.5
    min_sentences_per_type: int = 1

    # Others
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    quiet: bool = False
    use_wandb: bool = False
    upload_model_wandb: bool = False
    num_workers: int = 2
