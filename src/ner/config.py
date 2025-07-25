import torch

from dataclasses import dataclass
from configs._constants import MODEL_DIR
from pathlib import Path


@dataclass
class NerConfig:
    base_model_name: str = "google-bert/bert-base-uncased"
    freeze_bert: bool = False

    # Training
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 5e-5
    dropout_rate: float = 0.1
    num_workers: int = 8  # NOTE: Tune it down if you using a toaster
    weight_decay: float = 0.01
    warmup_steps: int = 500  # NOTE: Should set to 10% of total training steps
    max_grad_norm: float = 1.0  # Default for gradient clipping

    # Evaluation
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000

    # Early stopping
    early_stopping_delta: float = 0.001  # Early stopping threshold
    early_stopping_patience: int = 3

    # Data processing
    ignore_index: int = -100
    max_length: int = 512  # TODO: Make this dynamic based on input lengths

    # Output
    model_dir: Path = MODEL_DIR / "ner"
    deduplicate_entities: bool = True
    similarity_threshold: float = 0.8

    # Predict
    confidence_threshold: float = 0.0
    return_confidence: bool = True
    aggregate_subtokens: bool = True

    # Others
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_wandb: bool = False
    upload_model_wandb: bool = False
    quite: bool = False
