import torch

from dataclasses import dataclass
from configs._constants import MODEL_DIR
from pathlib import Path


@dataclass
class AbsSumConfig:
    base_model_name: str = "google-t5/t5-base"
    freeze_t5_encoder: bool = True

    # Training
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 3e-5
    dropout_rate: float = 0.1
    num_workers: int = 2
    weight_decay: float = 0.01
    warmup_steps: int = 100  # NOTE: Should set to 10% of total training steps
    max_grad_norm: float = 0.5  # Default for gradient clipping

    # Evaluation
    logging_steps: int = 2
    eval_steps: int = 20
    save_steps: int = 1000
    eval_batch_size: int = 8
    eval_max_length: int = 128
    eval_num_beams: int = 2

    # Early stopping
    early_stopping_delta: float = 0.001  # Early stopping threshold
    early_stopping_patience: int = 3

    # Data processing
    data_max_length: int = 128

    # Output
    model_dir: Path = MODEL_DIR / "summary" / "abstractive"

    # Generation
    length_penalty: float = 1.4
    repetition_penalty: float = 1.6
    no_repeat_ngram_size: int = 3
    early_stopping: bool = False
    gen_batch_size: int = 8
    gen_max_length: int = 200  # TODO: Make this dynamic based on input length
    gen_min_length: int = 20  # TODO: Make this dynamic based on input length
    gen_num_beams: int = 4

    # Others
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    quiet: bool = False
    use_wandb: bool = False
    upload_model_wandb: bool = False
