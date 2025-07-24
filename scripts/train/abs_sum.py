import numpy as np
import torch

from src.summary.abstractive.config import AbsSumConfig
from src.summary.abstractive.trainer import AbsSumTrainer


class AbsSumTrainScript:
    def __init__(self) -> None:
        # Set random seeds
        seed = AbsSumConfig.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self) -> None:
        config = AbsSumConfig()
        trainer = AbsSumTrainer(config)
        trainer.train()


if __name__ == "__main__":
    asts = AbsSumTrainScript()
    asts.run()
