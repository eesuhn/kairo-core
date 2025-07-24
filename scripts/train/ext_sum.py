import justsdk
import numpy as np
import torch
import argparse

from src.summary.extractive.config import ExtSumConfig
from src.summary.extractive.trainer import ExtSumTrainer


class ExtSumTrainScript:
    def __init__(self) -> None:
        # Set random seeds
        seed = ExtSumConfig.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.args = self._init_parser().parse_args()

    def run(self) -> None:
        config = ExtSumConfig(
            use_wandb=self.args.use_wandb,
            upload_model_wandb=self.args.upload_model_wandb,
        )
        trainer = ExtSumTrainer(config)
        try:
            trainer.train()
        except KeyboardInterrupt:
            print("Training interrupted...")
            trainer.save_model()
        except Exception as e:
            justsdk.print_error(f"Training failed: {e}")
            raise

    def _init_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Train Extractive Summarization Model"
        )

        # parser.add_argument(
        #     "--batch-size",
        #     type=int,
        #     default=32,
        #     help="Batch size for training.",
        # )
        # parser.add_argument(
        #     "--epochs",
        #     type=int,
        #     default=10,
        #     help="Number of epochs for training.",
        # )
        # parser.add_argument(
        #     "--learning-rate",
        #     type=float,
        #     default=1e-4,  # NOTE: Maybe default to `5e-5`?
        #     help="Learning rate for the optimizer.",
        # )
        parser.add_argument(
            "--use-wandb",
            action="store_true",
            help="Use Weights & Biases for logging.",
        )
        parser.add_argument(
            "--upload-model-wandb",
            action="store_true",
            help="Upload the trained model to Weights & Biases.",
        )
        return parser


if __name__ == "__main__":
    ests = ExtSumTrainScript()
    ests.run()
