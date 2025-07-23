import torch
import numpy as np
import argparse

from src.ner.trainer import NerConfig, NerTrainer


class NerScript:
    def __init__(self) -> None:
        # Set random seeds
        seed = NerConfig.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():  # NOTE: Check CUDA
            torch.cuda.manual_seed_all(seed)

        self.args = self._init_parser().parse_args()

    def run(self) -> None:
        config = NerConfig(
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            learning_rate=self.args.learning_rate,
        )
        trainer = NerTrainer(config)
        trainer.train()

    def _init_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Train NER model")
        parser.add_argument(
            "--mode",
            type=str,
            required=True,
            choices=["new", "resume"],
            help="'new' for a new training session, 'resume' to continue from a checkpoint.",
        )

        # TODO: Support 'resume' mode
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Path to the checkpoint file to resume training from.",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default=None,
            help="Path to the dataset file to resume training from.",
        )

        # General options
        parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size for training.",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=10,
            help="Number of epochs for training.",
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=1e-4,  # NOTE: Maybe default to `5e-5`?
            help="Learning rate for the optimizer.",
        )
        return parser


if __name__ == "__main__":
    ns = NerScript()
    ns.run()
