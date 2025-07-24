import torch
import justsdk
import os
import sys
import wandb
import signal

from .config import AbsSumConfig
from src.inter_data_handler import InterDataHandler
from torch.utils.data import Dataset, DataLoader
from .model import AbsSumModel
from transformers import T5TokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from torch import nn
from pathlib import Path
from rouge_score import rouge_scorer
from configs._constants import REPORTS_DIR


REPORTS_SUMMARY_DIR = REPORTS_DIR / "summary" / "abstractive"


class AbsSumDataset(Dataset):
    def __init__(
        self,
        data: list,
        tokenizer: T5TokenizerFast,
        config: AbsSumConfig,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        item = self.data[index]
        document = item["document"]
        challenge = item["challenge"]
        approach = item["approach"]
        outcome = item["outcome"]

        summary: list = []
        if challenge:
            summary.append(challenge)
        if approach:
            summary.append(approach)
        if outcome:
            summary.append(outcome)

        summary = "\n".join(summary)
        inputs = self.tokenizer(
            document,
            truncation=True,
            padding="max_length",
            max_length=self.config.data_max_length,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            summary,
            truncation=True,
            padding="max_length",
            max_length=self.config.data_max_length,
            return_tensors="pt",
        )

        labels = targets["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            # NOTE: These are torch.Tensor
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


class AbsSumTrainer:
    TARGET_DATASETS = ["sobamchan/aclsum-abstractive"]

    def __init__(self, config: AbsSumConfig) -> None:
        self.config = config
        self.idh = InterDataHandler(quiet=self.config.quiet)

        if self.config.device != "cuda":
            # NOTE: Disable parallelism for tokenizers to avoid issues with CPU
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not self.config.model_dir.exists():
            self.config.model_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir: Path = self.config.model_dir / "checkpoints"
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if not REPORTS_SUMMARY_DIR.exists():
            REPORTS_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

        if self.config.use_wandb:
            justsdk.print_info("Initializing wandb...")
            wandb.init(
                project="kairo-core",
                config=vars(self.config),
                dir=REPORTS_SUMMARY_DIR,
                name=f"abs_sum_{self.config.base_model_name.replace('/', '_')}",
            )

        if self.config.upload_model_wandb:
            justsdk.print_info("Enabling model upload to wandb...")

        self.model = AbsSumModel(self.config)
        self.optimizer = self._create_optimizer()

        self.pin_memory: bool = True if self.config.device == "cuda" else False
        self.global_step: int = 0
        self.interrupted: bool = False
        self.best_rougeL: float = 0.0
        self.patience_count: int = 0

        # Listen for interrupts
        signal.signal(signal.SIGINT, self._signal_handler)

    def train(self) -> None:
        self._remap_ds()
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.pin_memory,
        )
        num_training_steps = len(train_dl) * self.config.epochs
        self.scheduler = self._create_scheduler(num_training_steps)

        justsdk.print_info("Starting training...", newline_before=True)
        justsdk.print_info(f"  No. of examples: {len(self.train_ds)}")
        justsdk.print_info(f"  No. of epochs: {self.config.epochs}")
        justsdk.print_info(f"  Batch size: {self.config.batch_size}")
        justsdk.print_info(f"  Total training steps: {num_training_steps}")

        if self.config.use_wandb:
            wandb.log(
                {
                    "train/total_examples": len(self.train_ds),
                    "train/total_epochs": self.config.epochs,
                    "train/batch_size": self.config.batch_size,
                    "train/total_steps": num_training_steps,
                }
            )

        scaler = torch.amp.GradScaler(device=self.config.device)

        # Training loop...
        for epoch in range(self.config.epochs):
            if self.interrupted:
                break
            self.model.train()

            epoch_loss = 0.0
            progress_bar = tqdm(
                train_dl, desc=f"Epoch {epoch + 1}/{self.config.epochs}"
            )
            for step, batch in enumerate(progress_bar):
                if self.interrupted:
                    break

                # NOTE: Move batch to CUDA if available
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                # Forward pass
                with torch.amp.autocast(device_type=self.config.device):
                    outputs = self.model(**batch)
                    loss: torch.Tensor = outputs["loss"]

                # Backward pass
                scaler.scale(loss).backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                self.global_step += 1

                # Log metrics
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    progress_bar.set_postfix(loss=avg_loss)

                    if self.config.use_wandb:
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": self.scheduler.get_last_lr()[0],
                                "train/global_step": self.global_step,
                                "train/epoch": epoch + 1,
                            },
                            step=self.global_step,
                        )

                # Evaluate
                if self.val_ds and self.global_step % self.config.eval_steps == 0:
                    eval_res = self.evaluate()

                    # Check for early stopping
                    if self._check_early_stop(eval_res["rougeL_f1"]):
                        return

                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

            if self.val_ds and not self.interrupted:
                justsdk.print_info(f"End of epoch {epoch + 1}, evaluating...")
                self.evaluate()

        if self.interrupted:
            self._save_checkpoint()
            justsdk.print_success("Training interrupted, saving state...")
            if self.config.use_wandb:
                wandb.finish()
            sys.exit(0)  # A bit brutal but effective

        if self.config.use_wandb:
            self._save_wandb_artifacts()
            wandb.finish()

    def evaluate(self) -> dict:
        eval_dl = DataLoader(
            self.val_ds,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.pin_memory,
        )
        self.model.eval()
        eval_loss = 0.0

        predictions = []
        references = []

        with torch.no_grad():
            for batch in tqdm(eval_dl, desc="Evaluating..."):
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                if "loss" in outputs:
                    eval_loss += outputs["loss"].item()

                # Generate predictions for ROUGE
                generated_ids = self.model.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=self.config.eval_max_length,
                    num_beams=self.config.eval_num_beams,
                    early_stopping=True,
                )

                # Decode predictions and references
                preds = self.model.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                refs = self.model.tokenizer.batch_decode(
                    batch["labels"].masked_fill(
                        batch["labels"] == -100, self.model.tokenizer.pad_token_id
                    ),
                    skip_special_tokens=True,
                )

                predictions.extend(preds)
                references.extend(refs)

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)

        avg_rouge = {k: sum(v) / len(v) for k, v in rouge_scores.items()}
        avg_loss = eval_loss / len(eval_dl)

        res = {
            "loss": avg_loss,
            "rouge1_f1": avg_rouge["rouge1"],
            "rouge2_f1": avg_rouge["rouge2"],
            "rougeL_f1": avg_rouge["rougeL"],
        }

        justsdk.print_info("Evaluation results", newline_before=True)
        justsdk.print_info(f"  Loss: {res['loss']:.4f}")
        justsdk.print_info(f"  ROUGE-1 F1: {res['rouge1_f1']:.4f}")
        justsdk.print_info(f"  ROUGE-2 F1: {res['rouge2_f1']:.4f}")
        justsdk.print_info(f"  ROUGE-L F1: {res['rougeL_f1']:.4f}")

        if self.config.use_wandb:
            wandb.log(
                {
                    "eval/loss": res["loss"],
                    "eval/rouge1_f1": res["rouge1_f1"],
                    "eval/rouge2_f1": res["rouge2_f1"],
                    "eval/rougeL_f1": res["rougeL_f1"],
                    "eval/global_step": self.global_step,
                },
                step=self.global_step,
            )

        return res

    def _signal_handler(self, signum, frame) -> None:
        justsdk.print_warning("Training interrupted...")
        self.interrupted = True

    def _check_early_stop(self, current_rougeL: float) -> bool:
        """
        Check if early stopping criteria are met.
        """
        if current_rougeL > self.best_rougeL + self.config.early_stopping_delta:
            self.best_rougeL = current_rougeL
            self.patience_count = 0
            self._save_checkpoint(is_best=True)

            if self.config.use_wandb:
                wandb.log(
                    {
                        "train/best_rougeL": self.best_rougeL,
                        "train/patience_count": self.patience_count,
                    },
                    step=self.global_step,
                )
            return False
        else:
            self.patience_count += 1
            if self.config.use_wandb:
                wandb.log(
                    {"train/patience_count": self.patience_count}, step=self.global_step
                )

            if self.patience_count >= self.config.early_stopping_patience:
                justsdk.print_info("Early stopping triggered")
                if self.config.use_wandb:
                    wandb.log({"train/early_stopped": True}, step=self.global_step)
                return True
            return False

    def _save_checkpoint(self, is_best: bool = False) -> None:
        if self.interrupted:
            filepath = Path(
                self.checkpoint_dir / f"interrupted_checkpoint_{self.global_step}.pt"
            )
        elif is_best:
            filepath = Path(self.config.model_dir / "model.pt")
        else:
            filepath = Path(self.checkpoint_dir / f"checkpoint_{self.global_step}.pt")

        torch.save(self.model.state_dict(), filepath)
        justsdk.print_success(f"Checkpoint saved to {filepath}")

        if self.config.use_wandb and is_best:
            wandb.log(
                {"checkpoint/best_rougeL": self.best_rougeL}, step=self.global_step
            )

    def _save_wandb_artifacts(self) -> None:
        if not self.config.use_wandb:
            return

        try:
            model_artifact = wandb.Artifact(name="kairo_abs_sum", type="model")

            # NOTE: Not necessary to upload the model
            if (
                self.config.use_wandb
                and self.config.upload_model_wandb
                and (self.config.model_dir / "model.pt").exists()
            ):
                model_artifact.add_file(
                    str(self.config.model_dir / "model.pt"),
                )

            wandb.log_artifact(model_artifact)
            justsdk.print_success(f"wandb artifacts saved to {REPORTS_SUMMARY_DIR}")

        except Exception as e:
            justsdk.print_warning(f"Failed to save wandb artifacts: {e}")

    def _create_optimizer(self) -> AdamW:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

    def _remap_ds(self) -> tuple:
        combined_train: list = []
        combined_val: list = []
        combined_test: list = []

        justsdk.print_info("Loading dataset...")
        for ds_name in self.TARGET_DATASETS:
            ds_dict = self.idh.load_dataset(ds_name, quiet=True)

            for split, target in [
                ("train", combined_train),
                ("validation", combined_val),
                ("test", combined_test),
            ]:
                for item in ds_dict[split]:
                    remapped = {
                        "document": item["document"],
                        # XXX: Does this even fit???
                        "challenge": item["challenge"],
                        "approach": item["approach"],
                        "outcome": item["outcome"],
                    }
                    target.append(remapped)

        self.train_ds = AbsSumDataset(combined_train, self.model.tokenizer, self.config)
        self.val_ds = AbsSumDataset(combined_val, self.model.tokenizer, self.config)
        self.test_ds = AbsSumDataset(combined_test, self.model.tokenizer, self.config)

    def _create_scheduler(self, num_training_steps: int):
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )
