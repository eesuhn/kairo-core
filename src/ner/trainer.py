import sys
import justsdk
import torch
import numpy as np
import signal
import os
import wandb

from ..inter_data_handler import InterDataHandler
from configs._constants import CONFIGS_DIR, REPORTS_DIR
from .model import NerModel
from .config import NerConfig
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torch import nn
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


REPORTS_NER_DIR = REPORTS_DIR / "ner"


class NerTrainer:
    def __init__(self, config: NerConfig) -> None:
        self.config = config
        self.idh = InterDataHandler()

        if self.config.device != "cuda":
            # NOTE: Disable parallelism for tokenizers to avoid issues with CPU
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not self.config.model_dir.exists():
            self.config.model_dir.mkdir(parents=True, exist_ok=True)

        if not self.config.checkpoint_dir.exists():
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Init wandb
        if self.config.use_wandb:
            wandb.init(
                project="kairo-core",
                config=vars(self.config),
                dir=REPORTS_NER_DIR,
                name=f"ner_{self.config.base_model_name.replace('/', '_')}",
            )

        self.all_ds = self.idh.list_datasets_by_category("ner")
        self.uni_rules = justsdk.read_file(CONFIGS_DIR / "ner" / "rules.yml")

        # Prepare labels and remap datasets
        self.uni_labels, self.label_map = self._get_uni_label_map()
        self.label_to_id = {label: i for i, label in enumerate(self.uni_labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.uni_labels)}

        self.model = NerModel(num_labels=len(self.uni_labels))
        self.model.to(self.config.device)

        self.tokenizer = BertTokenizerFast.from_pretrained(self.config.base_model_name)
        self.optimizer = self._create_optimizer()

        # Training state
        self.pin_memory: bool = True if self.config.device == "cuda" else False
        self.global_step = 0
        self.best_f1 = 0.0
        self.interrupted = False
        self.patience_count = 0

        # Listen for interrupts
        signal.signal(signal.SIGINT, self._signal_handler)

        self._remap_ds()

    def train(self, resume: bool = False) -> None:
        train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.pin_memory,
        )
        num_training_steps = len(train_dl) * self.config.epochs
        self.scheduler = self._create_scheduler(num_training_steps)

        # TODO: Support resume training
        if resume:
            pass

        justsdk.print_info("Starting training...", newline_before=True)
        justsdk.print_info(f"  No. of examples: {len(self.train_dataset)}")
        justsdk.print_info(f"  No. of epochs: {self.config.epochs}")
        justsdk.print_info(f"  Batch size: {self.config.batch_size}")
        justsdk.print_info(f"  Total training steps: {num_training_steps}")

        # Log initial metrics
        if self.config.use_wandb:
            wandb.log(
                {
                    "train/total_examples": len(self.train_dataset),
                    "train/total_epochs": self.config.epochs,
                    "train/batch_size": self.config.batch_size,
                    "train/total_steps": num_training_steps,
                }
            )

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
                outputs = self.model(**batch)
                loss: torch.Tensor = outputs["loss"]

                # Backward pass
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

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
                if self.eval_dataset and self.global_step % self.config.eval_steps == 0:
                    eval_results = self.evaluate()

                    # Check for early stopping
                    if self._check_early_stop(eval_results["overall_f1"]):
                        return

                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

            if self.eval_dataset and not self.interrupted:
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
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.pin_memory,
        )
        self.model.eval()
        all_preds = []
        all_labels = []
        eval_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(eval_dl, desc="Evaluating..."):
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                if "loss" in outputs:
                    eval_loss += outputs["loss"].item()

                # NOTE: Might be different if CRF is used
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)
                preds = preds.cpu().numpy()  # Convert to CPU for processing

                labels = batch["labels"].cpu().numpy()
                for pred, label in zip(preds, labels):
                    valid_i = label != NerConfig.ignore_index
                    valid_preds = (
                        pred[valid_i]
                        if isinstance(pred, np.ndarray)
                        else [p for p, v in zip(pred, valid_i) if v]
                    )
                    valid_labels = label[valid_i]

                    all_preds.extend(valid_preds)
                    all_labels.extend(valid_labels)

        avg_loss = eval_loss / len(eval_dl)
        true_labels = [[self.id_to_label[label] for label in all_labels]]
        pred_labels = [[self.id_to_label[pred] for pred in all_preds]]

        res = {
            "loss": avg_loss,
            "overall_f1": f1_score(true_labels, pred_labels),
            "overall_precision": precision_score(true_labels, pred_labels),
            "overall_recall": recall_score(true_labels, pred_labels),
            "classification_report": classification_report(
                true_labels, pred_labels, output_dict=True
            ),
        }

        justsdk.print_info("Evaluation results", newline_before=True)
        justsdk.print_info(f"  Loss: {res['loss']:.4f}")
        justsdk.print_info(f"  F1: {res['overall_f1']:.4f}")
        justsdk.print_info(f"  Precision: {res['overall_precision']:.4f}")
        justsdk.print_info(f"  Recall: {res['overall_recall']:.4f}")

        if self.config.use_wandb:
            wandb.log(
                {
                    "eval/loss": res["loss"],
                    "eval/f1": res["overall_f1"],
                    "eval/precision": res["overall_precision"],
                    "eval/recall": res["overall_recall"],
                    "eval/global_step": self.global_step,
                },
                step=self.global_step,
            )

            if "classification_report" in res:
                for label, metrics in res["classification_report"].items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                wandb.log(
                                    {f"eval/{label}_{metric_name}": value},
                                    step=self.global_step,
                                )

        return res

    def _signal_handler(self, signum, frame) -> None:
        justsdk.print_warning("Training interrupted...")
        self.interrupted = True

    def _save_checkpoint(self, is_best: bool = False) -> None:
        if self.interrupted:
            filename = f"interrupted_checkpoint_{self.global_step}.pt"
        elif is_best:
            filename = "best_model.pt"
        else:
            filename = f"checkpoint_{self.global_step}.pt"

        checkpoint_path = self.config.checkpoint_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_f1": self.best_f1,
            "label_map": self.label_map,
        }
        torch.save(checkpoint, checkpoint_path)
        justsdk.print_success(f"Checkpoint saved to {checkpoint_path}")

        if self.config.use_wandb and is_best:
            wandb.log({"checkpoint/best_f1": self.best_f1}, step=self.global_step)

    def _check_early_stop(self, current_f1: float) -> bool:
        """
        Check if early stopping criteria are met.
        """
        if current_f1 > self.best_f1 + self.config.early_stopping_delta:
            self.best_f1 = current_f1
            self.patience_count = 0
            self._save_checkpoint(is_best=True)

            if self.config.use_wandb:
                wandb.log(
                    {
                        "train/best_f1": self.best_f1,
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

    def _create_scheduler(self, num_training_steps: int):
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _get_uni_label_map(self) -> tuple:
        def _get_uni_label(ori_label: str) -> str:
            if ori_label == "O":
                return "O"
            if ori_label.startswith(("B-", "I-")):
                prefix, ent_type = ori_label[:2], ori_label[2:]
                for uni_type, patterns in self.uni_rules.items():
                    for pattern in patterns:
                        if ent_type.lower().startswith(pattern.lower()):
                            return f"{prefix}{uni_type}"
            return ori_label

        uni_labels: list = ["O"]
        label_map: dict = {}

        for ds_name in self.all_ds:
            ds_labels = (
                self.idh.load_dataset(ds_name)["train"]
                .features["ner_tags"]
                .feature.names
            )
            label_map[ds_name] = {}

            for ori_id, label in enumerate(ds_labels):
                uni_label = _get_uni_label(label)
                if uni_label not in uni_labels:
                    uni_labels.append(uni_label)
                label_map[ds_name][ori_id] = uni_labels.index(uni_label)
        return uni_labels, label_map

    def _remap_ds(self):
        combined_train: list = []
        combined_val: list = []
        combined_test: list = []

        for ds_name in self.all_ds:
            ds_dict = self.idh.load_dataset(ds_name, quiet=True)
            mapping = self.label_map[ds_name]

            for split, target in [
                ("train", combined_train),
                ("validation", combined_val),
                ("test", combined_test),
            ]:
                for item in ds_dict[split]:
                    remapped = {
                        "tokens": item["tokens"],
                        "ner_tags": [mapping[label] for label in item["ner_tags"]],
                    }
                    target.append(remapped)

        self.train_dataset = NerDataset(combined_train, self.tokenizer)
        self.eval_dataset = NerDataset(combined_val, self.tokenizer)
        self.test_dataset = NerDataset(combined_test, self.tokenizer)

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

    def _save_wandb_artifacts(self) -> None:
        if not self.config.use_wandb:
            return

        try:
            model_artifact = wandb.Artifact(name="kairo_ner", type="model")

            # NOTE: Not necessary to upload the model
            if (
                self.config.use_wandb
                and self.config.upload_model_wandb
                and (self.config.checkpoint_dir / "best_model.pt").exists()
            ):
                model_artifact.add_file(
                    str(self.config.checkpoint_dir / "best_model.pt")
                )

            label_map_artifact = {
                "label_to_id": self.label_to_id,
                "id_to_label": self.id_to_label,
                "uni_labels": self.uni_labels,
            }

            justsdk.write_file(
                label_map_artifact, REPORTS_NER_DIR / "label_mapping.json"
            )
            model_artifact.add_file(REPORTS_NER_DIR / "label_mapping.json")
            wandb.log_artifact(model_artifact)

            justsdk.print_success(f"wandb artifacts saved to {REPORTS_NER_DIR}")

        except Exception as e:
            justsdk.print_warning(f"Failed to save wandb artifacts: {e}")


class NerDataset(Dataset):
    def __init__(
        self,
        data: list,
        tokenizer: BertTokenizerFast,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        tokens = item["tokens"]
        labels = item["ner_tags"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=NerConfig.max_length,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids()
        aligned_labels: list = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens ignored by the loss func
                aligned_labels.append(NerConfig.ignore_index)
            elif word_id != prev_word_id:
                # NOTE: Only the first token should align with label
                aligned_labels.append(labels[word_id])
            else:
                # That's why we ignore the rest of the token
                aligned_labels.append(NerConfig.ignore_index)

            prev_word_id = word_id

        # Convert to tensors
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(aligned_labels, dtype=torch.long)
        return encoding
