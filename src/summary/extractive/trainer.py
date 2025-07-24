import torch
import justsdk
import os
import sys
import wandb
import signal

from .config import ExtSumConfig
from .model import ExtSumModel
from src.inter_data_handler import InterDataHandler
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from configs._constants import REPORTS_DIR


REPORTS_SUMMARY_DIR = REPORTS_DIR / "summary" / "extractive"


class ExtSumDataset(Dataset):
    def __init__(
        self,
        data: list,
        tokenizer: AutoTokenizer,
        config: ExtSumConfig,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

        self.processed_data = []
        for item in tqdm(data, desc="Processing dataset"):
            processed = self._process_item(item)
            if processed:
                self.processed_data.append(processed)

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, index: int) -> dict:
        return self.processed_data[index]

    def _process_item(self, item: dict) -> dict:
        sentences = item["source_sentences"]

        if len(sentences) > self.config.max_sentences:
            sentences = sentences[: self.config.max_sentences]
            challenge_labels = item["challenge_labels"][: self.config.max_sentences]
            approach_labels = item["approach_labels"][: self.config.max_sentences]
            outcome_labels = item["outcome_labels"][: self.config.max_sentences]
        else:
            challenge_labels = item["challenge_labels"]
            approach_labels = item["approach_labels"]
            outcome_labels = item["outcome_labels"]

        encoding = self._tokenize_sentences(sentences)

        if encoding is None:
            return None

        encoding["labels"] = {
            "challenge": torch.tensor(challenge_labels, dtype=torch.long),
            "approach": torch.tensor(approach_labels, dtype=torch.long),
            "outcome": torch.tensor(outcome_labels, dtype=torch.long),
        }

        return encoding

    def _tokenize_sentences(self, sentences: list) -> dict:
        text = f" {self.tokenizer.sep_token} ".join(sentences)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"][0]
        sentence_masks = self._create_sentence_masks(input_ids, len(sentences))

        if sentence_masks is None:
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"][0],
            "sentence_masks": sentence_masks,
        }

    def _create_sentence_masks(
        self, input_ids: torch.Tensor, num_sentences: int
    ) -> torch.Tensor:
        sep_token_id = self.tokenizer.sep_token_id
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]

        positions = [0] + sep_positions.tolist() + [len(input_ids)]
        masks = torch.zeros((num_sentences, len(input_ids)))

        for i in range(min(num_sentences, len(positions) - 1)):
            start = positions[i] + (1 if i > 0 else 0)
            end = positions[i + 1]
            masks[i, start:end] = 1

        return masks


class ExtSumTrainer:
    TARGET_DATASETS = ["sobamchan/aclsum-extractive"]

    def __init__(self, config: ExtSumConfig) -> None:
        self.config = config
        self.idh = InterDataHandler(quiet=self.config.quiet)

        # if self.config.device != "cuda":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.config.model_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        REPORTS_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

        if self.config.use_wandb:
            justsdk.print_info("Initializing wandb...")
            wandb.init(
                project="kairo-core",
                config=vars(self.config),
                dir=REPORTS_SUMMARY_DIR,
                name=f"ext_sum_{self.config.base_model_name.replace('/', '_')}",
            )

        self.model = ExtSumModel(self.config)
        self.optimizer = self._create_optimizer()

        self.pin_memory = self.config.device == "cuda"
        self.global_step = 0
        self.interrupted = False
        self.best_f1 = 0.0
        self.patience_count = 0

        signal.signal(signal.SIGINT, self._signal_handler)

    def train(self) -> None:
        self._load_datasets()
        self._calculate_class_weights()

        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
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

                batch = self._move_batch_to_device(batch)

                with torch.amp.autocast(device_type=self.config.device):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]

                loss = loss / self.config.gradient_accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps

                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    progress_bar.set_postfix(loss=avg_loss)

                    if self.config.use_wandb:
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": self.scheduler.get_last_lr()[0],
                                "train/global_step": self.global_step,
                            },
                            step=self.global_step,
                        )

                if self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    if self._check_early_stopping(eval_metrics["avg_f1"]):
                        return

                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

            if not self.interrupted:
                justsdk.print_info(f"End of epoch {epoch + 1}, evaluating...")
                self.evaluate()

        if self.interrupted:
            self._save_checkpoint()
            justsdk.print_success("Training interrupted, saving state...")
            if self.config.use_wandb:
                wandb.finish()
            sys.exit(0)

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
            collate_fn=self._collate_fn,
        )

        self.model.eval()

        all_predictions = {
            "challenge": [],
            "approach": [],
            "outcome": [],
        }
        all_labels = {
            "challenge": [],
            "approach": [],
            "outcome": [],
        }

        eval_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(eval_dl, desc="Evaluating"):
                batch = self._move_batch_to_device(batch)

                outputs = self.model(**batch)

                if "loss" in outputs:
                    eval_loss += outputs["loss"].item()

                challenge_probs = torch.sigmoid(outputs["challenge_logits"])
                approach_probs = torch.sigmoid(outputs["approach_logits"])
                outcome_probs = torch.sigmoid(outputs["outcome_logits"])

                challenge_preds = (
                    challenge_probs > self.config.challenge_threshold
                ).long()
                approach_preds = (
                    approach_probs > self.config.approach_threshold
                ).long()
                outcome_preds = (outcome_probs > self.config.outcome_threshold).long()

                for label_type, preds in [
                    ("challenge", challenge_preds),
                    ("approach", approach_preds),
                    ("outcome", outcome_preds),
                ]:
                    batch_labels = batch["labels"][label_type].cpu().numpy().flatten()
                    batch_preds = preds.cpu().numpy().flatten()

                    valid_mask = batch_labels != -100
                    valid_labels = batch_labels[valid_mask]
                    valid_preds = batch_preds[valid_mask]

                    all_predictions[label_type].extend(valid_preds)
                    all_labels[label_type].extend(valid_labels)

        metrics = {}
        avg_f1 = 0.0

        for label_type in ["challenge", "approach", "outcome"]:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels[label_type],
                all_predictions[label_type],
                average="binary",
                zero_division=0,
            )

            metrics[f"{label_type}_precision"] = precision
            metrics[f"{label_type}_recall"] = recall
            metrics[f"{label_type}_f1"] = f1
            avg_f1 += f1

        metrics["avg_f1"] = avg_f1 / 3.0
        metrics["eval_loss"] = eval_loss / len(eval_dl)

        justsdk.print_info("Evaluation results:", newline_before=True)
        justsdk.print_info(f"  Loss: {metrics['eval_loss']:.4f}")
        for label_type in ["challenge", "approach", "outcome"]:
            justsdk.print_info(
                f"  {label_type.capitalize()}: "
                f"    P={metrics[f'{label_type}_precision']:.3f}, "
                f"    R={metrics[f'{label_type}_recall']:.3f}, "
                f"    F1={metrics[f'{label_type}_f1']:.3f}"
            )
        justsdk.print_info(f"  Average F1: {metrics['avg_f1']:.3f}")

        if self.config.use_wandb:
            wandb_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            wandb_metrics["eval/global_step"] = self.global_step
            wandb.log(wandb_metrics, step=self.global_step)

        return metrics

    def _load_datasets(self) -> None:
        combined_train = []
        combined_val = []
        combined_test = []

        justsdk.print_info("Loading datasets...")

        for ds_name in self.TARGET_DATASETS:
            ds_dict = self.idh.load_dataset(ds_name, quiet=True)

            for split, target in [
                ("train", combined_train),
                ("validation", combined_val),
                ("test", combined_test),
            ]:
                if split in ds_dict:
                    target.extend(list(ds_dict[split]))

        self.train_ds = ExtSumDataset(combined_train, self.model.tokenizer, self.config)
        self.val_ds = ExtSumDataset(combined_val, self.model.tokenizer, self.config)
        self.test_ds = ExtSumDataset(combined_test, self.model.tokenizer, self.config)

    def _calculate_class_weights(self) -> None:
        if not hasattr(self, "train_ds"):
            return

        label_counts = {"challenge": 0, "approach": 0, "outcome": 0}
        total_samples = 0

        for item in self.train_ds.processed_data:
            for label_type in label_counts:
                label_counts[label_type] += item["labels"][label_type].sum().item()
                total_samples += len(item["labels"][label_type])

        weights = []
        for label_type in ["challenge", "approach", "outcome"]:
            pos_count = label_counts[label_type]
            neg_count = total_samples - pos_count
            weight = neg_count / pos_count if pos_count > 0 else 1.0
            weights.append(weight)

        self.config.class_weights = weights
        justsdk.print_info(f"Class weights: {weights}")

    def _collate_fn(self, batch: list) -> dict:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])

        max_sentences = max(item["sentence_masks"].shape[0] for item in batch)

        padded_sentence_masks = []
        padded_labels = {
            "challenge": [],
            "approach": [],
            "outcome": [],
        }

        for item in batch:
            num_sentences = item["sentence_masks"].shape[0]

            if num_sentences < max_sentences:
                padding = torch.zeros(
                    (max_sentences - num_sentences, item["sentence_masks"].shape[1])
                )
                padded_mask = torch.cat([item["sentence_masks"], padding], dim=0)
            else:
                padded_mask = item["sentence_masks"]
            padded_sentence_masks.append(padded_mask)

            for label_type in ["challenge", "approach", "outcome"]:
                labels = item["labels"][label_type]
                if len(labels) < max_sentences:
                    padding = torch.full(
                        (max_sentences - len(labels),), -100, dtype=labels.dtype
                    )
                    padded_label = torch.cat([labels, padding])
                else:
                    padded_label = labels
                padded_labels[label_type].append(padded_label)

        sentence_masks = torch.stack(padded_sentence_masks)
        labels = {key: torch.stack(values) for key, values in padded_labels.items()}

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sentence_masks": sentence_masks,
            "labels": labels,
        }

    def _move_batch_to_device(self, batch: dict) -> dict:
        batch["input_ids"] = batch["input_ids"].to(self.config.device)
        batch["attention_mask"] = batch["attention_mask"].to(self.config.device)
        batch["sentence_masks"] = batch["sentence_masks"].to(self.config.device)

        for key in batch["labels"]:
            batch["labels"][key] = batch["labels"][key].to(self.config.device)

        return batch

    def _check_early_stopping(self, current_f1: float) -> bool:
        if current_f1 > self.best_f1 + self.config.early_stopping_delta:
            self.best_f1 = current_f1
            self.patience_count = 0
            self._save_checkpoint(is_best=True)
            return False
        else:
            self.patience_count += 1
            if self.patience_count >= self.config.early_stopping_patience:
                justsdk.print_info("Early stopping triggered")
                return True
            return False

    def _save_checkpoint(self, is_best: bool = False) -> None:
        if is_best:
            filepath = self.config.model_dir / "model.pt"
        else:
            filepath = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"

        torch.save(self.model.state_dict(), filepath)
        justsdk.print_success(f"Checkpoint saved to {filepath}")

    def _save_wandb_artifacts(self) -> None:
        if not self.config.use_wandb:
            return

        try:
            model_artifact = wandb.Artifact(name="kairo_ext_sum", type="model")

            if (
                self.config.upload_model_wandb
                and (self.config.model_dir / "model.pt").exists()
            ):
                model_artifact.add_file(str(self.config.model_dir / "model.pt"))

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

    def _create_scheduler(self, num_training_steps: int):
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _signal_handler(self, signum, frame) -> None:
        justsdk.print_warning("Training interrupted...")
        self.interrupted = True
