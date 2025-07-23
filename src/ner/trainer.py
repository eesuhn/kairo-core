import justsdk
import torch

from ..inter_data_handler import InterDataHandler
from configs._constants import CONFIGS_DIR
from .model import NerModel
from .config import NerConfig
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


class NerTrainer:
    def __init__(self, config: NerConfig) -> None:
        self.config = config
        self.idh = InterDataHandler()

        if not self.config.model_dir.exists():
            self.config.model_dir.mkdir(parents=True, exist_ok=True)

        self.all_ds = self.idh.list_datasets_by_category("ner")
        self.uni_rules = justsdk.read_file(CONFIGS_DIR / "ner" / "rules.yml")

        # Prepare labels and remap datasets
        self.uni_labels, self.label_map = self._get_uni_label_map()
        self.label_to_id = {label: i for i, label in enumerate(self.uni_labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.uni_labels)}

        self.model = NerModel(num_labels=len(self.uni_labels))
        self.tokenizer = BertTokenizer.from_pretrained(self.config.base_model_name)
        self.optimizer = self._create_optimizer()

        # Training state
        self.global_step = 0
        self.best_f1 = 0.0

        self._remap_ds()

    def train(self, resume: bool = False) -> None:
        dl = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        num_training_steps = len(dl) * self.config.epochs
        self.scheduler = self._create_scheduler(num_training_steps)

        # TODO: Support resume training
        if resume:
            pass

        justsdk.print_info("Starting training...")
        justsdk.print_info(f"  No. of examples: {len(self.train_dataset)}")
        justsdk.print_info(f"  No. of epochs: {self.config.epochs}")
        justsdk.print_info(f"  Batch size: {self.config.batch_size}")
        justsdk.print_info(f"  Total training steps: {num_training_steps}")

        # Training loop...
        for epoch in range(self.config.epochs):
            pass

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


class NerDataset(Dataset):
    def __init__(
        self,
        data: list,
        tokenizer: BertTokenizer,
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
