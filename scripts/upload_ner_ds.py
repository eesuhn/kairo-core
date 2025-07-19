import justsdk
import os
import argparse

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    ClassLabel,
    Value,
)
from huggingface_hub import HfApi, login
from config._constants import CONFIG_DIR


class UploadNerDataset:
    def __init__(self) -> None:
        self.domain_labels = justsdk.read_file(CONFIG_DIR / "ner" / "labels.yml")

    def _read_conll_file(self, filepath, label_to_id):
        """Read CoNLL format file."""
        tokens_list = []
        ner_tags_list = []
        current_tokens = []
        current_tags = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if current_tokens:
                        tokens_list.append(current_tokens)
                        ner_tags_list.append(current_tags)
                        current_tokens = []
                        current_tags = []
                else:
                    parts = line.split("\t")
                    if len(parts) == 2:
                        token, label = parts
                        current_tokens.append(token)
                        current_tags.append(label_to_id.get(label, 0))

            if current_tokens:
                tokens_list.append(current_tokens)
                ner_tags_list.append(current_tags)

        return tokens_list, ner_tags_list

    def _create_ds_card(self, domain, dataset_dict):
        """Create a dataset card with full dataset_info like conll2003."""
        labels = self.domain_labels[domain]

        num_train = len(dataset_dict.get("train", []))
        num_validation = len(dataset_dict.get("validation", []))
        num_test = len(dataset_dict.get("test", []))

        yaml_lines = [
            "---",
            "annotations_creators:",
            "  - expert-generated",
            "language_creators:",
            "  - found",
            "language:",
            "  - en",
            "license:",
            "  - other",
            "multilinguality:",
            "  - monolingual",
            "size_categories:",
            "  - 10K<n<100K",
            "source_datasets:",
            "  - original",
            "task_categories:",
            "  - token-classification",
            "task_ids:",
            "  - named-entity-recognition",
            "paperswithcode_id: crossner",
            f"pretty_name: CrossNER-{domain.upper()}",
            "dataset_info:",
            "  features:",
            "  - name: tokens",
            "    sequence: string",
            "  - name: ner_tags",
            "    sequence:",
            "      class_label:",
            "        names:",
        ]

        for i, label in enumerate(labels):
            yaml_lines.append(f'            "{i}": {label}')

        yaml_lines.extend(
            [
                "  splits:",
                "  - name: train",
                f"    num_bytes: {num_train * 100}",
                f"    num_examples: {num_train}",
                "  - name: validation",
                f"    num_bytes: {num_validation * 100}",
                f"    num_examples: {num_validation}",
                "  - name: test",
                f"    num_bytes: {num_test * 100}",
                f"    num_examples: {num_test}",
                "---",
            ]
        )

        yaml_str = "\n".join(yaml_lines)
        readme_content = f"""{yaml_str}

    # CrossNER {domain.upper()} Dataset

    This is a NER dataset for cross-domain evaluation.  
    This split contains data from the {domain.upper()} domain, read more [here](https://arxiv.org/abs/2012.04373).

    ## Features

    - **tokens**: A list of words in the sentence
    - **ner_tags**: A list of NER labels (as integers) corresponding to each token

    ## Label Mapping

    The dataset uses the following {len(labels)} labels:

    | Index | Label |
    |-------|-------|
    {chr(10).join(f"| {i} | {label} |" for i, label in enumerate(labels))}

    ## Usage

    ```python
    from datasets import load_dataset

    dataset = load_dataset("eesuhn/crossner-{domain}")
    ```
    """

        return readme_content

    def run(self) -> None:
        parser = argparse.ArgumentParser(description="Upload CrossNER datasets")
        parser.add_argument("--username", type=str, required=True)
        parser.add_argument("--token", type=str, required=True)
        parser.add_argument(
            "--delete-existing",
            action="store_true",
            help="Whether to delete existing datasets with the same name",
        )
        args = parser.parse_args()

        login(args.token)
        api = HfApi(token=args.token)

        for domain in ["ai", "literature", "science"]:
            print(f"\nProcessing {domain}...")
            repo_id = f"{args.username}/crossner-{domain}"

            if args.delete_existing:
                api.delete_repo(repo_id, repo_type="dataset", token=args.token)
                print(f"Deleted existing dataset: {repo_id}")
                continue

            labels = self.domain_labels[domain]
            label_to_id = {label: idx for idx, label in enumerate(labels)}

            features = Features(
                {
                    "tokens": Sequence(Value("string")),
                    "ner_tags": Sequence(
                        ClassLabel(num_classes=len(labels), names=labels)
                    ),
                }
            )

            splits = {}
            for split_name, filename in [
                ("train", "train.txt"),
                ("validation", "dev.txt"),
                ("test", "test.txt"),
            ]:
                filepath = os.path.join("./ner_data", domain, filename)
                if os.path.exists(filepath):
                    tokens_list, ner_tags_list = self._read_conll_file(
                        filepath, label_to_id
                    )
                    dataset = Dataset.from_dict(
                        {"tokens": tokens_list, "ner_tags": ner_tags_list},
                        features=features,
                    )
                    splits[split_name] = dataset
                    print(f"  {split_name}: {len(dataset)} examples")

            dataset_dict = DatasetDict(splits)

            card_content = self._create_ds_card(domain, dataset_dict)
            dataset_dict.push_to_hub(repo_id, token=args.token)

            api.upload_file(
                path_or_fileobj=card_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=args.token,
            )

            print(f"Uploaded {repo_id}")

        print("\nAll done!")


if __name__ == "__main__":
    und = UploadNerDataset()
    und.run()
