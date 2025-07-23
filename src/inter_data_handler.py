import justsdk
import platform

from typing import Optional
from config._constants import CONFIG_DIR, INTER_DATA_DIR
from datasets import load_dataset, DatasetDict, load_from_disk
from pathlib import Path


class InterDataHandler:
    TARGET_DATA_CONFIG = CONFIG_DIR / "datasets.yml"

    def __init__(self):
        self._dataset_cache: dict = {}

    def download_data(self) -> None:
        """
        Download datasets from `TARGET_DATA_CONFIG`, save them to `INTER_DATA_DIR`
        """
        target_data = justsdk.read_file(self.TARGET_DATA_CONFIG)

        for category, datasets in target_data.items():
            justsdk.print_info(f"category/{category}")

            for dataset in datasets:
                name = dataset["name"]

                # Use custom `dir` if specified, otherwise use dataset name
                dir_name = dataset.get("dir", name)
                output_path = INTER_DATA_DIR / dir_name

                if output_path.exists():
                    status = "exists", justsdk.Fore.GREEN
                else:
                    self._download_dataset(dataset, output_path)
                    status = "downloaded", justsdk.Fore.MAGENTA

                # Display both dataset name and directory if they differ
                display_name = name if dir_name == name else f"{dir_name} <- {name}"
                self._print_ds_status(display_name, *status)

    def _download_dataset(self, dataset: dict, output_path: Path) -> None:
        """
        Save Hugging Face dataset to disk
        """
        config = dataset.get("config")
        raw_data = load_dataset(dataset["name"], config, trust_remote_code=True)
        raw_data.save_to_disk(output_path)

    def _print_ds_status(self, name: str, status: str, color: str) -> None:
        print(f"  - {name} ({color}{status}{justsdk.Fore.RESET})")

    def load_dataset(
        self, dataset_name: str, use_cache: bool = True
    ) -> Optional[DatasetDict]:
        """
        Load a dataset from local storage with optional caching

        `load_from_disk(keep_in_memory=True)` if running on Linux/Windows

        Args:
            dataset_name: Name of the dataset (e.g., 'allenai/scitldr')
            use_cache: Whether to use cached dataset if available

        Returns:
            DatasetDict if found, None otherwise
        """
        if use_cache and dataset_name in self._dataset_cache:
            justsdk.print_info(f"Loading cached dataset: {dataset_name}")
            return self._dataset_cache[dataset_name]

        dataset_path = INTER_DATA_DIR / dataset_name

        if not dataset_path.exists():
            justsdk.print_error(f"Dataset not found: {dataset_path}")
            return None

        try:
            justsdk.print_info(f"Loading dataset from disk: {dataset_name}")
            # dataset = DatasetDict.load_from_disk(
            #     str(dataset_path)
            # )  # NOTE: Only returns `DatasetDict`

            # NOTE: Cache dataset in RAM for Linux/Windows
            keep_in_memory = platform.system() in ["Linux", "Windows"]
            dataset = load_from_disk(
                dataset_path=str(dataset_path), keep_in_memory=keep_in_memory
            )

            if use_cache:
                self._dataset_cache[dataset_name] = dataset

            return dataset

        except Exception as e:
            justsdk.print_error(f"Error loading dataset {dataset_name}: {e}")
            return None

    def list_datasets_by_category(self, category: str) -> list:
        """
        List dataset names by category from the config

        Returns:
            List of dataset if category exists, empty list otherwise
        """
        return [
            dataset["name"]
            for dataset in justsdk.read_file(self.TARGET_DATA_CONFIG).get(category, [])
        ]

    def clear_cache(self) -> None:
        self._dataset_cache.clear()
        justsdk.print_info("Cleared DataHandler cache")

    def list_cached_datasets(self) -> list:
        return list(self._dataset_cache.keys())


if __name__ == "__main__":
    dh = InterDataHandler()
    dh.download_data()
