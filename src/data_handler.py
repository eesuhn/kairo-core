import justsdk

from typing import Optional
from configs._constants import CONFIGS_DIR, RAW_DATA_DIR
from datasets import load_dataset, DatasetDict, load_from_disk
from pathlib import Path


class DataHandler:
    TARGET_DATA_CONFIG = CONFIGS_DIR / "datasets.yml"

    def __init__(self):
        self._dataset_cache: dict = {}

    def download_data(self) -> None:
        """
        Download datasets from `TARGET_DATA_CONFIG`, save them to `RAW_DATA_DIR`
        """
        target_data = justsdk.read_file(self.TARGET_DATA_CONFIG)

        for category, datasets in target_data.items():
            justsdk.print_info(f"category/{category}")

            for dataset in datasets:
                name = dataset["name"]
                output_path = RAW_DATA_DIR / name

                if output_path.exists():
                    status = "exists", justsdk.Fore.GREEN
                else:
                    self._download_dataset(dataset, output_path)
                    status = "downloaded", justsdk.Fore.MAGENTA

                self._print_ds_status(name, *status)

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

        Args:
            dataset_name: Name of the dataset (e.g., 'allenai/scitldr')
            use_cache: Whether to use cached dataset if available

        Returns:
            DatasetDict if found, None otherwise
        """
        if use_cache and dataset_name in self._dataset_cache:
            justsdk.print_info(f"Loading cached dataset: {dataset_name}")
            return self._dataset_cache[dataset_name]

        dataset_path = RAW_DATA_DIR / dataset_name

        if not dataset_path.exists():
            justsdk.print_error(f"Dataset not found: {dataset_path}")
            return None

        try:
            justsdk.print_info(f"Loading dataset from disk: {dataset_name}")
            # dataset = DatasetDict.load_from_disk(
            #     str(dataset_path)
            # )  # NOTE: Only returns `DatasetDict`
            dataset = load_from_disk(str(dataset_path))

            if use_cache:
                self._dataset_cache[dataset_name] = dataset

            return dataset

        except Exception as e:
            justsdk.print_error(f"Error loading dataset {dataset_name}: {e}")
            return None

    def clear_cache(self) -> None:
        """
        Clear the dataset cache
        """
        self._dataset_cache.clear()
        justsdk.print_info("Cleared DataHandler cache")

    def list_cached_datasets(self) -> list:
        """
        Get list of currently cached dataset names
        """
        return list(self._dataset_cache.keys())


if __name__ == "__main__":
    dh = DataHandler()
    dh.download_data()
