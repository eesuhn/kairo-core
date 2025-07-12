import justsdk

from configs._constants import CONFIGS_DIR, RAW_DATA_DIR
from datasets import load_dataset
from pathlib import Path


class DataHandler:
    TARGET_DATA_CONFIG = CONFIGS_DIR / "datasets.yml"

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


if __name__ == "__main__":
    dh = DataHandler()
    dh.download_data()
