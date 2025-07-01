import justsdk

from ._constants import CONFIGS_DIR, RAW_DATA_DIR
from datasets import load_dataset


class Data:
    def __init__(self) -> None:
        self.download_all_data()

    def download_all_data(self) -> None:
        self.list_of_datasets = justsdk.read_file(CONFIGS_DIR / "datasets.yml")
        for category, datasets in self.list_of_datasets["datasets"].items():
            justsdk.print_info(f"Category: {category}")
            for ds in datasets:
                config = ds.get("config", None)

                output_path = RAW_DATA_DIR / ds["name"]
                if not output_path.exists():
                    exist_marker = f"{justsdk.Fore.GREEN}downloaded{justsdk.Fore.RESET}"
                    raw_data = load_dataset(ds["name"], config, trust_remote_code=True)
                    raw_data.save_to_disk(output_path)
                else:
                    exist_marker = f"{justsdk.Fore.MAGENTA}exists{justsdk.Fore.RESET}"
                print(
                    f"  - {ds['name']}: {ds.get('description', 'n/a')} ({exist_marker})"
                )


if __name__ == "__main__":
    data = Data()
