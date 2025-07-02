import justsdk

from ._constants import CONFIGS_DIR, RAW_DATA_DIR
from datasets import load_dataset


class DataHandler:
    TARGET_DATA_CONFIG = CONFIGS_DIR / "datasets.yml"

    def download_data(self) -> None:
        target_data: dict = justsdk.read_file(self.TARGET_DATA_CONFIG)
        for c, datasets in target_data.items():
            justsdk.print_info(f"category/{c}")
            for ds in datasets:
                config = ds.get("config", None)
                output = RAW_DATA_DIR / ds["name"]
                if output.exists():
                    marker = f"{justsdk.Fore.GREEN}exists{justsdk.Fore.RESET}"
                else:
                    marker = f"{justsdk.Fore.MAGENTA}downloaded{justsdk.Fore.RESET}"
                    raw_data = load_dataset(ds["name"], config, trust_remote_code=True)
                    raw_data.save_to_disk(output)
                print(f"  - {ds['name']} ({marker})")


if __name__ == "__main__":
    dh = DataHandler()
    dh.download_data()
