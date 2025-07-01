import justsdk

from ._constants import CONFIGS_DIR


class Data:
    def __init__(self) -> None:
        self.list_of_datasets = justsdk.read_file(CONFIGS_DIR / "datasets.yml")


if __name__ == "__main__":
    data = Data()
