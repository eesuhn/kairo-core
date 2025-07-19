import justsdk
import argparse

from src.inter_data_handler import InterDataHandler


def inspect_local_dataset(name, split="train", n=3) -> None:
    """
    Inspect a local dataset by loading it and displaying a few examples

    Args:
        name: Name of the dataset to inspect.
        split: The split of the dataset to inspect.
        n: Number of examples to display.
    """
    dh = InterDataHandler()
    ds_dict = dh.load_dataset(name)

    # NOTE: Avaiable splits: ['train', 'validation', 'test']
    if split not in ds_dict:
        raise ValueError(
            f"Split '{split}' not available. Available splits: {list(ds_dict.keys())}"
        )
    ds = ds_dict[split]

    justsdk.print_info(f"Available splits: {list(ds_dict.keys())}", newline_before=True)
    justsdk.print_info(f"Loaded '{split}' split with {len(ds)} sequences")

    justsdk.print_info("Index → NER Label mapping:", newline_before=True)
    label_list = ds.features["ner_tags"].feature.names
    for idx, lbl in enumerate(label_list):
        print(f"  {idx:>2}: {lbl}")

    justsdk.print_info(f"Showing {n} examples (token: label):", newline_before=True)
    for ex in ds.select(range(n)):
        mappings = {
            token: label_list[tag_idx]
            for token, tag_idx in zip(ex["tokens"], ex["ner_tags"])
        }
        justsdk.print_debug("DIVIDER")
        justsdk.print_data(mappings, use_orjson=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a local NER dataset.")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to inspect."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Split of the dataset to inspect.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=3,
        help="Number of examples to display.",
    )
    args = parser.parse_args()

    inspect_local_dataset(name=args.dataset, split=args.split, n=args.num)
