import justsdk

from ..inter_data_handler import InterDataHandler
from configs._constants import CONFIGS_DIR


class NerHelper:
    idh = InterDataHandler(quiet=True)
    uni_rules = justsdk.read_file(CONFIGS_DIR / "ner" / "rules.yml")
    all_ds = idh.list_datasets_by_category("ner")

    @staticmethod
    def get_uni_label_map() -> tuple:
        def _get_uni_label(ori_label: str) -> str:
            if ori_label == "O":
                return "O"
            if ori_label.startswith(("B-", "I-")):
                prefix, ent_type = ori_label[:2], ori_label[2:]
                for uni_type, patterns in NerHelper.uni_rules.items():
                    for pattern in patterns:
                        if ent_type.lower().startswith(pattern.lower()):
                            return f"{prefix}{uni_type}"
            return ori_label

        uni_labels: list = ["O"]
        label_map: dict = {}

        for ds_name in NerHelper.all_ds:
            ds_labels = (
                NerHelper.idh.load_dataset(ds_name)["train"]
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
