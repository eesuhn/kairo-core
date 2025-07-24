from ..inter_data_handler import InterDataHandler


class NerHelper:
    @staticmethod
    def get_uni_label_map(self) -> tuple:
        idh = InterDataHandler()

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
                idh.load_dataset(ds_name)["train"].features["ner_tags"].feature.names
            )
            label_map[ds_name] = {}

            for ori_id, label in enumerate(ds_labels):
                uni_label = _get_uni_label(label)
                if uni_label not in uni_labels:
                    uni_labels.append(uni_label)
                label_map[ds_name][ori_id] = uni_labels.index(uni_label)
        return uni_labels, label_map
