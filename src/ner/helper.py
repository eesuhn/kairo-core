class NerHelper:
    @staticmethod
    def map_unified_ds_to_base_labels(
        unified_ds_labels: list, base_labels: dict
    ) -> dict:
        mapping: dict = {}

        # NOTE: Not attending to "OTHER", "MISC"
        rules_unified_to_base = {
            "CARDINAL": ["CARDINAL"],
            "DATE": ["DATE"],
            "EVENT": ["EVENT"],
            "FAC": ["FAC", "BUILDING"],
            "GPE": ["GPE"],  # NOTE: Should we consider `"LOC", "LOCATION"`?
            "LANGUAGE": ["LANGUAGE"],
            "LAW": ["LAW"],
            "LOC": ["LOC", "LOCATION"],
            "MONEY": ["MONEY"],
            "NORP": ["NORP"],  # NOTE: Nationalities
            "ORDINAL": ["ORDINAL"],
            "ORG": ["ORG", "ORGANIZATION"],
            "PERCENT": ["PERCENT"],
            "PERSON": ["PERSON", "PER"],
            "PRODUCT": ["PRODUCT"],
            "QUANTITY": ["QUANTITY"],
            "TIME": ["TIME"],
            "WORK_OF_ART": ["WORK_OF_ART", "ART"],
        }

        for base_label, base_id in base_labels.items():
            mapping[base_id] = []
            if base_label == "O":  # NOTE: Link 0 for outside label
                mapping[base_id].append(0)
            elif base_label in rules_unified_to_base:
                for unified_type in rules_unified_to_base[base_label]:
                    for idx, label in enumerate(unified_ds_labels):
                        label: str
                        if label.endswith(f"-{unified_type}"):
                            mapping[base_id].append(idx)
            else:
                raise ValueError(
                    f"Base label '{base_label}' not found in mapping rules"
                )
        return mapping
