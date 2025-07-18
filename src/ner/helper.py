class NerHelper:
    @staticmethod
    def map_unified_ds_to_base_labels(
        unified_ds_labels: list, base_labels: dict
    ) -> dict:
        """
        Maps unified dataset labels to base labels

        Args:
            unified_ds_labels: List of unified dataset labels
            base_labels: Dictionary of base labels with their IDs

        Returns:
            Dict mapping base label IDs to unified dataset label indices
        """
        rules_unified_to_base = {
            "CARDINAL": ["CARDINAL"],
            "DATE": ["DATE"],
            "EVENT": ["EVENT"],
            "FAC": ["FAC", "BUILDING"],
            "GPE": ["GPE"],  # XXX: Should we consider `"LOC", "LOCATION"`?
            "LANGUAGE": ["LANGUAGE"],
            "LAW": ["LAW"],
            "LOC": ["LOC", "LOCATION"],
            "MONEY": ["MONEY"],
            "NORP": ["NORP"],  # NOTE: Nationalities
            "O": ["O"],  # XXX: Should we add "OTHER", "MISC"?
            "ORDINAL": ["ORDINAL"],
            "ORG": ["ORG", "ORGANIZATION"],
            "PERCENT": ["PERCENT"],
            "PERSON": ["PERSON", "PER"],
            "PRODUCT": ["PRODUCT"],
            "QUANTITY": ["QUANTITY"],
            "TIME": ["TIME"],
            "WORK_OF_ART": ["WORK_OF_ART", "ART"],
        }

        label_to_idx = {label: idx for idx, label in enumerate(unified_ds_labels)}

        missing_labels = set(base_labels.keys()) - set(rules_unified_to_base.keys())
        if missing_labels:
            raise ValueError(f"Base labels {missing_labels} not found in mapping rules")

        return {
            base_id: [
                idx
                for unified_type in rules_unified_to_base[base_label]
                for label, idx in label_to_idx.items()
                if label == unified_type or label.endswith(f"-{unified_type}")
            ]
            for base_label, base_id in base_labels.items()
        }

    @staticmethod
    def map_base_to_dataset(base_labels: dict, dataset_labels: dict) -> dict:
        """
        Maps base labels to dataset-specific labels
        """
        return {
            ds_name: NerHelper._create_single_dataset_mapping(labels, base_labels)
            for ds_name, labels in dataset_labels.items()
        }

    @staticmethod
    def _create_single_dataset_mapping(labels: list, base_labels: dict) -> dict:
        """
        Create mapping for a single dataset's labels to base labels
        """
        rules_base_to_dataset = {
            # DFKI-SLT/few-nerd
            "art": "WORK_OF_ART",
            "building": "FAC",
            "event": "EVENT",
            "location": "LOC",
            "organization": "ORG",
            "other": "O",  # NOTE: other is mapped to O
            "person": "PERSON",
            "product": "PRODUCT",
            # eriktks/conll2003
            "LOC": "LOC",
            "MISC": "O",  # NOTE: MISC is mapped to O
            "ORG": "ORG",
            "PER": "PERSON",
        }

        mapping = {}
        for label in labels:
            if label == "O":
                mapping[label] = base_labels["O"]
            else:
                entity_type = label[2:] if label.startswith(("B-", "I-")) else label
                if entity_type not in rules_base_to_dataset:
                    raise ValueError(
                        f"Entity type '{entity_type}' not found in mapping rules"
                    )

                base_type = rules_base_to_dataset[entity_type]
                if base_type not in base_labels:
                    raise ValueError(
                        f"Base label '{base_type}' not found in base labels"
                    )

                mapping[label] = base_labels[base_type]

        return mapping
