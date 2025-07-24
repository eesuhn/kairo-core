import justsdk
import argparse

from src.input_processor import InputProcessor
from src.ner import NerMain


class NerPredictScript:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="Predict NER")
        parser.add_argument(
            "--file",
            type=str,
            help="Path to the text file for NER prediction",
        )
        parser.add_argument(
            "--return-conf",
            action="store_true",
            help="Return confidence scores for entities",
        )
        self.args = parser.parse_args()

    def run(self) -> None:
        self.ner = NerMain()

        if self.args.file is not None:
            self.ip = InputProcessor()

            # TODO: Support audio script
            target = self.ip.process(self.args.file)
            entities = self.ner.extract_entities(texts=target)
            self._print_entities(entities)

        else:  # Interactive mode
            justsdk.print_info(
                "Interactive mode. Type 'q' to quit.\n", newline_before=True
            )
            while True:
                text = input("\nInput: ").strip()
                if text.lower() == "q":
                    justsdk.print_success("Exiting interactive mode.")
                    break

                if text:
                    entities = self.ner.extract_entities(texts=text)
                    self._print_entities(entities)

    def _print_entities(self, entities: list) -> None:
        if not entities:
            justsdk.print_warning("No entities found.")
            return

        for ent in entities:
            print(f"  {ent.text} -> {ent.label}")


if __name__ == "__main__":
    nps = NerPredictScript()
    nps.run()
