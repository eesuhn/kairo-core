import justsdk
import argparse

from src.input_processor import InputProcessor
from src.ner import Ner


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
        if self.args.file is not None:
            self.ip = InputProcessor()

            payload = self.ip.process(self.args.file)
            if payload["status"] != "success":
                justsdk.print_error("Failed to process the input file.")
                return
            texts = payload["content"]

            entities = Ner.extract_entities(texts=texts)
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
                    entities = Ner.extract_entities(texts=text)
                    self._print_entities(entities)

    def _print_entities(self, entities: dict) -> None:
        if not entities:
            justsdk.print_warning("No entities found.")
            return

        justsdk.print_data(entities)


if __name__ == "__main__":
    nps = NerPredictScript()
    nps.run()
