import justsdk
import argparse

from src.ner.predictor import NerPredictor, NerEntity
from src.input_processor import InputProcessor
from typing import Optional


class NerPredictScript:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="Predict NER")
        parser.add_argument(
            "--file",
            type=str,
            help="Path to the text file for NER prediction",
        )
        self.args = parser.parse_args()

    def run(self) -> None:
        self.predictor = NerPredictor()

        if self.args.file is not None:
            self.ip = InputProcessor()

            # TODO: Support audio script
            target = self.ip.process(self.args.file)
            entities = self.predictor.predict(texts=target)
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
                    entities = self.predictor.predict(texts=text)
                    self._print_entities(entities)

    def _print_entities(self, entities: Optional[list[NerEntity]]) -> None:
        if not entities:
            justsdk.print_warning("No entities found.")
            return

        for ent in entities:
            print(f"  {ent.text} -> {ent.label} ({ent.confidence:.2f})")


if __name__ == "__main__":
    nps = NerPredictScript()
    nps.run()
