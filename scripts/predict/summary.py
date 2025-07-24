import justsdk
import argparse

from src.summary import SumMain
from src.input_processor import InputProcessor


class SumPredictScript:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="Predict NER")
        parser.add_argument(
            "--file",
            type=str,
            help="Path to the text file for NER prediction",
        )
        self.args = parser.parse_args()

    def run(self) -> None:
        self.sum = SumMain()

        if self.args.file is not None:
            self.ip = InputProcessor()

            # TODO: Support audio script
            target = self.ip.process(self.args.file)
            summaries = self.sum.summarize(texts=target)
            self._print_summaries(summaries)

        else:  # Interactive mode
            print("Interactive mode. Type 'q' to quit.\n")
            while True:
                text = input("\nInput: ").strip()
                if text.lower() == "q":
                    print("Exiting interactive mode.")
                    break

                if text:
                    summaries = self.sum.summarize(texts=text)
                    self._print_summaries(summaries)

    def _print_summaries(self, summaries: dict) -> None:
        abs_sum = summaries.get("abstract", [])
        justsdk.print_success("Abstract:")
        print(abs_sum)


if __name__ == "__main__":
    sps = SumPredictScript()
    sps.run()
