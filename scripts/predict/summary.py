import justsdk
import argparse

from src.summary import Summary
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
        if self.args.file is not None:
            self.ip = InputProcessor()

            payload = self.ip.process(self.args.file)
            if payload["status"] != "success":
                justsdk.print_error("Failed to process the input file.")
                return
            texts = payload["content"]

            abs_sum = Summary.abstract_summarize(texts=texts)
            ext_sum = Summary.extract_summarize(texts=texts)

            summaries = {
                "abstract": abs_sum,
                "extractive": ext_sum,
            }

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
        justsdk.print_data(abs_sum)

        ext_sum = summaries.get("extractive", [])
        justsdk.print_success("Extractive:")
        justsdk.print_data(ext_sum)


if __name__ == "__main__":
    sps = SumPredictScript()
    sps.run()
