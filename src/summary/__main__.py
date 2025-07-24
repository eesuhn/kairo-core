from . import AbsSumPredictor, AbsSumConfig, ExtSumPredictor, ExtSumConfig
from typing import Union


class SumMain:
    def __init__(self) -> None:
        abs_config = AbsSumConfig(quiet=True)
        self.asp = AbsSumPredictor(abs_config)

        ext_config = ExtSumConfig(quiet=True)
        self.esp = ExtSumPredictor(ext_config)

    def summarize(self, texts: Union[str, list[str]]) -> dict:
        abs_sum: list = self.asp.predict(texts=texts)
        ext_sum: list = self.esp.predict(texts=texts)

        res = {
            "abstract": abs_sum,
            "extractive": ext_sum,
        }
        return res
