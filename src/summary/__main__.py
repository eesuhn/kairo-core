from . import AbsSumPredictor, AbsSumConfig
from typing import Union


class SumMain:
    def __init__(self) -> None:
        config = AbsSumConfig(quiet=True)
        self.asp = AbsSumPredictor(config)

    def summarize(self, texts: Union[str, list[str]]) -> dict:
        abs_sum: list = self.asp.predict(texts=texts)

        res = {
            "abstract": abs_sum,
        }
        return res
