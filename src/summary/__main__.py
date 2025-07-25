from . import AbsSumPredictor, AbsSumConfig, ExtSumPredictor, ExtSumConfig
from typing import Union


class Summary:
    @staticmethod
    def abstract_summarize(texts: Union[str, list]) -> Union[str, list[str]]:
        abs_config = AbsSumConfig(quiet=True)
        asp = AbsSumPredictor(abs_config)
        return asp.predict(texts=texts)

    @staticmethod
    def extract_summarize(texts: Union[str, list]) -> list:
        ext_config = ExtSumConfig(quiet=True)
        esp = ExtSumPredictor(ext_config)
        return esp.generate_summary(texts=texts)
