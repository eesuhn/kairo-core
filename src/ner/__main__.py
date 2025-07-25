from . import NerPredictor, NerConfig
from typing import Union


class Ner:
    @staticmethod
    def extract_entities(texts: Union[str, list]) -> list:
        config = NerConfig(quite=True)
        predictor = NerPredictor(config=config)
        entities = predictor.predict(texts=texts)
        return entities
