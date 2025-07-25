from . import NerPredictor, NerConfig


class Ner:
    @staticmethod
    def extract_entities(texts: list) -> list:
        config = NerConfig(quite=True)
        predictor = NerPredictor(config=config)
        entities = predictor.predict(texts=texts)
        return entities
