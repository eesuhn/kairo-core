from . import NerPredictor, NerConfig


class NerMain:
    def __init__(self) -> None:
        config = NerConfig(quite=True)
        self.predictor = NerPredictor(config=config)

    def extract_entities(self, texts: list) -> list:
        entities = self.predictor.predict(texts=texts)
        return entities
