from . import NerPredictor, NerConfig
from pathlib import Path
from ..input_processor import InputProcessor


class NerMain:
    def __init__(self) -> None:
        config = NerConfig(quite=True)
        self.predictor = NerPredictor(config=config)

    def extract_entities(self, file_path: Path) -> list:
        ip = InputProcessor()
        target = ip.process(file_path, quiet=True)
        entities = self.predictor.predict(texts=target)
        return entities
