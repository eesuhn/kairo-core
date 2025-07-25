from . import NerPredictor, NerConfig
from typing import Union
from http import HTTPStatus


class Ner:
    @staticmethod
    def extract_entities(texts: Union[str, list]) -> dict:
        """
        Extract named entities from a single text or a batch of texts.

        Returns:
            status (in dict): Status of the entity extraction, either "success" or "error".
            status_code (in dict): HTTP status code indicating the result of the entity extraction.
            message (in dict): Message indicating the result of the entity extraction.
            data (in dict): Contains the extracted entities and input type.
        """
        try:
            if not texts:
                return {
                    "status": "error",
                    "status_code": HTTPStatus.BAD_REQUEST,
                    "message": "Text input cannot be empty",
                    "data": None,
                }

            if isinstance(texts, str) and not texts.strip():
                return {
                    "status": "error",
                    "status_code": HTTPStatus.BAD_REQUEST,
                    "message": "Text input cannot be empty string",
                    "data": None,
                }

            config = NerConfig(quite=True)
            predictor = NerPredictor(config=config)
            entities = predictor.predict(texts=texts)

            return {
                "status": "success",
                "status_code": HTTPStatus.OK,
                "message": "Entities extracted successfully",
                "data": {
                    "entities": entities,
                    "input_type": "single" if isinstance(texts, str) else "batch",
                    "count": len(entities) if isinstance(entities, list) else 0,
                },
            }

        except Exception as e:
            return {
                "status": "error",
                "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
                "message": f"Failed to extract entities: {str(e)}",
                "data": None,
            }
