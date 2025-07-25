from . import AbsSumPredictor, AbsSumConfig, ExtSumPredictor, ExtSumConfig
from typing import Union
from http import HTTPStatus


class Summary:
    @staticmethod
    def abstract_summarize(texts: Union[str, list]) -> dict:
        """
        Generate abstractive summary from a single text or a batch of texts.

        Returns:
            status (in dict): Status of the summary generation, either "success" or "error".
            status_code (in dict): HTTP status code indicating the result of the summary generation.
            message (in dict): Message indicating the result of the summary generation.
            data (in dict): Contains the generated summary and input type.
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

            abs_config = AbsSumConfig(quiet=True)
            asp = AbsSumPredictor(abs_config)
            summary = asp.predict(texts=texts)

            return {
                "status": "success",
                "status_code": HTTPStatus.OK,
                "message": "Summary generated successfully",
                "data": {
                    "summary": summary,
                    "input_type": "single" if isinstance(texts, str) else "batch",
                    "summary_type": "abstractive",
                },
            }

        except Exception as e:
            return {
                "status": "error",
                "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
                "message": f"Failed to generate abstractive summary: {str(e)}",
                "data": None,
            }

    @staticmethod
    def extract_summarize(texts: Union[str, list]) -> dict:
        """
        Generate extractive summary from a single text or a batch of texts.

        Returns:
            status (in dict): Status of the summary generation, either "success" or "error".
            status_code (in dict): HTTP status code indicating the result of the summary generation.
            message (in dict): Message indicating the result of the summary generation.
            data (in dict): Contains the generated summary and input type.
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

            ext_config = ExtSumConfig(quiet=True)
            esp = ExtSumPredictor(ext_config)
            summary = esp.generate_summary(texts=texts)

            return {
                "status": "success",
                "status_code": HTTPStatus.OK,
                "message": "Summary generated successfully",
                "data": {
                    "summary": summary,
                    "input_type": "single" if isinstance(texts, str) else "batch",
                    "summary_type": "extractive",
                    "count": len(summary) if isinstance(summary, list) else 0,
                },
            }

        except Exception as e:
            return {
                "status": "error",
                "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
                "message": f"Failed to generate extractive summary: {str(e)}",
                "data": None,
            }
