import pymupdf
import warnings

from http import HTTPStatus
from pathlib import Path
from .audio_processor import AudioProcessor
from configs._constants import ROOT_PATH

warnings.filterwarnings("ignore")


class InputProcessor:
    # TODO: To support markdown and .txt files
    TEXT_EXT = (".pdf", ".docx")
    AUDIO_EXT = (".mp3", ".mp4")

    @staticmethod
    def process(input_path: Path) -> dict:
        """
        Process the input file based on supported formats.

        Returns:
            status (in dict): Status of the processing, either "success" or "error".
            status_code (in dict): HTTP status code indicating the result of the processing.
            content (in dict): Extracted content from the file, or an error message.
        """
        try:
            full_path = ROOT_PATH / input_path
            if not full_path.exists():
                return {
                    "status": "error",
                    "status_code": HTTPStatus.NOT_FOUND,
                    "error": f"File not found: {input_path}",
                }

            file_ext = full_path.suffix.lower()

            if file_ext in InputProcessor.TEXT_EXT:
                content = InputProcessor._process_text_file(full_path)
            elif file_ext in InputProcessor.AUDIO_EXT:
                content = InputProcessor._process_audio_file(full_path)
            else:
                return {
                    "status": "error",
                    "status_code": HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    "error": f"Unsupported file type: {file_ext}",
                }

            return {
                "status": "success",
                "status_code": HTTPStatus.OK,
                "content": content,
            }

        except Exception as e:
            return {
                "status": "error",
                "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
                "error": str(e),
            }

    @staticmethod
    def _process_text_file(file_path: Path) -> str:
        """
        Extract text from text-based files like `.pdf`
        """
        with pymupdf.open(file_path) as doc:
            return "".join(page.get_text() for page in doc)

    @staticmethod
    def _format_audio_segments(segments_data: dict) -> str:
        """
        Format audio segments into readable text
        """
        segments = segments_data.get("segments", [])

        return "\n".join(
            f"{segment.get('speaker', 'UNKNOWN')}: {text}"
            for segment in segments
            if (text := segment.get("text", "").strip())
        )

    @staticmethod
    def _process_audio_file(file_path: Path) -> str:
        audio_processor = AudioProcessor(quiet=True)
        segments_data = audio_processor.process(file_path)
        return InputProcessor._format_audio_segments(segments_data)
