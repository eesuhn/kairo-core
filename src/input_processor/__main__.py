import pymupdf
import warnings

from pathlib import Path
from .audio_processor import AudioProcessor
from configs._constants import ROOT_PATH

warnings.filterwarnings("ignore")


class InputProcessor:
    # TODO: To support markdown and .txt files
    TEXT_EXT = (".pdf", ".docx")
    AUDIO_EXT = (".mp3", ".mp4")

    @staticmethod
    def process(input_path: Path) -> str:
        """
        Process the input file based on supported formats
        """
        full_path = ROOT_PATH / input_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")

        file_ext = full_path.suffix.lower()

        if file_ext in InputProcessor.TEXT_EXT:
            return InputProcessor._process_text_file(full_path)
        elif file_ext in InputProcessor.AUDIO_EXT:
            return InputProcessor._process_audio_file(full_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

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
