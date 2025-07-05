import pymupdf
import justsdk

from pathlib import Path
from typing import Union
from .audio_processor import AudioProcessor


class InputProcessor:
    TEXT_EXT = (".pdf", ".docx")
    AUDIO_EXT = (".mp3", ".mp4")

    def __init__(self) -> None:
        self.ap = AudioProcessor()

    def process(self, input_file: Path) -> Union[str, dict]:
        """
        Process the input file based on its type:
            - Text files eg. `.pdf`, `.docx`
            - Audio files eg. `.mp3`, `.mp4`
        """
        if not input_file.exists():
            raise FileNotFoundError(f"File not found: {input_file}")

        file_ext = input_file.suffix.lower()
        if file_ext in self.TEXT_EXT:
            return self._process_text_file(input_file)
        elif file_ext in self.AUDIO_EXT:
            return self._process_audio_file(input_file)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def _process_text_file(self, file_path: Path) -> str:
        """
        Extract text from text-based files like `.pdf`
        """
        justsdk.print_info(f"Processing text file: {file_path}")
        content = pymupdf.open(file_path)
        text = "".join([content.load_page(i).get_text() for i in range(len(content))])
        return text.replace("\n", " ").replace("\r", " ")

    def _process_audio_file(self, file_path: Path) -> dict:
        return self.ap.process(file_path)
