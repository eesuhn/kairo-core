import pymupdf
import justsdk

from pathlib import Path
from typing import Union
from .audio_processor import AudioProcessor
from configs._constants import ROOT_PATH


class InputProcessor:
    # TODO: To support markdown and .txt files
    TEXT_EXT = (".pdf", ".docx")
    AUDIO_EXT = (".mp3", ".mp4")

    @staticmethod
    def process(input_path: Path, quiet: bool = False) -> Union[str, dict]:
        """
        Process the input file based on supported formats
        """
        input_path = ROOT_PATH / input_path
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        file_ext = input_path.suffix.lower()
        if file_ext in InputProcessor.TEXT_EXT:
            return InputProcessor._process_text_file(input_path, quiet=quiet)

        elif file_ext in InputProcessor.AUDIO_EXT:
            return InputProcessor._process_audio_file(input_path, quiet=quiet)

        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def _process_text_file(file_path: Path, quiet: bool = False) -> str:
        """
        Extract text from text-based files like `.pdf`
        """
        if not quiet:
            justsdk.print_info(
                f"Processing text file: {file_path}", newline_before=True
            )
        content = pymupdf.open(file_path)
        text = "".join([content.load_page(i).get_text() for i in range(len(content))])
        return text

    def _process_audio_file(file_path: Path) -> dict:
        ap = AudioProcessor()
        return ap.process(file_path)
