import pymupdf

from pathlib import Path
from ._constants import SAMPLE_DATA_DIR
from typing import Union


class InputProcessing:
    TEXT_EXT = (".pdf", ".docx")
    AUDIO_EXT = (".mp3", ".mp4")

    def process(self, input_file: str) -> Union[str, dict]:
        file_path = Path(input_file)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {input_file}")

        file_ext = file_path.suffix.lower()
        if file_ext in self.TEXT_EXT:
            return self._process_text_file(file_path)
        elif file_ext in self.AUDIO_EXT:
            return self._process_audio_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def _process_text_file(self, file_path: Path) -> str:
        content = pymupdf.open(file_path)
        text = "".join([content.load_page(i).get_text() for i in range(len(content))])
        return text.replace("\n", " ").replace("\r", " ")

    def _process_audio_file(self, file_path: Path) -> dict:
        pass


if __name__ == "__main__":
    processor = InputProcessing()
    result = processor.process(SAMPLE_DATA_DIR / "text" / "agile-method.pdf")
    print(result)
