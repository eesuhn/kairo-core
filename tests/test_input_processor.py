import pytest
import justsdk

from pathlib import Path
from configs._constants import SAMPLE_DATA_DIR, REPORTS_DIR
from src.input_processor import InputProcessor


SAMPLE_AUDIO_TARGET = "project-proposal"
SAMPLE_TEXT_TARGET = "agile-method"


@pytest.fixture(scope="session")
def sample_audio_file() -> Path:
    return SAMPLE_DATA_DIR / "audio" / f"{SAMPLE_AUDIO_TARGET}.mp3"


@pytest.fixture(scope="session")
def audio_processing_result(sample_audio_file: Path) -> dict:
    processor = InputProcessor()
    return processor.process(sample_audio_file)


@pytest.fixture(scope="session")
def sample_text_file() -> Path:
    return SAMPLE_DATA_DIR / "text" / f"{SAMPLE_TEXT_TARGET}.pdf"


@pytest.fixture(scope="session")
def text_processing_result(sample_text_file: Path) -> dict:
    processor = InputProcessor()
    return processor.process(sample_text_file)


def test_audio_processor(audio_processing_result: dict, capsys) -> None:
    captured = capsys.readouterr()
    print(captured.out)

    assert audio_processing_result is not None
    assert "transcription" in audio_processing_result
    assert "diarization" in audio_processing_result

    _save_audio_processing_result(audio_processing_result)


def _save_audio_processing_result(result: dict) -> None:
    output_path = REPORTS_DIR / "sample" / "audio" / f"{SAMPLE_AUDIO_TARGET}.json"
    justsdk.write_file(result, file_path=output_path, use_orjson=True, atomic=True)
    justsdk.print_info(f"Result written to: {output_path}")


def test_text_processor(text_processing_result: dict, capsys) -> None:
    captured = capsys.readouterr()
    print(captured.out)

    assert text_processing_result is not None

    _save_text_processing_result(text_processing_result)


def _save_text_processing_result(result: dict) -> None:
    output_path = REPORTS_DIR / "sample" / "text" / f"{SAMPLE_TEXT_TARGET}.md"
    justsdk.write_file(result, file_path=output_path, atomic=True)
    justsdk.print_info(f"Result written to: {output_path}")
