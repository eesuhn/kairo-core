import pytest
import justsdk

from pathlib import Path
from ._constants import SAMPLE_DATA_DIR, REPORTS_DIR
from src.input_processor import InputProcessor


SAMPLE_AUDIO_TARGET = "project-proposal"


@pytest.fixture(scope="session")
def sample_audio_file() -> Path:
    return SAMPLE_DATA_DIR / "audio" / f"{SAMPLE_AUDIO_TARGET}.mp3"


@pytest.fixture(scope="session")
def audio_processing_result(sample_audio_file: Path) -> dict:
    processor = InputProcessor()
    return processor.process(sample_audio_file)


def test_audio_processor(audio_processing_result: dict, capsys) -> None:
    captured = capsys.readouterr()
    print(captured.out)

    assert audio_processing_result is not None
    assert "transcription" in audio_processing_result
    assert "diarization" in audio_processing_result

    _save_audio_processing_result(audio_processing_result)


def _save_audio_processing_result(result: dict) -> None:
    output_path = REPORTS_DIR / "audio" / f"{SAMPLE_AUDIO_TARGET}.json"
    justsdk.write_file(result, file_path=output_path, use_orjson=True, atomic=True)
    justsdk.print_info(f"Result written to: {output_path}")
