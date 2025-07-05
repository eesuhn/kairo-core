import pytest
import justsdk

from ._constants import SAMPLE_DATA_DIR
from src.input_processor import InputProcessor
from pathlib import Path


@pytest.fixture(scope="session")
def target_audio_file() -> Path:
    return SAMPLE_DATA_DIR / "audio" / "project-proposal.mp3"


@pytest.fixture(scope="session")
def result_audio_file(target_audio_file) -> dict:
    ip = InputProcessor()
    return ip.process(target_audio_file)


def test_audio_processor(result_audio_file, capsys) -> None:
    # write_result_audio_file(result_audio_file)

    captured = capsys.readouterr()
    print(captured.out)


def write_result_audio_file(result: dict) -> None:
    target_file_path = SAMPLE_DATA_DIR / "audio" / "script" / "project-proposal.json"
    justsdk.write_file(result, file_path=target_file_path, use_orjson=True, atomic=True)
    justsdk.print_info(f"Result written to: {target_file_path}")
