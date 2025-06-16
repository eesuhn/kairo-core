import pytest
import warnings

from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_pdf_agile_methodology():
    pdf_path = DATA_DIR / "sample-pdf" / "agile-methodology.pdf"
    if not pdf_path.exists():
        pytest.skip(f"PDF file not found: {pdf_path}")
    return str(pdf_path)
