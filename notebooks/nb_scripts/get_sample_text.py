import warnings
import pymupdf

from _constants import SAMPLE_DIR


warnings.filterwarnings("ignore")
TARGET_DIR = SAMPLE_DIR / "text"


def check_sample_files() -> dict:
    sample_ext = [".pdf", ".docx"]
    samples = [sample for ext in sample_ext for sample in TARGET_DIR.glob(f"*{ext}")]
    samples_dict = {}
    for sample in samples:
        size_mb = sample.stat().st_size / (1024 * 1024)
        samples_dict[sample.name] = {
            "path": sample,
            "size_mb": size_mb,
        }
    return samples_dict


def get_sample_text(filename: str, clean: bool = False) -> str:
    samples = check_sample_files()
    if filename not in samples:
        raise ValueError(f"Sample '{filename}' not found.")

    target_doc = pymupdf.open(samples[filename]["path"])
    text_target_doc = "".join(
        [target_doc.load_page(i).get_text() for i in range(len(target_doc))]
    )
    if clean:
        return text_target_doc.replace("\n", " ").replace("\r", " ")
    return text_target_doc
