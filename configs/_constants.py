import os

from dotenv import load_dotenv
from pathlib import Path


ROOT_PATH = Path.cwd()
CONFIGS_DIR = ROOT_PATH / "configs"
DATA_DIR = ROOT_PATH / "data"
INTER_DATA_DIR = DATA_DIR / "interim"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODEL_DIR = ROOT_PATH / "models"
NB_PATH = ROOT_PATH / "notebooks"
REPORTS_DIR = ROOT_PATH / "reports"

load_dotenv()
HF_READ_ONLY_TOKEN = os.getenv("HF_READ_ONLY_TOKEN")
