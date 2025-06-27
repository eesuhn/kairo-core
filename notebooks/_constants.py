import os

from dotenv import load_dotenv
from pathlib import Path


ROOT_PATH = Path.cwd().parent
DATA_DIR = ROOT_PATH / "data"
SAMPLE_DIR = DATA_DIR / "sample"
MODEL_DIR = ROOT_PATH / "models"

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
