import os
import sys

from pathlib import Path


ROOT = Path().resolve().parent
sys.path.append(str(ROOT))
os.chdir(ROOT)
