import os
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_ROOT", "/Volumes/WX"))
DATA = ROOT / "scratch"
STEPS = DATA / "steps_data"
PREPARES = DATA / "prepares_data"
FIGURES = DATA / "figures_data"
OUTPUT = ROOT / "outputs"
CODE = Path(__file__).resolve().parent
