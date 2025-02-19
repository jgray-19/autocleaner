import os
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

ANALYSIS_DIR = CURRENT_DIR / "analysis"
FREQ_OUT_DIR = ANALYSIS_DIR / "lin_files"
DATA_DIR = CURRENT_DIR / "data"
ACC_MODELS = CURRENT_DIR / "acc-models-lhc"
PLOT_DIR = CURRENT_DIR / "plots"
if not ACC_MODELS.exists():
    os.system(f"ln -s /afs/cern.ch/eng/acc-models/lhc/2024/ {ACC_MODELS}")

NORMAL_SEXTUPOLE_RDTS: tuple[str] = (
    "f1200_x",
    "f3000_x",
    "f1002_x",
    "f1020_x",
    "f0111_y",
    "f0120_y",
    "f1011_y",
    "f1020_y",
)
SKEW_SEXTUPOLE_RDTS: tuple[str] = (
    "f0012_y",
    "f0030_y",
    "f1101_x",
    "f1110_x",
    "f2001_x",
    "f2010_x",
    "f0210_y",
    "f2010_y",
)