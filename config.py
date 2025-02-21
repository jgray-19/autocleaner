from pathlib import Path
from generic_parser.tools import DotDict

CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
PLOT_DIR = CURRENT_DIR / "plots"

def get_model_dir(beam: int) -> Path:
    """Return the model directory for the given test parameters."""
    return CURRENT_DIR / f"model_b{beam}"

def get_file_suffix(beam: int, nturns: int) -> str:
    """Return the file suffix for the test files based on beam and order."""
    assert beam in [1, 2], "Beam must be 1 or 2"
    return f"b{beam}_{nturns}t"

def get_tbt_path(beam: int, nturns: int, index: int) -> Path:
    """Return the name of the TBT file for the given test parameters."""
    suffix = get_file_suffix(beam, nturns) + (f"_{index}" if index != -1 else "_zero_noise")
    return DATA_DIR / f"tbt_{suffix}.sdds"

# General Settings
BEAM = 1
NUM_FILES = 100
LOAD_MODEL = False

# Data Settings
NBPMS = 563
NTURNS = 1000
BATCH_SIZE = 20
TRAIN_RATIO = 0.8
MODEL_SAVE_PATH = "conv_autoencoder.pth"
MODEL_DIR = get_model_dir(beam=BEAM)

NUM_PLANES = 2
NUM_CHANNELS = NBPMS

# Optimisation Settings
NUM_EPOCHS = 10
BOTTLENECK_SIZE = 64
BASE_CHANNELS = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

BPM_DIFF_WEIGHT = 1e-1
TURN_DIFF_WEIGHT = 1e-1

ALPHA = 0.5

DENOISED_INDEX = -2

NOISE_FACTOR=1e-8

# Additional weight for the frequency component
FFT_WEIGHT = 0#1e-4 # Time loss initially on order of 0.2, Freq loss on order of 11.7 -> so 11.7*1e-4 = 1.17e-3 -> around 100 times smaller

STRING_PREFIX = (
    f"beam{BEAM}_"
    + f"fft{FFT_WEIGHT:.1e}_"
    + f"{BPM_DIFF_WEIGHT:.1e}diff_"
    + f"{TURN_DIFF_WEIGHT:.1e}turn_"
    + f"alpha{ALPHA}_"
    + f"btnk{BOTTLENECK_SIZE}_"
    + f"ch{BASE_CHANNELS}_"
    + f"lr{LEARNING_RATE}_"
    + f"wd{WEIGHT_DECAY}_"
    + f"files{NUM_FILES}_"
    + f"ep{NUM_EPOCHS}_"
    + f"bs{BATCH_SIZE}_"
    + f"noise{NOISE_FACTOR}_"
)

HARPY_INPUT = DotDict({
    "turn_bits": 10,
    "output_bits": 10,
    "window": "hann",
    "to_write": ["lin", "full_spectra"],
    "tunes": [0.28, 0.31, 0.0],
    "natdeltas": [0.0, -0.0, 0.0],
    "resonances": 4,
    "tolerance": 0.01,
})

# Set seed for reproducibility
SEED = 42

def print_config():
    print("Configuration:")
    for key, value in globals().items():
        if key.isupper():
            print(f"{key}: {value}")