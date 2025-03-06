import json
import os
from datetime import datetime
from pathlib import Path
from math import floor

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
NUM_FILES = 200
LOAD_MODEL = False

# Data Settings
NBPMS = 563
TOTAL_TURNS = 1500   # Total turns in the simulated data file
NTURNS = 1000        # Training window length
BATCH_SIZE = 10
TRAIN_RATIO = 0.8
MODEL_SAVE_PATH = "conv_autoencoder.pth"
MODEL_DIR = get_model_dir(beam=BEAM)

NLOGSTEPS = max(floor(TRAIN_RATIO * NUM_FILES / BATCH_SIZE), 1)

NUM_PLANES = 2
NUM_CHANNELS = NUM_PLANES

# Optimisation Settings
NUM_EPOCHS = 1500
BOTTLENECK_SIZE = 32
BASE_CHANNELS = 24
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

ALPHA = 0.5

DENOISED_INDEX = -2
SAMPLE_INDEX = 2

NOISE_FACTOR = 1e-5

MODEL_TYPE = "unet_fixed"
MODEL_DEPTH = 3

LOSS_TYPE = "ssp"
SCHEDULER = True
MIN_LR = 5e-5
INIT = "xavier"
DATA_SCALING = "minmax"
USE_OFFSETS = True

experiment_config = {
    "beam": BEAM,
    "num_files": NUM_FILES,
    "load_model": LOAD_MODEL,
    "nbpms": NBPMS,
    "nturns": NTURNS,
    "batch_size": BATCH_SIZE,
    "train_ratio": TRAIN_RATIO,
    "num_planes": NUM_PLANES,
    "num_channels": NUM_CHANNELS,
    "num_epochs": NUM_EPOCHS,
    "base_channels": BASE_CHANNELS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "noise_factor": NOISE_FACTOR,
    "model_type": MODEL_TYPE,
    "loss_type": LOSS_TYPE,
    # "fft_weight": FFT_WEIGHT,
    "scheduler": SCHEDULER,
    "data_scaling": DATA_SCALING,
    "use_offsets": USE_OFFSETS,
}
if MODEL_TYPE != "deep":
    experiment_config["bottleneck_size"] = BOTTLENECK_SIZE
if MODEL_TYPE == "unet" or MODEL_TYPE == "fno":
    experiment_config["depth"] = MODEL_DEPTH

if LOSS_TYPE == "fft" or LOSS_TYPE == "combined":
    experiment_config["alpha"] = ALPHA
if SCHEDULER:
    experiment_config["min_lr"] = MIN_LR


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

CONFIG_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_experiment_config(output_dir, config_name= CONFIG_NAME + "_config.json"):
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, config_name)
    with open(config_path, "w") as f:
        json.dump(experiment_config, f, indent=4)
    return config_path