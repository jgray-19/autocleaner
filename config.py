import json
import os
from datetime import datetime
from math import floor
from generic_parser.tools import DotDict

# General Settings
NUM_NOISY_PER_CLEAN = 200
LOAD_MODEL = False
RESUME_FROM_CKPT = True
if RESUME_FROM_CKPT:
    CONFIG_NAME = "2025-03-21_09-20-38"
else:
    CONFIG_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Data Settings
NBPMS = 563
TOTAL_TURNS = 2000  # Total turns in the simulated data file
NTURNS = 1500  # Training window length

# Various Clean Data Settings
TUNE_LIST = [
    [0.19, 0.18],
    [0.28, 0.29],
    [0.31, 0.32],
    [0.43, 0.49],
]
COUPLING = [False, 1e-4, 1e-3]
BEAMS = [1, 2]
CLEAN_PARAM_LIST = []
for beam in BEAMS:
    for tunes in TUNE_LIST:
        for coupling in COUPLING:
            CLEAN_PARAM_LIST.append(
                {
                    "beam": beam,
                    "tunes": tunes,
                    "coupling": coupling,
                }
            )


BATCH_SIZE = 4
ACCUMULATE_BATCHES = 10
TRAIN_RATIO = 0.8

MODEL_SAVE_PATH = "conv_autoencoder.pth"
NLOGSTEPS = max(floor(TRAIN_RATIO * NUM_NOISY_PER_CLEAN / BATCH_SIZE), 1)

# NUM_PLANES = 2
NUM_CHANNELS = 1
PRECISION = "16-mixed"

NUM_EPOCHS = 5000
BOTTLENECK_SIZE = 4
BASE_CHANNELS = 12

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

ALPHA = 0.5

DENOISED_INDEX = "denoised"
SAMPLE_INDEX = "noisy"
NONOISE_INDEX = "zero_noise"

# NOISE_FACTORS = [1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 5e-5]
NOISE_FACTORS = [1e-3, 5e-4, 1e-4, 5e-5]
# NOISE_FACTORS = [5e-4, 1e-4]


# MODEL_TYPE = "leaky"
MODEL_TYPE = "unet_fixed"
MODEL_DEPTH = 4
RESIDUALS = False

LOSS_TYPE = "comb_ssp"
# LOSS_TYPE = "mse"
SCHEDULER = True
MIN_LR = 1e-5

INIT = "xavier"
DATA_SCALING = "minmax"

experiment_config = {
    "accumulate_batches": ACCUMULATE_BATCHES,
    "alpha": ALPHA if LOSS_TYPE in ["fft", "combined"] else None,
    "base_channels": BASE_CHANNELS,
    "batch_size": BATCH_SIZE,
    "beams": BEAMS,
    "bottleneck_size": BOTTLENECK_SIZE if MODEL_TYPE != "deep" else None,
    "coupling": COUPLING,
    "data_scaling": DATA_SCALING,
    "depth": MODEL_DEPTH if MODEL_TYPE in ["unet", "fno"] else None,
    "learning_rate": LEARNING_RATE,
    "load_model": LOAD_MODEL,
    "loss_type": LOSS_TYPE,
    "min_lr": MIN_LR if SCHEDULER else None,
    "nbpms": NBPMS,
    "noise_factor": NOISE_FACTORS,
    "num_channels": NUM_CHANNELS,
    "num_epochs": NUM_EPOCHS,
    "num_files": NUM_NOISY_PER_CLEAN,
    "precision": PRECISION,
    "residuals": RESIDUALS,
    "scheduler": SCHEDULER,
    "tune_list": TUNE_LIST,
    "total_turns": TOTAL_TURNS,
    "train_ratio": TRAIN_RATIO,
    "weight_decay": WEIGHT_DECAY,
}
# Remove None values from the dictionary
experiment_config = {k: v for k, v in experiment_config.items() if v is not None}

HARPY_INPUT = DotDict(
    {
        "turn_bits": 10,
        "output_bits": 10,
        "window": "hann",
        "to_write": ["lin", "full_spectra"],
        "tunes": [0.28, 0.31, 0.0],
        "natdeltas": [0.0, -0.0, 0.0],
        "resonances": 4,
        "tolerance": 0.01,
    }
)

# Set seed for reproducibility
SEED = 42


def print_config():
    print("Configuration:")
    for key, value in globals().items():
        if key.isupper():
            print(f"{key}: {value}")


def save_experiment_config(output_dir, config_name=CONFIG_NAME + "_config.json"):
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, config_name)
    with open(config_path, "w") as f:
        json.dump(experiment_config, f, indent=4)
    return config_path
