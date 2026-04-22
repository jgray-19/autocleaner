import json
import os
from datetime import datetime

from generic_parser.tools import DotDict

from project_paths import get_model_dir

# General Settings
BEAM = 1
NUM_FILES = 500
LOAD_MODEL = False
RESUME_FROM_CKPT = False
if RESUME_FROM_CKPT:
    # CONFIG_NAME = "2025-03-12_10-35-48" # First Long training with ideal
    # CONFIG_NAME = "2025-03-13_09-39-18" # Added mse to the loss (comb_ssp instead of ssp)
    # CONFIG_NAME = "2025-03-13_13-50-14" # Residuals
    # CONFIG_NAME = "2025-03-13_14-08-38" # Residuals but better
    # CONFIG_NAME = "2025-03-13_18-21-08" # back to 2 but more files, more base channels, more files, smaller batches
    # CONFIG_NAME = "2025-03-17_09-47-23" # Fixed noise level, noise with beta functions 10 um.
    # CONFIG_NAME = "2025-03-17_16-31-06"  # Above but 100 um
    # CONFIG_NAME = "2025-03-18_17-07-40" # Above but now doing many noises on updated thing
    # CONFIG_NAME = "2025-03-19_22-46-55" # Above but tenth the initial learning rate. Also split x and y.
    # CONFIG_NAME = "2025-03-20_09-13-50" # Above but half same noise and 8 base channels

    # CONFIG_NAME = '2025-03-20_15-37-57' # See config
    # CONFIG_NAME = '2025-03-20_15-40-29' # Above but residuals
    # CONFIG_NAME = '2025-03-20_20-56-06' # 24 base channels, 5 batch size and lower learning rate

    # CONFIG_NAME = '2025-03-21_09-20-38'

    # CONFIG_NAME = '2026-04-14_08-34-32' # Single tune set

    # CONFIG_NAME = '2026-04-17_14-18-36' # Two tunes, the training window is split among number of windows not files. aa

    CONFIG_NAME = "2026-04-19_17-52-29" # Higher LR, less noisy, slightly larger model, schedular off, alpha=0.01

    # CONFIG_NAME = "2026-04-21_12-42-43"
else:
    CONFIG_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Data Settings
NBPMS = 559
TOTAL_TURNS = 6600  # Total turns in the simulated data file
NTURNS = 1000  # Training window length

BATCH_SIZE = 25
ACCUMULATE_BATCHES = 2
TRAIN_RATIO = 0.8

NUM_SAME_NOISE = 1
NUM_SAME_OFFSET = 1

MODEL_SAVE_PATH = "conv_autoencoder.pth"
MODEL_DIR = get_model_dir(beam=BEAM)

NLOGSTEPS = 16

# NUM_PLANES = 2
NUM_CHANNELS = 1
# Use mixed precision for Tensor Core throughput on modern NVIDIA GPUs.
PRECISION = "bf16-mixed"

NUM_EPOCHS = 10_000
BOTTLENECK_SIZE = 4
BASE_CHANNELS = 5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

ALPHA = 0.01

# Improved comb_ssp_norm controls:
# - NOISE_NORM_GAMMA < 1 softens inverse-variance weighting to reduce
#   over-dominance from the lowest-noise samples.
# - LOW_NOISE_IDENTITY_WEIGHT enforces near-identity behavior for low-noise inputs.
# - LOW_NOISE_QUANTILE selects which samples in a batch are considered low-noise.
NOISE_NORM_GAMMA = 0.75
LOW_NOISE_IDENTITY_WEIGHT = 0.1
LOW_NOISE_QUANTILE = 0.3

DENOISED_INDEX = "denoised"
SAMPLE_INDEX = "noisy"
NONOISE_INDEX = "zero_noise"

# NOISE_FACTORS = [1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 5e-5]
NOISE_FACTORS = [5e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7]
# NOISE_FACTORS = [5e-4, 1e-4]

# MODEL_TYPE = "leaky"
MODEL_TYPE = "unet_fixed"
MODEL_DEPTH = 4
RESIDUALS = False

LOSS_TYPE = "comb_ssp_norm"
# LOSS_TYPE = "mse"
SCHEDULER = False
MIN_LR = 5e-4

INIT = "identity"
DATA_SCALING = "minmax"
USE_OFFSETS = True

experiment_config = {
    "beam": BEAM,
    "num_files": NUM_FILES,
    "load_model": LOAD_MODEL,
    "nbpms": NBPMS,
    "nturns": NTURNS,
    "total_turns": TOTAL_TURNS,
    # "num_same_noise": NUM_SAME_NOISE,
    "batch_size": BATCH_SIZE,
    "accumulate_batches": ACCUMULATE_BATCHES,
    "train_ratio": TRAIN_RATIO,
    # "num_planes": NUM_PLANES,
    "num_channels": NUM_CHANNELS,
    "num_epochs": NUM_EPOCHS,
    "base_channels": BASE_CHANNELS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "noise_factor": NOISE_FACTORS,
    "model_type": MODEL_TYPE,
    "loss_type": LOSS_TYPE,
    # "fft_weight": FFT_WEIGHT,
    "precision": PRECISION,
    "scheduler": SCHEDULER,
    "data_scaling": DATA_SCALING,
    "use_offsets": USE_OFFSETS,
    "residuals": RESIDUALS,
}
if MODEL_TYPE != "deep":
    experiment_config["bottleneck_size"] = BOTTLENECK_SIZE
if MODEL_TYPE == "unet" or MODEL_TYPE == "fno":
    experiment_config["depth"] = MODEL_DEPTH

if LOSS_TYPE == "fft" or LOSS_TYPE == "combined":
    experiment_config["alpha"] = ALPHA
if LOSS_TYPE == "comb_ssp_norm":
    experiment_config["alpha"] = ALPHA
    experiment_config["noise_norm_gamma"] = NOISE_NORM_GAMMA
    experiment_config["low_noise_identity_weight"] = LOW_NOISE_IDENTITY_WEIGHT
    experiment_config["low_noise_quantile"] = LOW_NOISE_QUANTILE
if SCHEDULER:
    experiment_config["min_lr"] = MIN_LR


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
