import json
import os
from datetime import datetime

from generic_parser.tools import DotDict

# General Settings
NUM_NOISY_PER_CLEAN = 20
LOAD_MODEL = False
RESUME_FROM_CKPT = False
NUM_EPOCHS = 1000

if RESUME_FROM_CKPT:
    # CONFIG_NAME = "2025-07-04_17-52-43" # 1000 mseloss
    # CONFIG_NAME = "2025-07-05_21-47-29" # 1_000_000 mseloss
    # CONFIG_NAME = "2025-07-06_00-34-00"  # Just mse, no ssp
    # CONFIG_NAME = "2025-07-06_01-27-53"  # 0 missing prob, 1e-4 noise, mse loss

    # CONFIG_NAME = "2025-07-06_18-00-02"
    # CONFIG_NAME = "2025-07-07_18-32-46"  # Better learning rate scheduler

    # CONFIG_NAME = "2025-07-09_08-45-04"
    # CONFIG_NAME = "2025-07-10_10-57-09"  # Better scheduler, only 1e-4 noise, batch size 1000 (from 400)

    CONFIG_NAME = "2025-07-14_13-44-01"  # Now using mask
else:
    CONFIG_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Data Settings
NBPMS = 563
TOTAL_TURNS = 3000  # Total turns in the simulated data file
NTURNS = 1000  # Training window length

# Various Clean Data Settings
TUNE_LIST = [
    # [0.07, 0.06],
    # [0.19, 0.18],
    # [0.26, 0.27],
    [0.27, 0.28],
    # [0.27, 0.29],
    # [0.27, 0.31],
    # [0.27, 0.32],
    [0.28, 0.29],
    [0.28, 0.31],
    [0.28, 0.32],
    [0.29, 0.31],
    [0.29, 0.32],
    [0.31, 0.32],
    # [0.32, 0.31],
    # [0.43, 0.49],
    # [0.74, 0.73],
    [0.72, 0.69],
    [0.71, 0.68],
    [0.69, 0.68],
]
COUPLING = [1e-4, 1e-3]
KICK_AMPS = [1e-4]

BEAMS = [1]
CLEAN_PARAM_LIST = []
for beam in BEAMS:
    for tunes in TUNE_LIST:
        for coupling in COUPLING:
            for kick_amp in KICK_AMPS:
                CLEAN_PARAM_LIST.append(
                    {
                        "beam": beam,
                        "tunes": tunes,
                        "coupling": coupling,
                        "kick_amp": kick_amp,
                    }
                )
NUM_PARAMS = len(CLEAN_PARAM_LIST)

BATCH_SIZE = 5
ACCUMULATE_BATCHES = (
    NUM_PARAMS * NUM_NOISY_PER_CLEAN // (BATCH_SIZE * 2)
)  # 2 so I get 2 logs per epoch
TRAIN_RATIO = 0.8

MODEL_SAVE_PATH = "conv_autoencoder.pth"
NLOGSTEPS = 1

# NUM_PLANES = 2
NUM_CHANNELS = 1
PRECISION = "32-true"

BOTTLENECK_SIZE = 4
BASE_CHANNELS = 12

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

ALPHA = 0.99  # For ssp ALPHA*mse_loss + (1 - ALPHA)*ssp_loss

DENOISED_INDEX = "denoised"
HARPY_CLEAN_INDEX = "harpy_cleaned"
SAMPLE_INDEX = "noisy"
NONOISE_INDEX = "zero_noise"

NOISE_FACTORS = [1e-4, 1e-5]

# MODEL_TYPE = "leaky"
MODEL_TYPE = "unet_fixed"
MODEL_DEPTH = 4
RESIDUALS = False  # we are NOT predicting additive residuals
USE_MASK = True  # tell the pl-module to form  x̂ = M ⊙ x_noisy

LOSS_TYPE = "comb_ssp"
# LOSS_TYPE = "mse"
SCHEDULER = True
MIN_LR = 1e-5
NUM_CONSTANT_LR_EPOCHS = 100
NUM_DECAY_EPOCHS = 200

DATA_SCALING = "meanstd"
MISSING_PROB = 0

INIT = "weiner"
MASK_MAX_GAIN = 5.0  # Set to a float to clamp mask, e.g. 2.0
LAMBDA_SPEC = 1.0  # weight of spec‐magnitude loss vs time‐MSE

experiment_config = {
    "accumulate_batches": ACCUMULATE_BATCHES,
    "alpha": ALPHA if LOSS_TYPE in ["fft", "combined", "comb_ssp"] else None,
    "base_channels": BASE_CHANNELS,
    "batch_size": BATCH_SIZE,
    "beams": BEAMS,
    "bottleneck_size": BOTTLENECK_SIZE if MODEL_TYPE != "deep" else None,
    "coupling": COUPLING,
    "data_scaling": DATA_SCALING,
    "depth": MODEL_DEPTH if MODEL_TYPE in ["unet", "fno"] else None,
    "lambda_spec": LAMBDA_SPEC,
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
    "Noise_decay": True,
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
