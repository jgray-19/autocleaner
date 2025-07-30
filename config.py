import json
import os
from datetime import datetime

from generic_parser.tools import DotDict

# General Settings
NUM_NOISY_PER_CLEAN = 10
LOAD_MODEL = False
RESUME_FROM_CKPT = True
NUM_EPOCHS = 2000

if RESUME_FROM_CKPT:
    # CONFIG_NAME = "2025-07-04_17-52-43" # 1000 mseloss
    # CONFIG_NAME = "2025-07-05_21-47-29" # 1_000_000 mseloss
    # CONFIG_NAME = "2025-07-06_00-34-00"  # Just mse, no ssp
    # CONFIG_NAME = "2025-07-06_01-27-53"  # 0 missing prob, 1e-4 noise, mse loss

    # CONFIG_NAME = "2025-07-06_18-00-02"
    # CONFIG_NAME = "2025-07-07_18-32-46"  # Better learning rate scheduler

    # CONFIG_NAME = "2025-07-09_08-45-04"
    # CONFIG_NAME = "2025-07-10_10-57-09"  # Better scheduler, only 1e-4 noise, batch size 1000 (from 400)

    # CONFIG_NAME = "2025-07-14_18-17-15"  # Now using residuals
    # CONFIG_NAME = "2025-07-14_18-51-31"  # Residuals, 0.5 alpha, 16 base channels

    # CONFIG_NAME = "2025-07-15_12-17-23" # 60 lots of data, more noise factors, custom scheduler, more kicks (even x & y hopefully)

    # CONFIG_NAME = "2025-07-15_17-03-07" # Using an identity penalty + loss with residuals only. 

    # CONFIG_NAME = "2025-07-15_22-39-40" # Back to comb_ssp - No recreating the spectrum. 

    # CONFIG_NAME = "2025-07-16_07-25-15" # Using residuals again, but no identity and using spectral convergence loss
    
    # CONFIG_NAME = "2025-07-16_20-02-57" # Working with B2 now, 12 base channels, 100 -> 500 -> 1000 turns

    # CONFIG_NAME = "2025-07-17_13-53-50"

    CONFIG_NAME = "2025-07-24_10-54-43"
else:
    CONFIG_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Data Settings
NBPMS = 564
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
COUPLING = [False, 1e-4, 1e-3]
KICK_AMPS = [1e-4, 5e-5]

BEAMS = [1, 2]
CLEAN_PARAM_LIST = []
for beam in BEAMS:
    for tunes in TUNE_LIST:
        for coupling in COUPLING:
            for kick_amp in KICK_AMPS:
                if beam == 2 and tunes == [0.69, 0.68]:
                    continue  # Skip this combination as it is unstable.
                CLEAN_PARAM_LIST.append(
                    {
                        "beam": beam,
                        "tunes": tunes,
                        "coupling": coupling,
                        "kick_amp": kick_amp,
                    }
                )
NUM_PARAMS = len(CLEAN_PARAM_LIST)

BATCH_SIZE = 4
# ACCUMULATE_BATCHES = (
#     NUM_PARAMS * NUM_NOISY_PER_CLEAN // (BATCH_SIZE * 5)
# )  # 5 so I get 5 steps per epoch
ACCUMULATE_BATCHES = 40
TRAIN_RATIO = 0.8

MODEL_SAVE_PATH = "conv_autoencoder.pth"
NLOGSTEPS = 1

# NUM_PLANES = 2
NUM_CHANNELS = 1
PRECISION = "32-true"

BOTTLENECK_SIZE = 4
BASE_CHANNELS = 12

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
IDENTITY_PENALTY = 0

ALPHA = 0.5 # For ssp ALPHA*mse_loss + (1 - ALPHA)*ssp_loss

DENOISED_INDEX = "denoised"
HARPY_CLEAN_INDEX = "harpy_cleaned"
SAMPLE_INDEX = "noisy"
NONOISE_INDEX = "zero_noise"

NOISE_FACTORS = [5e-4, 3e-4, 1e-4, 5e-5, 1e-5, 5e-6]

# MODEL_TYPE = "leaky"
MODEL_TYPE = "unet_fixed"
MODEL_DEPTH = 4
RESIDUALS = True

LOSS_TYPE = "comb_ssp_resid"
# LOSS_TYPE = "residual"
SCHEDULER = True
MIN_LR = 1e-4
# NUM_CONSTANT_LR_EPOCHS = 100
NUM_DECAY_EPOCHS = 1000

DATA_SCALING = "meanstd"
MISSING_PROB = 0.01  # 1 % chance of data missing

INIT = "xavier"

experiment_config = {
    "accumulate_batches": ACCUMULATE_BATCHES,
    "alpha": ALPHA if LOSS_TYPE in ["fft", "combined", "comb_ssp", "residual"] else None,
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
    "identity_penalty": IDENTITY_PENALTY,
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
