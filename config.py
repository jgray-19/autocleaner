import json
import os
from datetime import datetime
from generic_parser.tools import DotDict

# General Settings
NUM_NOISY_PER_CLEAN = 15
LOAD_MODEL = False
RESUME_FROM_CKPT = False
if RESUME_FROM_CKPT:
    # CONFIG_NAME = "2025-03-21_09-20-38"
    # CONFIG_NAME = "2025-03-25_12-35-50" # Now using many datas
    # Lower learning rate
    # Even lower learning rate. 
    # Lower noise and lower learning rate
    # Higher learning rate more batches
    # CONFIG_NAME = "2025-03-25_15-19-35"
    # Now I fixed the nbpms (was using 1 BPM then using beta functions to create all the bpms)
    # CONFIG_NAME = "2025-03-27_11-04-33"
    # Now I fixed the noises being linked to the clean data, it should all be randomly distributed. 
    # CONFIG_NAME = "2025-03-27_16-43-06"
    # CONFIG_NAME = "2025-03-27_16-54-52" # Smaller size of module
    # Even smaller base channels as the autoencoder is still learning the different tunes. 
    CONFIG_NAME = "2025-03-28_10-10-08"
else:
    CONFIG_NAME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Data Settings
NBPMS = 563
TOTAL_TURNS = 2000  # Total turns in the simulated data file
NTURNS = 1000  # Training window length

# Various Clean Data Settings
TUNE_LIST = [
    # [0.07, 0.06],
    [0.19, 0.18],
    [0.26, 0.27],
    [0.27, 0.28],
    [0.28, 0.29],
    [0.29, 0.31],
    [0.32, 0.31],
    [0.31, 0.32],
    [0.43, 0.49],
    [0.74, 0.73],
    [0.72, 0.69],
    [0.71, 0.68],
    [0.69, 0.68],
]
COUPLING = [False, 1e-4, 1e-3]
KICK_AMPS = [5e-5, 1e-4, 5e-4]

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

BATCH_SIZE = 25
ACCUMULATE_BATCHES = 4
TRAIN_RATIO = 0.8

MODEL_SAVE_PATH = "conv_autoencoder.pth"
NLOGSTEPS = 1

# NUM_PLANES = 2
NUM_CHANNELS = 1
PRECISION = "16-mixed"

NUM_EPOCHS = 500
BOTTLENECK_SIZE = 4
BASE_CHANNELS = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

ALPHA = 0.5

DENOISED_INDEX = "denoised"
HARPY_CLEAN_INDEX = "harpy_cleaned"
SAMPLE_INDEX = "noisy"
NONOISE_INDEX = "zero_noise"

# NOISE_FACTORS = [1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 5e-5]
# NOISE_FACTORS = [1e-3, 5e-4, 1e-4, 5e-5]
NOISE_FACTORS = [1e-3]

# MODEL_TYPE = "leaky"
MODEL_TYPE = "unet_fixed"
MODEL_DEPTH = 4
RESIDUALS = True

LOSS_TYPE = "comb_ssp"
# LOSS_TYPE = "mse"
SCHEDULER = False
MIN_LR = 1e-5

INIT = "xavier"
DATA_SCALING = "minmax"
MISSING_PROB = 2e-2

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
