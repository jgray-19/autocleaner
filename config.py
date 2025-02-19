# config.py

from rdt_functions import get_model_dir
from generic_parser.tools import DotDict

# General Settings
BEAM = 1
NUM_FILES = 100
LOAD_MODEL = False

# Data Settings
NBPMS = 563
NTURNS = 1000
BATCH_SIZE = 3
TRAIN_RATIO = 0.9
MODEL_SAVE_PATH = "conv_autoencoder.pth"
MODEL_DIR = get_model_dir(beam=BEAM)

NUM_PLANES = 2
NUM_CHANNELS = NBPMS * NUM_PLANES  # Now processing both X and Y planes

# Optimisation Settings
NUM_EPOCHS = 2000
BOTTLENECK_SIZE = 64
BASE_CHANNELS = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

BPM_DIFF_WEIGHT = 1e-1
TURN_DIFF_WEIGHT = 1e-1

ALPHA = 0.9

DENOISED_INDEX = -2
SAMPLE_INDEX = int(((1-TRAIN_RATIO) * NUM_FILES) // 2)

NOISE_FACTOR=1e-9

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