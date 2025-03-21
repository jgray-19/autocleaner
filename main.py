import os
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

# import hiddenlayer as hl
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger

# Import configurations and utilities
from config import (
    CONFIG_NAME,
    # DENOISED_INDEX,
    LEARNING_RATE,
    LOAD_MODEL,
    LOSS_TYPE,
    MODEL_SAVE_PATH,
    MODEL_TYPE,
    NUM_EPOCHS,
    PLOT_DIR,
    # SAMPLE_INDEX,
    WEIGHT_DECAY,
    print_config,
    save_experiment_config,
    NLOGSTEPS,
    # NTURNS,
    # NBPMS,
    # NUM_PLANES,
    RESUME_FROM_CKPT,
    RESIDUALS,
)
from dataloader import build_sample_dict, load_data#, write_data
from ml_models.conv_2d import (
    Conv2DAutoencoder,
    Conv2DAutoencoderLeaky,
    SineConv2DAutoencoder,
    Conv2DAutoencoderLeakyNoFC,
    Conv2DAutoencoderLeakyFourier,
    DeepConvAutoencoder,
)
from ml_models.fno import FNO2d
from ml_models.unet import (
    UNetAutoencoder,
    UNetAutoencoderFixedDepth,
    UNetAutoencoderFixedDepthCheckpoint,
)
from pl_module import LitAutoencoder, find_newest_file
from visualisation import (
    plot_data_distribution,
    plot_denoised_data,
    plot_noisy_data,
    # plot_model_architecture,
)

assert __name__ == "__main__", "This script is not meant to be imported."
print_config()
free, available = torch.cuda.mem_get_info()
print("Current GPU use:", (available - free) / 1e9, "GB")

# Set up reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load data
print("Loading data...")
b4_load = time.time()
train_loader, val_loader, dataset = load_data()
print(f"Data loaded. Took {time.time() - b4_load:.2f} seconds.")

# Visualise data distribution from one batch
# batch = next(iter(train_loader))
# plot_data_distribution(batch["noisy"], "Noisy Data Distribution")
# plot_data_distribution(batch["clean"], "Clean Data Distribution")
# plot_noisy_data(batch["noisy"][0, 0, :, :], batch["clean"][0, 0, :, :], 111)

# Initialize or Load Model
if MODEL_TYPE == "sine":
    model = SineConv2DAutoencoder()
elif MODEL_TYPE == "conv":
    model = Conv2DAutoencoder()
elif MODEL_TYPE == "leaky":
    model = Conv2DAutoencoderLeaky()
elif MODEL_TYPE == "nofc":
    model = Conv2DAutoencoderLeakyNoFC()
elif MODEL_TYPE == "fourier":
    model = Conv2DAutoencoderLeakyFourier()
elif MODEL_TYPE == "deep":
    model = DeepConvAutoencoder()
elif MODEL_TYPE == "unet":
    model = UNetAutoencoder()
elif MODEL_TYPE == "unet_fixed":
    model = UNetAutoencoderFixedDepth()
elif MODEL_TYPE == "unet_fixed_checkpoint":
    model = UNetAutoencoderFixedDepthCheckpoint()
elif MODEL_TYPE == "fno":
    model = FNO2d()
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

if LOAD_MODEL and os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print("Loaded pre-trained model.")
else:
    print("Training new model...")
    num_cpu = min(os.cpu_count(), 32)
    torch.set_num_threads(num_cpu)
    print(f"Using {num_cpu} CPUs for training.")

    lit_model = LitAutoencoder(
        model,
        loss_type=LOSS_TYPE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    root_dir = Path("/home/jovyan/")
    log_dir = root_dir / "tensor-logs/"
    logger = TensorBoardLogger(root_dir, name=log_dir.name, version=CONFIG_NAME)
    if not RESUME_FROM_CKPT:
        save_experiment_config(log_dir)

    b4_train = time.time()
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        log_every_n_steps=NLOGSTEPS,
        default_root_dir=root_dir,  # Set the logging path
        logger=logger,
        # max_time={"hours": 7.5},
        # enable_checkpointing=False,  # Disables automatic checkpoint saving
    )
    if RESUME_FROM_CKPT:
        ckpt_fldr = log_dir / CONFIG_NAME / "checkpoints"
        ckpt_file = find_newest_file(ckpt_fldr)
        ckpt_path = ckpt_fldr / ckpt_file
    else:
        ckpt_path = None
    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=ckpt_path)
    print(f"Training took {time.time() - b4_train:.2f} seconds.")

    # Save the model
    b4_save = time.time()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved. Took {time.time() - b4_save:.2f} seconds.")

# Denoise validation data
print("Denoising validation data...")
b4_denoise = time.time()

batch = next(iter(val_loader))
noisy_batch_x = batch["noisy_x"]  # shape: (B, 1, NBPMS, NTURNS)
noisy_batch_y = batch["noisy_y"]  # shape: (B, 1, NBPMS, NTURNS)

with torch.no_grad():
    if RESIDUALS:
        recon_x = noisy_batch_x - model(noisy_batch_x)
        recon_y = noisy_batch_y - model(noisy_batch_y)
    else:
        recon_x = model(noisy_batch_x)
        recon_y = model(noisy_batch_y)

assert recon_x.size(0) == noisy_batch_x.size(0) == noisy_batch_y.size(0) == recon_y.size(0)
sample = {
    "noisy_x": noisy_batch_x[0, 0, ...].numpy(),
    "noisy_y": noisy_batch_y[0, 0, ...].numpy(),
    "recon_x": recon_x[0, 0, ...].numpy(),
    "recon_y": recon_y[0, 0, ...].numpy(),
    "clean_x": batch["clean_x"][0, 0, ...].numpy(),
    "clean_y": batch["clean_y"][0, 0, ...].numpy(),
}

sample_dict = build_sample_dict(sample, dataset)

print(f"Denoising took {time.time() - b4_denoise:.2f} seconds.")

device_index = 111  # Select a BPM to plot the FFT spectrum
print(f"Denoised Data for Device {device_index}")

# Save the experiment configuration
save_experiment_config(PLOT_DIR)

# Plot the denoised data
plot_denoised_data(sample_dict, device_index)

plt.show()
