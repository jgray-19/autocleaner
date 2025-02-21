import os

from matplotlib import pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

# Import configurations and utilities
from config import (
    BOTTLENECK_SIZE,
    DENOISED_INDEX,
    LEARNING_RATE,
    LOAD_MODEL,
    MODEL_SAVE_PATH,
    NUM_CHANNELS,
    NUM_EPOCHS,
    NTURNS,
    WEIGHT_DECAY,
    print_config,
)
from dataloader import build_sample_dict, load_data, write_data

# from models import SimpleSkipAutoencoder
from models import ConvFC_Autoencoder
from pl_module import LitAutoencoder
from visualisation import plot_data_distribution, plot_denoised_data

assert __name__ == "__main__", "This script is not meant to be imported."
print_config()

# Set up reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load data
print("Loading data...")
train_loader, val_loader, dataset = load_data()

# Visualise data distribution from one batch
batch = next(iter(train_loader))
plot_data_distribution(batch["noisy"], "Noisy Data Distribution")
plot_data_distribution(batch["clean"], "Clean Data Distribution")

# Initialize or Load Model
model = ConvFC_Autoencoder(
    input_channels=NUM_CHANNELS, 
    input_length=NTURNS,
    bottleneck_dim=BOTTLENECK_SIZE
)

if LOAD_MODEL and os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print("Loaded pre-trained model.")
else:
    print("Training new model...")
    num_cpu = min(os.cpu_count(), 32)
    torch.set_num_threads(num_cpu)
    print(f"Using {num_cpu} CPUs for training.")
    lit_model = LitAutoencoder(
        model, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    log_dir="/home/jovyan/tensor-logs/"
    logger = TensorBoardLogger(log_dir, name="tensorboard_logs")
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        log_every_n_steps=5,
        default_root_dir=log_dir,  # Set the logging path
        logger=logger,
    )
    trainer.fit(lit_model, train_loader, val_loader)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved.")

# Denoise validation data
print("Denoising validation data...")

sample_list = []
for batch in val_loader:
    # collected_batches.append(batch)
    noisy_batch = batch["noisy"]
    # Process the batch through the model
    with torch.no_grad():
        denoised_batch = model(noisy_batch)
    # Add the model output to the batch dictionary.
    for i in range(denoised_batch.size(0)):
        sample = {
            "noisy": noisy_batch[i],
            "clean": batch["clean"][i],
            "denoised": denoised_batch[i],
            "plane": batch["plane"][i],
        }
        sample_list.append(sample)

# Reassemble one file (both planes) from the collected batches.
# (Here we assume that among the batches we have at least one X and one Y sample.)
sample_dict = build_sample_dict(sample_list, dataset)
selected_noisy_sample = torch.cat([sample_dict["x"], sample_dict["y"]], dim=0)
selected_clean_sample = torch.cat(
    [sample_dict["x_clean"], sample_dict["y_clean"]], dim=0
)
select_denoised_sample = torch.cat(
    [sample_dict["x_denoised"], sample_dict["y_denoised"]], dim=0
)

# Check if the selected clean sample matches the original clean data
if torch.max(torch.abs(selected_clean_sample - dataset.raw_data) < 1e-8):
    print("Selected clean sample matches the original clean data.")
else:
    print("Selected clean sample does not match the original clean data.")

device_index = 111  # Select a BPM to plot the FFT spectrum
print(f"Denoised Data for Device {device_index}")
plot_denoised_data(
    select_denoised_sample, selected_noisy_sample, selected_clean_sample, device_index
)

# Write out the denoised data
denoised_path, denoised_sdds = write_data(select_denoised_sample, DENOISED_INDEX)
noisy_path, noisy_sdds = write_data(selected_noisy_sample)
print(f"Denoised data written to {denoised_path}")
print(f"Noisy data written to {noisy_path}")

plt.show()
