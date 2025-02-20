import os

import numpy as np
import torch

# Import configurations and utilities
from config import (
    BOTTLENECK_SIZE,
    LOAD_MODEL,
    MODEL_SAVE_PATH,
    NUM_CHANNELS,
    print_config,
    DENOISED_INDEX,
    SAMPLE_INDEX,
)
from dataloader import load_data, write_data
# from models import SimpleSkipAutoencoder
from models import ImprovedConvAutoencoder
from training import train_model
from visualisation import plot_denoised_data

assert __name__ == "__main__", "This script is not meant to be imported."
print_config()

# Set up reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load data
print("Loading data...")
train_loader, val_loader, dataset = load_data()

# Initialize or Load Model
model = ImprovedConvAutoencoder(input_channels=NUM_CHANNELS, bottleneck_size=BOTTLENECK_SIZE)

if LOAD_MODEL and os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print("Loaded pre-trained model.")
else:
    print("Training new model...")
    num_cpu = min(os.cpu_count(), 32)
    torch.set_num_threads(num_cpu)
    print(f"Using {num_cpu} CPUs for training.")
    model = train_model(model, train_loader, val_loader)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved.")

# Denoise validation data
print("Denoising validation data...")

denoised_batches = []
noisy_batches = []
clean_batches = []
for noisy, clean in val_loader:
    with torch.no_grad():
        denoised_batch = model(noisy)
    denoised_batches.append(denoised_batch.cpu())
    noisy_batches.append(noisy.cpu())
    clean_batches.append(clean.cpu())

denoised_val_data = torch.cat(denoised_batches, dim=0)
noisy_val_data = torch.cat(noisy_batches, dim=0)
clean_val_data = torch.cat(clean_batches, dim=0)
print("Denoising complete.")

# --- Denormalize the outputs for analysis ---
denoised_val_data_denorm = dataset.denormalise(denoised_val_data)
noisy_val_data_denorm = dataset.denormalise(noisy_val_data)
clean_val_data_denorm = dataset.denormalise(clean_val_data)

# For FFT and visualization, select a sample from the denormalized denoised data.
select_denoised_sample = denoised_val_data_denorm[SAMPLE_INDEX]  # Shape: (2*NBPMS, NTURNS)
selected_noisy_sample = noisy_val_data_denorm[SAMPLE_INDEX]  # For comparison if needed
selected_clean_sample = clean_val_data_denorm[SAMPLE_INDEX]  # For comparison if needed

if torch.max(torch.abs(selected_clean_sample - dataset.clean_data) < 1e-6):
    print("Selected clean sample matches the original clean data.")
else:
    print("Selected clean sample does not match the original clean data.")

device_index = 111  # Select a BPM to plot the FFT spectrum
print(f"Denoised Data for Device {device_index} at sample index {SAMPLE_INDEX}")
plot_denoised_data(
    select_denoised_sample, selected_noisy_sample, selected_clean_sample, device_index
)

# Write out the denoised data
denoised_path, denoised_sdds = write_data(select_denoised_sample, DENOISED_INDEX)
noisy_path, noisy_sdds = write_data(selected_noisy_sample, SAMPLE_INDEX)
print(f"Denoised data written to {denoised_path}")
print(f"Noisy data written to {noisy_path}")
