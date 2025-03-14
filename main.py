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
    DENOISED_INDEX,
    LEARNING_RATE,
    LOAD_MODEL,
    LOSS_TYPE,
    MODEL_SAVE_PATH,
    MODEL_TYPE,
    NUM_EPOCHS,
    PLOT_DIR,
    SAMPLE_INDEX,
    WEIGHT_DECAY,
    print_config,
    save_experiment_config,
    NLOGSTEPS,
    # NTURNS,
    # NBPMS,
    # NUM_PLANES,
    RESUME_FROM_CKPT,
)
from dataloader import build_sample_dict, load_data, write_data
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
batch = next(iter(train_loader))
plot_data_distribution(batch["noisy"], "Noisy Data Distribution")
plot_data_distribution(batch["clean"], "Clean Data Distribution")
plot_noisy_data(batch["noisy"][0, 0, :, :], batch["clean"][0, 0, :, :], 111)

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
        max_time={"hours": 7.5},
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

sample_list = []
val_indices = val_loader.dataset.indices
for val_idx, batch in enumerate(val_loader):
    noisy_batch = batch["noisy"]

    # Process the batch through the model
    with torch.no_grad():
        denoised_batch = model(noisy_batch)

    # Add the model output to the batch dictionary.
    assert denoised_batch.size(0) == noisy_batch.size(0)  # Just checking
    for batch_idx in range(denoised_batch.size(0)):
        dataset_idx = val_indices[val_idx * val_loader.batch_size + batch_idx]
        full_noisy = dataset.get_full_noisy(dataset_idx)
        sample = {
            "noisy": noisy_batch[batch_idx],
            "noisy_full": full_noisy[batch_idx],
            "clean": batch["clean"][batch_idx],
            "denoised": denoised_batch[batch_idx],
        }
        sample_list.append(sample)
    break

# Reassemble one file (both planes) from the collected batches.
# (Here we assume that among the batches we have at least one X and one Y sample.)
sample_dict = build_sample_dict(sample_list, dataset)
selected_noisy_sample = torch.stack(
    [sample_dict["x_noisy"], sample_dict["y_noisy"]], dim=0
)
selected_noisy_full_sample = torch.stack(
    [sample_dict["x_full"], sample_dict["y_full"]], dim=0
)
selected_clean_sample = torch.stack(
    [sample_dict["x_clean"], sample_dict["y_clean"]], dim=0
)
select_denoised_sample = torch.stack(
    [sample_dict["x_denoised"], sample_dict["y_denoised"]], dim=0
)

print(f"Denoising took {time.time() - b4_denoise:.2f} seconds.")

device_index = 111  # Select a BPM to plot the FFT spectrum
print(f"Denoised Data for Device {device_index}")

# Save the experiment configuration
save_experiment_config(PLOT_DIR)

# Plot the denoised data
plot_denoised_data(
    select_denoised_sample, selected_noisy_sample, selected_clean_sample, device_index
)

# Write out the denoised data
denoised_path, denoised_sdds = write_data(select_denoised_sample, DENOISED_INDEX)
noisy_path, noisy_sdds = write_data(selected_noisy_full_sample, SAMPLE_INDEX)
noisy_path, noisy_sdds = write_data(selected_noisy_sample, SAMPLE_INDEX + "_split")
print(f"Denoised data written to {denoised_path}")
print(f"Noisy data written to {noisy_path}")

plt.show()
