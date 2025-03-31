import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from lhcng.config import PLOT_DIR

# Import your relevant modules
from config import (
    CONFIG_NAME,
    LEARNING_RATE,
    LOSS_TYPE,
    MODEL_SAVE_PATH,
    MODEL_TYPE,  # to choose model architecture
    PRECISION,
    RESIDUALS,
    WEIGHT_DECAY,
    save_experiment_config,
)
from dataloader import (
    BPMSDataset,
    build_sample_dict,
    load_data,
    save_global_norm_params,
)
from pl_module import LitAutoencoder, find_newest_file, get_model
from visualisation import plot_denoised_data


def denoise_validation_sample_from_checkpoint(
    checkpoint_name: str,
    val_loader,
    dataset: BPMSDataset,
    residuals: bool = RESIDUALS,
    device_index: int = 111,
    do_plot: bool = True,
):
    """
    Loads a model checkpoint (on CPU), denoises one batch from the validation loader (on CPU),
    and optionally plots the results.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint (`.ckpt` or `.pth`).
        val_loader (DataLoader): A validation DataLoader with the same structure used in training.
        dataset (BPMSDataset): The dataset object, for inverting normalization (via build_sample_dict).
        model_type (str): String specifying the model architecture (e.g., 'unet_fixed_checkpoint').
        residuals (bool): True if the model output is a residual to subtract from noisy signal.
        device_index (int): BPM index for FFT plots.
        do_plot (bool): If True, will plot the denoised data.

    Returns:
        sample_dict (dict): Dictionary containing “noisy”, “clean”, and “reconstructed” signals
                            (in original scale) for the chosen sample in the batch.
    """

    print(
        f"Building '{MODEL_TYPE}' model and loading checkpoint on CPU from: {checkpoint_name}"
    )
    # 1) Build the model and load checkpoint weights on CPU
    checkpoint_path = (
        Path("/home/jovyan/tensor-logs/") / checkpoint_name / "checkpoints"
    )
    checkpoint_file = find_newest_file(checkpoint_path)
    lit_model = LitAutoencoder.load_from_checkpoint(
        checkpoint_path=checkpoint_path / checkpoint_file,
        model=get_model(),
        loss_type=LOSS_TYPE,
        learning_rate=LEARNING_RATE,
        precision=PRECISION,
        weight_decay=WEIGHT_DECAY,
    )
    lit_model.eval()

    b4_save = time.time()
    torch.save(lit_model.model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved. Took {time.time() - b4_save:.2f} seconds.")

    # Ensure model is on CPU
    lit_model.to(torch.device("cpu"))

    print("Denoising validation data (CPU-only)...")
    b4_denoise = time.time()

    # 2) Fetch one batch from the validation loader
    batch = next(iter(val_loader))
    noisy_batch_x = batch["noisy_x"]  # shape: (B, 1, NBPMS, NTURNS)
    noisy_batch_y = batch["noisy_y"]  # shape: (B, 1, NBPMS, NTURNS)

    # 3) Forward pass (CPU)
    with torch.no_grad():
        if residuals:
            recon_x = noisy_batch_x - lit_model(noisy_batch_x)
            recon_y = noisy_batch_y - lit_model(noisy_batch_y)
        else:
            recon_x = lit_model(noisy_batch_x)
            recon_y = lit_model(noisy_batch_y)

    # Just to confirm shapes
    Bx = recon_x.size(0)
    By = recon_y.size(0)
    Nx = noisy_batch_x.size(0)
    Ny = noisy_batch_y.size(0)
    assert Bx == Nx, "Recon and input batch size mismatch for X"
    assert By == Ny, "Recon and input batch size mismatch for Y"

    # 4) Build a sample_dict for the first item in the batch
    norm_info = {
        "min_x": dataset.global_min_x,
        "max_x": dataset.global_max_x,
        "min_y": dataset.global_min_y,
        "max_y": dataset.global_max_y,
    }
    # Save global normalization parameters
    save_global_norm_params(dataset, filepath="global_norm_params.json")

    # If theres a problem, first check this!
    sample = {
        "noisy_x": noisy_batch_x[0, 0, ...].numpy(),
        "noisy_y": noisy_batch_y[0, 0, ...].numpy(),
        "recon_x": recon_x[0, 0, ...].numpy(),
        "recon_y": recon_y[0, 0, ...].numpy(),
        "clean_x": batch["clean_x"][0, 0, ...].numpy(),
        "clean_y": batch["clean_y"][0, 0, ...].numpy(),
        "norm_info": norm_info,
    }
    sample_dict = build_sample_dict(sample, dataset)

    print(f"Denoising took {time.time() - b4_denoise:.2f} seconds (CPU).")
    print(f"Denoised Data for BPM device {device_index}")

    # 5) (Optional) Save config and plot data
    if do_plot:
        # If you wish to save your current config to JSON:
        save_experiment_config(PLOT_DIR)

        # Plot the denoised data for the chosen BPM index
        plot_denoised_data(sample_dict, device_index)
        plt.show()

    return sample_dict


# Suppose you already have:
#   val_loader (DataLoader),
#   dataset (BPMSDataset),
#   and a path to your checkpoint: my_ckpt_path

print("Loading data...")
b4_load = time.time()
train_loader, val_loader, dataset = load_data()
print(f"Data loaded. Took {time.time() - b4_load:.2f} seconds.")


sample_dict = denoise_validation_sample_from_checkpoint(
    checkpoint_name=CONFIG_NAME,
    # checkpoint_name="2025-03-27_16-43-06",
    # checkpoint_name="2025-03-27_16-54-52",
    val_loader=val_loader,
    dataset=dataset,
    residuals=False,
    device_index=111,
    do_plot=True,
)
