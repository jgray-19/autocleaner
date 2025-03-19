from pathlib import Path

import numpy as np
import tfs
import torch
from turn_by_turn.lhc import read_tbt

from config import BEAM, DENOISED_INDEX, NBPMS, NTURNS, get_model_dir
from dataloader import load_clean_data, write_data
from ml_models.unet import UNetAutoencoderFixedDepthCheckpoint


def denoise_tbt(autoencoder_path: str, noisy_tbt_path: str) -> Path:
    """
    Denoise a turn-by-turn file using a saved autoencoder.

    Args:
        autoencoder_path (str): Path to the saved autoencoder state_dict.
        noisy_tbt_path (str): Path to the turn-by-turn file to be cleaned.

    Returns:
        str: The file path of the cleaned turn-by-turn file.
    """
    # Load beta functions from the twiss file (same as in load_clean_data)
    model_dat = tfs.read(get_model_dir(BEAM) / "twiss.dat")
    sqrt_betax = np.sqrt(model_dat["BETX"].values)
    sqrt_betay = np.sqrt(model_dat["BETY"].values)

    # --- Load clean data to compute normalization parameters ---
    # load_clean_data returns a tensor of shape (TOTAL_TURNS, 2*NBPMS) where data are β-scaled.
    clean_tensor_x, clean_tensor_y = load_clean_data()
    min_x = clean_tensor_x.min().item()
    max_x = clean_tensor_x.max().item()
    min_y = clean_tensor_y.min().item()
    max_y = clean_tensor_y.max().item()

    # --- Load the noisy TBT data ---
    tbt_data = read_tbt(noisy_tbt_path)
    # Get the X and Y data and β-scale them (like in load_clean_data)
    x_data = tbt_data.matrices[0].X.to_numpy() / sqrt_betax[:, None]
    y_data = tbt_data.matrices[0].Y.to_numpy() / sqrt_betay[:, None]

    assert x_data.shape == y_data.shape == (NBPMS, NTURNS), "Data shape mismatch"

    # --- Normalize noisy data using minmax scaling from the clean file ---
    norm_x = 2 * (x_data - min_x) / (max_x - min_x) - 1
    norm_y = 2 * (y_data - min_y) / (max_y - min_y) - 1
    # Stack into a tensor of shape (2, NBPMS, NTURNS)
    noisy_norm = torch.tensor(np.stack([norm_x, norm_y], axis=0), dtype=torch.float32)

    # --- Load the autoencoder model ---
    model = UNetAutoencoderFixedDepthCheckpoint()

    state_dict = torch.load(autoencoder_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    # --- Run the autoencoder ---
    with torch.no_grad():
        # If your model uses a residual connection, adjust accordingly.
        denoised_norm = model(noisy_norm)

    # --- Inverse minmax scaling ---
    denoised_norm_np = denoised_norm.detach().cpu().numpy()  # shape (2, NBPMS, NTURNS)
    denoised_x = (denoised_norm_np[0] + 1) / 2 * (max_x - min_x) + min_x
    denoised_y = (denoised_norm_np[1] + 1) / 2 * (max_y - min_y) + min_y
    # Stack back into cleaned tensor (2, NBPMS, NTURNS)
    cleaned = np.stack([denoised_x, denoised_y], axis=0)
    cleaned_tensor = torch.tensor(cleaned, dtype=torch.float32)

    # --- Write the cleaned data ---
    cleaned_file_path, _ = write_data(cleaned_tensor, noise_index=DENOISED_INDEX)
    return cleaned_file_path
