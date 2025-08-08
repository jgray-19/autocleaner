from pathlib import Path

import torch
from turn_by_turn.lhc import read_tbt

from config import RESIDUALS
from dataloader import load_clean_data, write_data
from pl_module import get_model

def denoise_tbt(
    autoencoder_path: Path,
    model_dir: Path,
    clean_tbt_path: Path,
    noisy_tbt_path: Path,
    denoised_tbt_path: Path,
    norm_params_path: Path = "global_norm_params.json",
) -> Path:
    """
    Denoise a turn-by-turn file using a saved autoencoder.

    Args:
        autoencoder_path (Path|str): Path to the saved autoencoder state_dict.
        noisy_tbt_path (Path|str): Path to the turn-by-turn file to be cleaned.

    Returns:
        Path: The file path of the cleaned turn-by-turn file.
    """
    # --- Load clean data to compute normalization parameters ---
    # load_clean_data returns a tensor of shape (TOTAL_TURNS, 2*NBPMS) where data are β-scaled.
    clean_tensor_x, clean_tensor_y, sqrt_betax, sqrt_betay = load_clean_data(
        clean_tbt_path, model_dir
    )

    # --- Load the noisy TBT data ---
    tbt_data = read_tbt(noisy_tbt_path)
    # Get the X and Y data and β-scale them (like in load_clean_data)
    x_data = tbt_data.matrices[0].X.to_numpy()
    y_data = tbt_data.matrices[0].Y.to_numpy()

    # --- Normalize noisy data using global stats (z-score normalization) ---
    x_mean = x_data.mean()
    y_mean = y_data.mean()
    x_std = x_data.std()
    y_std = y_data.std()
    norm_x = (x_data - x_mean) / x_std
    norm_y = (y_data - y_mean) / y_std

    # Convert from shape (NBPMS, NTURNS) to (1, 1, NBPMS, NTURNS)
    noisy_norm_x = torch.tensor(norm_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    noisy_norm_y = torch.tensor(norm_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # --- Load the autoencoder model ---
    model = get_model()

    state_dict = torch.load(autoencoder_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    noisy = torch.cat((noisy_norm_x, noisy_norm_y), dim=0)

    # --- Run the autoencoder ---
    with torch.no_grad():
        recon = model(noisy)
        if RESIDUALS:
            # If using residuals, return the noisy input minus the reconstruction
            recon = noisy - recon
    recon_x, recon_y = torch.chunk(recon, 2, dim=0)

    # -- Remove the batch dimension --
    recon_x = recon_x.squeeze(0).squeeze(0).detach().cpu().numpy()
    recon_y = recon_y.squeeze(0).squeeze(0).detach().cpu().numpy()

    # --- Inverse normalization (z-score denormalization) ---
    recon_x = recon_x * x_std + x_mean
    recon_y = recon_y * y_std + y_mean

    # --- Write the cleaned data ---
    cleaned_file_path, _ = write_data(recon_x, recon_y, noisy_tbt_path, denoised_tbt_path)
    assert cleaned_file_path == denoised_tbt_path, (
        "Written path different to path asked to be written"
    )
    return cleaned_file_path
