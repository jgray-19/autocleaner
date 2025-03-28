from pathlib import Path

import torch
from turn_by_turn.lhc import read_tbt

from config import NBPMS
from dataloader import load_clean_data, write_data
from pl_module import get_model


def denoise_tbt(
    autoencoder_path: Path,
    model_dir: Path,
    clean_tbt_path: Path,
    noisy_tbt_path: Path,
    denoised_tbt_path: Path,
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
    min_x = clean_tensor_x.min().item()
    max_x = clean_tensor_x.max().item()
    min_y = clean_tensor_y.min().item()
    max_y = clean_tensor_y.max().item()

    # --- Load the noisy TBT data ---
    tbt_data = read_tbt(noisy_tbt_path)
    # Get the X and Y data and β-scale them (like in load_clean_data)
    x_data = tbt_data.matrices[0].X.to_numpy() 
    y_data = tbt_data.matrices[0].Y.to_numpy() 

    assert x_data.shape[0] == NBPMS, "Missing some BPMs"
    assert y_data.shape[0] == NBPMS, "Missing some BPMs"

    x_data /= sqrt_betax[:, None] 
    y_data /= sqrt_betay[:, None]

    # --- Normalize noisy data using minmax scaling from the clean file ---
    norm_x = 2 * (x_data - min_x) / (max_x - min_x) - 1
    norm_y = 2 * (y_data - min_y) / (max_y - min_y) - 1

    # Convert from shape (NBPMS, NTURNS) to (1, 1, NBPMS, NTURNS)
    noisy_norm_x = torch.tensor(norm_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    noisy_norm_y = torch.tensor(norm_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # --- Load the autoencoder model ---
    model = get_model()

    state_dict = torch.load(autoencoder_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    # --- Run the autoencoder ---
    with torch.no_grad():
        recon_x = model(noisy_norm_x)
        recon_y = model(noisy_norm_y)

    # -- Remove the batch dimension --
    recon_x = recon_x.squeeze(0).squeeze(0).detach().cpu().numpy()
    recon_y = recon_y.squeeze(0).squeeze(0).detach().cpu().numpy()

    # --- Inverse minmax scaling ---
    recon_x = (recon_x + 1) / 2 * (max_x - min_x) + min_x
    recon_y = (recon_y + 1) / 2 * (max_y - min_y) + min_y

    # --- Write the cleaned data ---
    cleaned_file_path, _ = write_data(recon_x, recon_y, model_dir, denoised_tbt_path)
    assert cleaned_file_path == denoised_tbt_path, (
        "Written path different to path asked to be written"
    )
    return cleaned_file_path
