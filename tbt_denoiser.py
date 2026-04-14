from pathlib import Path

import numpy as np
import tfs
import torch
from turn_by_turn.lhc import read_tbt

from config import DENOISED_INDEX, NBPMS, NTURNS, NONOISE_INDEX, RESIDUALS, get_model_dir
from dataloader import get_twiss_path, load_clean_data, parse_tbt_path_metadata, write_data
from pl_module import get_model
from project_paths import get_tbt_path


def _get_matching_clean_path(noisy_tbt_path: Path) -> Path:
    metadata = parse_tbt_path_metadata(noisy_tbt_path)
    clean_tbt_path = get_tbt_path(
        beam=metadata["beam"],
        nturns=metadata["nturns"],
        coupling_knob=metadata["coupling_knob"],
        tunes=metadata["tunes"],
        kick_amp=metadata["kick_amp"],
        index=NONOISE_INDEX,
    )
    if not clean_tbt_path.exists():
        raise FileNotFoundError(
            f"Could not find matching clean TBT file for {noisy_tbt_path} at {clean_tbt_path}."
        )
    return clean_tbt_path


def _resolve_model_state_dict(checkpoint_data: dict) -> dict:
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint did not contain a valid state dict.")

    if any(key.startswith("model.") for key in state_dict):
        return {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
    return state_dict


def load_denoiser_model(weights_path: str | Path) -> torch.nn.Module:
    model = get_model()
    checkpoint_data = torch.load(weights_path, map_location=torch.device("cpu"))
    state_dict = _resolve_model_state_dict(checkpoint_data)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def denoise_tbt(autoencoder_path: str, noisy_tbt_path: str) -> Path:
    """
    Denoise a turn-by-turn file using a saved autoencoder.

    Args:
        autoencoder_path (str): Path to the saved autoencoder state_dict.
        noisy_tbt_path (str): Path to the turn-by-turn file to be cleaned.

    Returns:
        str: The file path of the cleaned turn-by-turn file.
    """
    noisy_tbt_path = Path(noisy_tbt_path)
    metadata = parse_tbt_path_metadata(noisy_tbt_path)

    # Load beta functions from the corresponding twiss file.
    model_dat = tfs.read(
        get_twiss_path(
            get_model_dir(
                beam=metadata["beam"],
                coupling_knob=metadata["coupling_knob"],
                tunes=metadata["tunes"],
            )
        )
    )
    sqrt_betax = np.sqrt(model_dat["BETX"].values)
    sqrt_betay = np.sqrt(model_dat["BETY"].values)

    # --- Load clean data to compute normalization parameters ---
    clean_tbt_path = _get_matching_clean_path(noisy_tbt_path)
    clean_tensor_x, clean_tensor_y, _, _ = load_clean_data(clean_tbt_path)
    min_x = clean_tensor_x.min().item()
    max_x = clean_tensor_x.max().item()
    min_y = clean_tensor_y.min().item()
    max_y = clean_tensor_y.max().item()

    # --- Load the noisy TBT data ---
    tbt_data = read_tbt(noisy_tbt_path)
    # Get the X and Y data and β-scale them (like in load_clean_data)
    x_data = tbt_data.matrices[0].X.to_numpy() / sqrt_betax[:, None]
    y_data = tbt_data.matrices[0].Y.to_numpy() / sqrt_betay[:, None]

    expected_shape = (NBPMS, metadata["nturns"])
    assert x_data.shape == y_data.shape == expected_shape, "Data shape mismatch"

    # --- Normalize noisy data using minmax scaling from the clean file ---
    norm_x = 2 * (x_data - min_x) / (max_x - min_x) - 1
    norm_y = 2 * (y_data - min_y) / (max_y - min_y) - 1

    # Convert from shape (NBPMS, NTURNS) to (1, 1, NBPMS, NTURNS)
    noisy_norm_x = torch.tensor(norm_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    noisy_norm_y = torch.tensor(norm_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # --- Load the autoencoder model ---
    model = load_denoiser_model(autoencoder_path)

    # --- Run the autoencoder ---
    with torch.no_grad():
        if RESIDUALS:
            recon_x = noisy_norm_x - model(noisy_norm_x)
            recon_y = noisy_norm_y - model(noisy_norm_y)
        else:
            recon_x = model(noisy_norm_x)
            recon_y = model(noisy_norm_y)
    
    # -- Remove the batch dimension --
    recon_x = recon_x.squeeze(0).squeeze(0).detach().cpu().numpy()
    recon_y = recon_y.squeeze(0).squeeze(0).detach().cpu().numpy()


    # --- Inverse minmax scaling ---
    recon_x = (recon_x + 1) / 2 * (max_x - min_x) + min_x
    recon_y = (recon_y + 1) / 2 * (max_y - min_y) + min_y

    # --- Write the cleaned data ---
    cleaned_file_path, _ = write_data(
        recon_x,
        recon_y,
        noise_index=DENOISED_INDEX,
        reference_tbt_path=noisy_tbt_path,
    )
    return cleaned_file_path
