import json
from pathlib import Path

import numpy as np
import pandas as pd
import tfs
import torch
from lhcng.model import get_model_dir
from lhcng.tracking import get_tbt_path
from torch.utils.data import DataLoader, Dataset
from turn_by_turn import TbtData, TransverseData
from turn_by_turn.lhc import read_tbt, write_tbt

from config import (
    BATCH_SIZE,
    CLEAN_PARAM_LIST,
    MISSING_PROB,
    NBPMS,
    NOISE_FACTORS,
    NONOISE_INDEX,
    NTURNS,
    NUM_NOISY_PER_CLEAN,
    TOTAL_TURNS,
    TRAIN_RATIO,
)


def load_clean_data(sdds_data_path, model_dir) -> tuple[torch.Tensor, torch.Tensor]:
    sdds_data = read_tbt(sdds_data_path)

    # Read the twiss file for beta functions.
    # model_dat = tfs.read(get_model_dir(beam, coupling, tunes) / "twiss.dat")

    # sqrt_betax = np.sqrt(model_dat["BETX"].values)
    # sqrt_betay = np.sqrt(model_dat["BETY"].values)
    # For now, let's not normalise the data.
    sqrt_betax = sqrt_betay = np.ones(NBPMS)

    x_data = sdds_data.matrices[0].X.to_numpy()
    y_data = sdds_data.matrices[0].Y.to_numpy()

    assert x_data.shape[0] == NBPMS, (
        f"Missing some BPMs, x data has shape: {x_data.shape}"
    )
    assert y_data.shape[0] == NBPMS, (
        f"Missing some BPMs, y data has shape: {y_data.shape}"
    )

    return (
        torch.tensor(x_data / sqrt_betax[:, None], dtype=torch.float32),
        torch.tensor(y_data / sqrt_betay[:, None], dtype=torch.float32),
        sqrt_betax,
        sqrt_betay,
    )


def write_data(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    model_dir: Path,
    tbt_path: Path,
) -> tuple[Path, TbtData]:
    model_dat = tfs.read(model_dir / "twiss.dat", index="NAME")
    # sqrt_betax = np.sqrt(model_dat["BETX"].values)
    # sqrt_betay = np.sqrt(model_dat["BETY"].values)
    # For now, let's not normalise the data.
    sqrt_betax = sqrt_betay = np.ones(NBPMS)

    x_bpm_names = model_dat.index.to_list()
    y_bpm_names = model_dat.index.to_list()

    assert x_data.shape[0] == NBPMS, "Missing BPMs"
    assert y_data.shape[0] == NBPMS, "Missing BPMs"

    print("Writing datashape with:", x_data.shape, y_data.shape)

    x_data = x_data * sqrt_betax[:, None]
    y_data = y_data * sqrt_betay[:, None]

    matrices = [
        TransverseData(
            X=pd.DataFrame(index=x_bpm_names, data=x_data, dtype=float),
            Y=pd.DataFrame(index=y_bpm_names, data=y_data, dtype=float),
        )
    ]
    out_data = TbtData(matrices=matrices, nturns=NTURNS)
    write_tbt(tbt_path, out_data)
    return tbt_path, out_data

def cyclic_slice_bpm_turns(raw_tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor of shape (NBPMS, TOTAL_TURNS), this function performs a cyclic
    permutation of the data to create a window of NTURNS turns, starting from turn 0.
    The output tensor will have shape (NBPMS, NTURNS).
    """
    # Flatten in column-major order.
    # Transpose so that rows are turns: shape (TOTAL_TURNS, NBPMS).
    transposed = raw_tensor.t()
    # Flattening now produces a vector where each turn's BPMs are contiguous.
    flat = transposed.reshape(-1)

    # Step 3: Choose one random offset (covering both BPM and turn offsets) and roll.
    offset = np.random.randint(0, (TOTAL_TURNS - NTURNS) * NBPMS-1)
    flat_shifted = torch.roll(flat, shifts=-offset)

    # Step 4: Reshape back and transpose so that the output is (NBPMS, window_length).
    # The flat vector is reshaped into (window_length, NBPMS) then transposed.
    shifted = flat_shifted.reshape(TOTAL_TURNS, -1).t()
    
    # We only want the first NTURNS turns.
    shifted = shifted[:, :NTURNS]
    return shifted



class BPMSDataset(Dataset):
    """
    Returns samples where each sample comes from a clean file determined by a specific
    combination of (beam, tunes, coupling) and a random offset. A different dataset index
    can yield a different clean file (and thus different clean data) rather than only a different noisy slice.
    """

    def __init__(
        self,
        clean_param_list,
        num_variations_per_clean,
        noise_factors=NOISE_FACTORS,
    ):
        super().__init__()
        self.clean_param_list = clean_param_list
        self.num_files_per_clean = num_variations_per_clean
        self.noise_factors = noise_factors

        # Load all clean files once into big lists so we never do it in __getitem__
        all_clean_x = []
        all_clean_y = []
        self.global_min_x = torch.inf
        self.global_max_x = 0
        self.global_min_y = torch.inf
        self.global_max_y = 0

        for params in self.clean_param_list:
            sdds_data_path = get_tbt_path(
                params["beam"],
                TOTAL_TURNS,
                params["coupling"],
                params["tunes"],
                params["kick_amp"],
                NONOISE_INDEX,
            )
            model_dir = get_model_dir(
                params["beam"], params["coupling"], params["tunes"]
            )

            # Load file once
            clean_x, clean_y, _, _ = load_clean_data(sdds_data_path, model_dir)
            # Each is shape (NBPMS, TOTAL_TURNS)
            all_clean_x.append(clean_x)
            all_clean_y.append(clean_y)

            self.global_min_x = min(self.global_min_x, clean_x.min())
            self.global_max_x = max(self.global_max_x, clean_x.max())
            self.global_min_y = min(self.global_min_y, clean_y.min())
            self.global_max_y = max(self.global_max_y, clean_y.max())

        # Now store each file's raw data (still in memory) so we can quickly sample from it
        # We'll keep them separate from the big concatenated arrays for clarity.
        self.cached_clean_x = all_clean_x  # list of Tensors
        self.cached_clean_y = all_clean_y  # list of Tensors

        # Next, define how many total samples we want
        self.num_clean = len(self.clean_param_list)
        self.total_files = self.num_clean * num_variations_per_clean

        # Precompute random indices, offsets, noise
        self.clean_indices = np.random.randint(0, self.num_clean, size=self.total_files)
        self.noise_factors = np.random.choice(
            noise_factors, size=self.total_files, replace=True
        )

        # Initialize noise multiplier for dynamic noise scaling
        self.current_noise_multiplier = 1.0

    def update_noise_multiplier(self, multiplier):
        self.current_noise_multiplier = multiplier

    def __len__(self):
        return self.total_files

    def __getitem__(self, idx):
        # Identify which file we want
        file_idx = self.clean_indices[idx]
        noise_factor = self.noise_factors[idx] * self.current_noise_multiplier

        # Pull out the raw (un-normalized) data from memory
        raw_x = self.cached_clean_x[file_idx]  # shape (NBPMS, TOTAL_TURNS)
        raw_y = self.cached_clean_y[file_idx]

        # Extract a window of NTURNS starting at turn 0 and apply BPM cyclic permutation.
        window_x = cyclic_slice_bpm_turns(raw_x)
        window_y = cyclic_slice_bpm_turns(raw_y)

        # Normalize using global min/max.
        norm_slice_x = 2.0 * (window_x - self.global_min_x) / (self.global_max_x - self.global_min_x) - 1.0
        norm_slice_y = 2.0 * (window_y - self.global_min_y) / (self.global_max_y - self.global_min_y) - 1.0

        # Add noise in normalized domain
        noise_x = torch.randn_like(norm_slice_x) * noise_factor
        noise_y = torch.randn_like(norm_slice_y) * noise_factor
        noisy_slice_x = norm_slice_x + noise_x
        noisy_slice_y = norm_slice_y + noise_y

        # Missing BPM simulation
        mask = torch.rand(NBPMS) < MISSING_PROB
        noisy_slice_x[mask, :] = 0
        noisy_slice_y[mask, :] = 0

        # Add the channel dimension
        # The final shape of the tensors is (1, NBPMS, NTURNS)
        return {
            "clean_x": norm_slice_x.unsqueeze(0),
            "clean_y": norm_slice_y.unsqueeze(0),
            "noisy_x": noisy_slice_x.unsqueeze(0),
            "noisy_y": noisy_slice_y.unsqueeze(0),
        }

    def denormalise(
        self, norm_data: np.ndarray, plane: str, norm_info: dict
    ) -> np.ndarray:
        """
        Inverse transforms a normalised output for a specific plane ('x' or 'y')
        back to the original scale using the provided normalisation parameters.
        """
        if plane not in ["x", "y"]:
            raise ValueError("Plane must be 'x' or 'y'.")

        if plane == "x":
            min_val = norm_info["min_x"].item()
            max_val = norm_info["max_x"].item()
            # betas = norm_info["sqrt_betax"]
        else:
            min_val = norm_info["min_y"].item()
            max_val = norm_info["max_y"].item()
            # betas = norm_info["sqrt_betay"]

        # Inverse min–max scaling.
        orig_data = (norm_data + 1) / 2 * (max_val - min_val) + min_val
        return orig_data  # * betas[:, None]


def load_data() -> tuple[DataLoader, DataLoader, BPMSDataset]:
    dataset = BPMSDataset(
        clean_param_list=CLEAN_PARAM_LIST, num_variations_per_clean=NUM_NOISY_PER_CLEAN
    )
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    return train_loader, val_loader, dataset


def build_sample_dict(sample: dict[str, np.ndarray], dataset: BPMSDataset) -> dict:
    """
    Given a list of batches, returns a dictionary with the X and Y samples,
    both in noisy and clean versions after inversion, for each plane.
    """
    norm_info = sample["norm_info"]
    sample_dict = {}
    for plane in ["x", "y"]:
        sample_dict[f"noisy_{plane}"] = dataset.denormalise(
            sample[f"noisy_{plane}"], plane=plane, norm_info=norm_info
        )
        sample_dict[f"clean_{plane}"] = dataset.denormalise(
            sample[f"clean_{plane}"], plane=plane, norm_info=norm_info
        )
        sample_dict[f"recon_{plane}"] = dataset.denormalise(
            sample[f"recon_{plane}"], plane=plane, norm_info=norm_info
        )
    return sample_dict


def save_global_norm_params(dataset, filepath="global_norm_params.json"):
    """
    Store global min/max for x and y as JSON so we can reload in inference.
    """
    # They are shape (NBPMS,1), so convert to lists
    norm_params = {
        "min_x": dataset.global_min_x.squeeze(1).tolist(),
        "max_x": dataset.global_max_x.squeeze(1).tolist(),
        "min_y": dataset.global_min_y.squeeze(1).tolist(),
        "max_y": dataset.global_max_y.squeeze(1).tolist(),
    }
    with open(filepath, "w") as f:
        json.dump(norm_params, f, indent=2)
