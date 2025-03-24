from pathlib import Path

import numpy as np
import pandas as pd
import tfs
import torch
from lhcng.model import get_model_dir
from lhcng.tracking import get_tbt_path
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, Dataset
from turn_by_turn import TbtData, TransverseData
from turn_by_turn.lhc import read_tbt, write_tbt

from config import (
    BATCH_SIZE,
    BEAM,
    NBPMS,
    NOISE_FACTORS,
    NTURNS,
    NUM_FILES,
    NUM_SAME_OFFSET,
    SEED,
    TOTAL_TURNS,
    TRAIN_RATIO,
    USE_OFFSETS,
)


def load_clean_data() -> tuple[torch.Tensor, torch.Tensor]:
    # Load zero-noise data
    sdds_data_path = get_tbt_path(beam=BEAM, nturns=TOTAL_TURNS, index=-1)
    sdds_data = read_tbt(sdds_data_path)

    # Read twiss file for beta functions
    model_dat = tfs.read(get_model_dir(beam=BEAM) / "twiss.dat")
    sqrt_betax = np.sqrt(model_dat["BETX"].values)  # For X plane
    sqrt_betay = np.sqrt(model_dat["BETY"].values)  # For Y plane

    # Extract data; assume sdds_data.matrices[0] contains both 'X' and 'Y'
    x_data = sdds_data.matrices[0].X.to_numpy() / sqrt_betax[:, None]
    y_data = sdds_data.matrices[0].Y.to_numpy() / sqrt_betay[:, None]

    assert x_data.shape == y_data.shape == (NBPMS, TOTAL_TURNS), "Data shape mismatch"

    # Return x_data and y_data as separate tensors
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(
        y_data, dtype=torch.float32
    )


def write_data(x_data: torch.Tensor, y_data: torch.Tensor, noise_index: int = 2) -> tuple[Path, TbtData]:
    model_dat = tfs.read(get_model_dir(beam=BEAM) / "twiss.dat", index="NAME")
    sqrt_betax = np.sqrt(model_dat["BETX"].values)
    sqrt_betay = np.sqrt(model_dat["BETY"].values)
    x_bpm_names = model_dat.index.to_list()
    y_bpm_names = model_dat.index.to_list()

    assert x_data.shape == (NBPMS, NTURNS), "Data shape mismatch"
    assert y_data.shape == (NBPMS, NTURNS), "Data shape mismatch"
    print("Writing datashape with:", x_data.shape, y_data.shape)

    x_data = x_data * sqrt_betax[:, None]
    y_data = y_data * sqrt_betay[:, None]
    out_path = get_tbt_path(beam=BEAM, nturns=NTURNS, index=noise_index)

    matrices = [
        TransverseData(
            X=pd.DataFrame(index=x_bpm_names, data=x_data, dtype=float),
            Y=pd.DataFrame(index=y_bpm_names, data=y_data, dtype=float),
        )
    ]
    out_data = TbtData(matrices=matrices, nturns=NTURNS)
    write_tbt(out_path, out_data)
    return out_path, out_data


class BPMSDataset(Dataset):
    """
    A compact dataset class that:
      - Assumes min–max normalization.
      - Precomputes the normalized clean data.
      - Pre-stores a per-sample RNG.
      - Precomputes a combined noise scaling factor that divides the min–max scale (2/(max-min))
        by the per-BPM beta function, so noise can be injected directly in the normalized space.
    """
    def __init__(self, num_files, noise_factors=NOISE_FACTORS, base_seed=SEED):
        super().__init__()
        # Load raw clean data (already beta-scaled) and store it.
        clean_x, clean_y = load_clean_data()  # shape: (NBPMS, TOTAL_TURNS)
        self.clean_data_x = clean_x
        self.clean_data_y = clean_y

        # Read beta functions for noise scaling.
        model_dat = tfs.read(get_model_dir(beam=BEAM) / "twiss.dat", index="NAME")
        self.sqrt_betax = np.sqrt(model_dat["BETX"].values)  # shape: (NBPMS,)
        self.sqrt_betay = np.sqrt(model_dat["BETY"].values)

        # Compute global min and max on the raw clean data.
        self.min_x = torch.min(clean_x)
        self.max_x = torch.max(clean_x)
        self.min_y = torch.min(clean_y)
        self.max_y = torch.max(clean_y)

        # Precompute normalized clean data: norm = 2*(x-min)/(max-min) - 1
        norm_clean_x = 2 * (clean_x - self.min_x) / (self.max_x - self.min_x) - 1
        norm_clean_y = 2 * (clean_y - self.min_y) / (self.max_y - self.min_y) - 1
        # Add a channel dimension to get shape: (1, NBPMS, TOTAL_TURNS)
        self.norm_clean_x = norm_clean_x.unsqueeze(0)
        self.norm_clean_y = norm_clean_y.unsqueeze(0)

        # Precompute per-sample offsets.
        self.offsets = []
        rng_offsets = np.random.default_rng(base_seed)
        for i in range(num_files):
            if USE_OFFSETS:
                if i % NUM_SAME_OFFSET == 0:
                    offset = int(rng_offsets.integers(0, TOTAL_TURNS - NTURNS + 1))
                self.offsets.append(offset)
            else:
                self.offsets.append(0)
        self.num_files = num_files
        self.noise_factors = noise_factors
        self.base_seed = base_seed

        # Pre-store an RNG for each sample.
        self.rngs = [np.random.default_rng(base_seed + i) for i in range(num_files)]

        # Precompute the min–max scaling factor (a scalar) and then combine it with the beta functions.
        # self.scale_x = 2/(max_x-min_x) and similarly for y.
        min_max_scale_x = 2 / (self.max_x - self.min_x)
        min_max_scale_y = 2 / (self.max_y - self.min_y)
        # Precompute the combined noise scaling factors (one per BPM).
        self.noise_scale_x = (min_max_scale_x / self.sqrt_betax).float() # shape: (NBPMS,)
        self.noise_scale_y = (min_max_scale_y / self.sqrt_betay).float() # shape: (NBPMS,)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        start_idx = self.offsets[idx]
        end_idx = start_idx + NTURNS

        # Get the precomputed normalized clean data slice.
        clean_slice_norm_x = self.norm_clean_x[:, :, start_idx:end_idx]
        clean_slice_norm_y = self.norm_clean_y[:, :, start_idx:end_idx]

        # Determine the noise factor deterministically.
        factor_idx = idx % len(self.noise_factors)
        noise_factor = self.noise_factors[factor_idx]

        # Retrieve the pre-stored RNG for this sample.
        rng = self.rngs[idx]

        # Generate raw noise (vectorized) and then scale it:
        # Instead of dividing by beta functions and then multiplying by scale_x,
        # we precompute noise_scale_x = scale_x / sqrt(betax) so that:
        # noise = noise_factor * standard_normal * noise_scale_x (applied per BPM).
        noise_x = noise_factor * rng.standard_normal((NBPMS, NTURNS))
        noise_y = noise_factor * rng.standard_normal((NBPMS, NTURNS))
        noise_x = torch.tensor(noise_x, dtype=torch.float32) * self.noise_scale_x[:, None]
        noise_y = torch.tensor(noise_y, dtype=torch.float32) * self.noise_scale_y[:, None]

        # The noisy normalized data is simply the clean normalized slice plus the scaled noise.
        noisy_norm_x = clean_slice_norm_x + noise_x.unsqueeze(0)
        noisy_norm_y = clean_slice_norm_y + noise_y.unsqueeze(0)

        return {
            "noisy_x": noisy_norm_x,  # shape: (1, NBPMS, NTURNS)
            "noisy_y": noisy_norm_y,
            "clean_x": clean_slice_norm_x,
            "clean_y": clean_slice_norm_y,
        }

    def denormalise(self, norm_data: np.ndarray, plane: str) -> np.ndarray:
        """
        Inverse transforms a normalized output for a specific plane ('x' or 'y')
        back to the original scale, treating the channel as a whole 2D image.
        """
        if plane not in ["x", "y"]:
            raise ValueError("Plane must be 'x' or 'y'.")

        min_val = (self.min_x if plane == "x" else self.min_y).numpy()
        max_val = (self.max_x if plane == "x" else self.max_y).numpy()
        betas   = self.sqrt_betax if plane == "x" else self.sqrt_betay
        orig_data = (norm_data + 1) / 2 * (max_val - min_val) + min_val
        return orig_data * betas[:, None]


def load_data() -> tuple[DataLoader, DataLoader, BPMSDataset]:
    """Loads the training and validation data."""

    dataset = BPMSDataset(num_files=NUM_FILES)
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )
    return train_loader, val_loader, dataset


def build_sample_dict(sample: dict[str, np.ndarray], dataset: BPMSDataset) -> dict:
    """
    Given a list of batches, returns a dictionary with the X and Y samples,
    both in noisy and clean versions after inversion, for each plane.
    """
    sample_dict = {}
    for plane in ["x", "y"]:
        sample_dict[f"noisy_{plane}"] = dataset.denormalise(sample[f"noisy_{plane}"], plane=plane)
        sample_dict[f"clean_{plane}"] = dataset.denormalise(sample[f"clean_{plane}"], plane=plane)
        sample_dict[f"recon_{plane}"] = dataset.denormalise(sample[f"recon_{plane}"], plane=plane)

    return sample_dict
