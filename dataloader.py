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

# Enable PyTorch optimizations
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.deterministic = (
    False  # Allow non-deterministic algorithms for speed
)


def load_clean_data(sdds_data_path, model_dir) -> tuple[torch.Tensor, torch.Tensor]:
    sdds_data = read_tbt(sdds_data_path)

    # Read the twiss file for beta functions.
    # model_dat = tfs.read(get_model_dir(beam, coupling, tunes) / "twiss.dat")

    # sqrt_betax = np.sqrt(model_dat["BETX"].values)
    # sqrt_betay = np.sqrt(model_dat["BETY"].values)
    # For now, let's not normalise the data.
    sqrt_betax = sqrt_betay = np.ones(NBPMS, dtype=np.float32)

    x_data = sdds_data.matrices[0].X.to_numpy()
    y_data = sdds_data.matrices[0].Y.to_numpy()

    assert x_data.shape[0] == NBPMS, (
        f"Missing some BPMs, x data has shape: {x_data.shape}"
    )
    assert y_data.shape[0] == NBPMS, (
        f"Missing some BPMs, y data has shape: {y_data.shape}"
    )

    # Since we're dividing by ones, skip the division for performance
    # More efficient tensor creation from numpy arrays
    return (
        torch.from_numpy(x_data.astype(np.float32)),
        torch.from_numpy(y_data.astype(np.float32)),
        sqrt_betax,
        sqrt_betay,
    )


def write_data(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    model_dir: Path,
    tbt_path: Path,
    nturns: int = NTURNS,
) -> tuple[Path, TbtData]:
    model_dat = tfs.read(model_dir / "twiss.tfs.bz2", index="NAME")
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
    out_data = TbtData(matrices=matrices, nturns=nturns)
    write_tbt(tbt_path, out_data)
    return tbt_path, out_data


def cyclic_slice_bpm_turns(raw_tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor of shape (NBPMS, TOTAL_TURNS), this function performs a cyclic
    permutation of the data to create a window of NTURNS turns, starting from turn 0.
    The output tensor will have shape (NBPMS, NTURNS).
    """
    flat_o = torch.randint(0, (TOTAL_TURNS - NTURNS) * NBPMS, (), dtype=torch.long)
    turn_o = flat_o // NBPMS
    bpm_o = flat_o % NBPMS
    # roll 2 dims in one go, then slice the first NTURNS columns
    rolled = torch.roll(raw_tensor, shifts=(bpm_o, -turn_o), dims=(0, 1))
    return rolled[:, :NTURNS]


def random_slice_turns(raw_tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a tensor of shape (NBPMS, TOTAL_TURNS), this function selects a random
    contiguous slice of NTURNS turns from the data.
    The output tensor will have shape (NBPMS, NTURNS).
    """
    # Select a random starting turn index using config constants
    max_start_turn = TOTAL_TURNS - NTURNS
    start_turn = torch.randint(0, max_start_turn + 1, (), dtype=torch.long).item()

    # Return the slice of NTURNS turns starting from the random position
    return raw_tensor[:, start_turn : start_turn + NTURNS]


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
        noise_factors = np.array(noise_factors, dtype=np.float32)

        # Load all clean files once into big lists so we never do it in __getitem__
        all_clean_x = []
        all_clean_y = []
        mean_x_list = []
        std_x_list = []
        mean_y_list = []
        std_y_list = []

        # First pass: collect all data to compute global statistics
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
            # Compute mean and std for each file over the NTURNS (so we have a single value per BPM)
            mean_x_list.append(clean_x.mean())
            std_x_list.append(clean_x.std())
            mean_y_list.append(clean_y.mean())
            std_y_list.append(clean_y.std())

        # Convert lists to tensors for faster indexing
        self.cached_clean_x = all_clean_x  # Keep as list for memory efficiency
        self.cached_clean_y = all_clean_y  # Keep as list for memory efficiency
        self.mean_x = mean_x_list  # shape: (num_files, NBPMS)
        self.std_x = std_x_list  # shape: (num_files, NBPMS)
        self.mean_y = mean_y_list  # shape: (num_files, NBPMS)
        self.std_y = std_y_list  # shape: (num_files, NBPMS)

        # Next, define how many total samples we want
        self.num_clean = len(self.clean_param_list)
        self.total_files = self.num_clean * num_variations_per_clean

        # Precompute random indices, offsets, noise - convert to tensors for speed
        self.clean_indices = torch.from_numpy(
            np.random.randint(0, self.num_clean, size=self.total_files, dtype=np.int64)
        )
        self.noise_factors = torch.from_numpy(
            np.random.choice(noise_factors, size=self.total_files, replace=True)
        )
        # Pre-generate missing data masks to avoid repeated random number generation
        self.missing_masks_x = torch.rand(self.total_files, NBPMS) < MISSING_PROB
        self.missing_masks_y = torch.rand(self.total_files, NBPMS) < MISSING_PROB

    def __len__(self):
        return self.total_files

    def __getitem__(self, idx):
        # Identify which file we want
        file_idx = self.clean_indices[idx]
        noise_factor = self.noise_factors[idx]

        # Pull out the raw (un-normalized) data from memory
        raw_x = self.cached_clean_x[file_idx]  # shape (NBPMS, TOTAL_TURNS)
        raw_y = self.cached_clean_y[file_idx]

        # Extract a random window of NTURNS turns
        window_x = random_slice_turns(raw_x)
        window_y = random_slice_turns(raw_y)

        # Add noise in raw domain (before normalization for physical consistency)
        noise_x = torch.randn_like(window_x) * noise_factor
        noise_y = torch.randn_like(window_y) * noise_factor
        noisy_window_x = window_x + noise_x
        noisy_window_y = window_y + noise_y

        # Use tensor indexing for faster access
        mean_x = self.mean_x[file_idx]  # shape: (1,)
        std_x = self.std_x[file_idx]  # shape: (1,)
        mean_y = self.mean_y[file_idx]  # shape: (1,)
        std_y = self.std_y[file_idx]  # shape: (1,)

        norm_slice_x = (window_x - mean_x) / std_x
        norm_slice_y = (window_y - mean_y) / std_y
        noisy_slice_x = (noisy_window_x - mean_x) / std_x
        noisy_slice_y = (noisy_window_y - mean_y) / std_y

        # Use pre-computed masks for missing data
        missing_mask_x = self.missing_masks_x[idx]
        missing_mask_y = self.missing_masks_y[idx]
        noisy_slice_x[missing_mask_x, :] = 0
        noisy_slice_y[missing_mask_y, :] = 0

        # Add the channel dimension
        # The final shape of the tensors is (1, NBPMS, NTURNS)
        return {
            "clean_x": norm_slice_x.unsqueeze(0),
            "clean_y": norm_slice_y.unsqueeze(0),
            "noisy_x": noisy_slice_x.unsqueeze(0),
            "noisy_y": noisy_slice_y.unsqueeze(0),
            "mean_x": mean_x,
            "mean_y": mean_y,
            "std_x": std_x,
            "std_y": std_y,
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
            mean_val = norm_info["mean_x"].item()
            std_val = norm_info["std_x"].item()
            # betas = norm_info["sqrt_betax"]
        else:
            mean_val = norm_info["mean_y"].item()
            std_val = norm_info["std_y"].item()
            # betas = norm_info["sqrt_betay"]

        # Inverse z-score normalization.
        orig_data = norm_data * std_val + mean_val
        return orig_data  # * betas[:, None]


def load_data(num_workers: int = 4) -> tuple[DataLoader, DataLoader, BPMSDataset]:
    dataset = BPMSDataset(
        clean_param_list=CLEAN_PARAM_LIST, num_variations_per_clean=NUM_NOISY_PER_CLEAN
    )
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    # Optimize DataLoader settings for better performance
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=4,  # Increased prefetch for better overlap
        drop_last=True,  # Avoid smaller final batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2,  # Less prefetch needed for validation
        drop_last=False,  # Keep all validation data
    )
    return train_loader, val_loader, dataset


def denormalise_sample_dict(
    sample: dict[str, np.ndarray], dataset: BPMSDataset
) -> dict:
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
    Store global mean/std for x and y as JSON so we can reload in inference.
    """
    norm_params = {
        "mean_x": dataset.mean_x.mean(dim=0).item(),
        "std_x": dataset.std_x.mean(dim=0).item(),
        "mean_y": dataset.mean_y.mean(dim=0).item(),
        "std_y": dataset.std_y.mean(dim=0).item(),
    }
    with open(filepath, "w") as f:
        json.dump(norm_params, f, indent=2)
