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
    NBPMS,
    NOISE_FACTORS,
    NTURNS,
    NUM_NOISY_PER_CLEAN,
    SEED,
    TOTAL_TURNS,
    TRAIN_RATIO,
    NONOISE_INDEX,
    CLEAN_PARAM_LIST,
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

    assert x_data.shape[0] == NBPMS, f"Missing some BPMs, x data has shape: {x_data.shape}"
    assert y_data.shape[0] == NBPMS, f"Missing some BPMs, y data has shape: {y_data.shape}"

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
        base_seed=SEED,
    ):
        super().__init__()
        self.clean_param_list = (
            clean_param_list  # List of dicts with keys: beam, tunes, coupling.
        )
        self.num_files_per_clean = num_variations_per_clean
        self.noise_factors = noise_factors
        self.base_seed = base_seed
        self.total_files = len(clean_param_list) * num_variations_per_clean

        # Preload clean files and store per-file normalisation parameters.
        self.norm_data = []  # A list of dictionaries (one per clean file)
        self.num_clean = len(self.clean_param_list)
        for params in self.clean_param_list:
            sdds_data_path = get_tbt_path(
                params["beam"],
                TOTAL_TURNS,
                params["coupling"],
                params["tunes"],
                NONOISE_INDEX,
            )
            model_dir = get_model_dir(
                params["beam"], params["coupling"], params["tunes"]
            )
            clean_x, clean_y, sqrt_betax, sqrt_betay = load_clean_data(
                sdds_data_path, model_dir
            )
            # Compute normalisation parameters.
            min_x = torch.min(clean_x)
            max_x = torch.max(clean_x)
            min_y = torch.min(clean_y)
            max_y = torch.max(clean_y)
            sqrt_betax = torch.tensor(sqrt_betax, dtype=torch.float32)
            sqrt_betay = torch.tensor(sqrt_betay, dtype=torch.float32)
            norm_clean_x = 2 * (clean_x - min_x) / (max_x - min_x) - 1
            norm_clean_y = 2 * (clean_y - min_y) / (max_y - min_y) - 1
            noise_scale_x = ((2 / (max_x - min_x)) / sqrt_betax).view(1, -1, 1)
            noise_scale_y = ((2 / (max_y - min_y)) / sqrt_betay).view(1, -1, 1)

            # Store all needed parameters for later denormalisation.
            self.norm_data.append(
                {
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                    "sqrt_betax": sqrt_betax,
                    "sqrt_betay": sqrt_betay,
                    "noise_scale_x": noise_scale_x,
                    "noise_scale_y": noise_scale_y,
                    "norm_clean_x": norm_clean_x.unsqueeze(0),  # add channel dimension
                    "norm_clean_y": norm_clean_y.unsqueeze(0),
                }
            )

        # Precompute random clean indices, offsets, and noise factors for each sample.
        self.clean_indices = np.random.randint(0, self.num_clean, size=self.total_files)
        self.offsets = np.random.randint(0, TOTAL_TURNS - NTURNS + 1, size=self.total_files)
        self.noise_factors = np.random.choice(noise_factors, size=self.total_files, replace=True)
        
        # Introduce a noise multiplier that can be updated during training.
        self.current_noise_multiplier = 1.0

        # Initialise cache for slices.
        # self.slice_cache = {}
    def update_noise_multiplier(self, multiplier):
        self.current_noise_multiplier = multiplier

    def __len__(self):
        return self.total_files

    def __getitem__(self, idx):
        clean_idx = self.clean_indices[idx]  # Use precomputed clean_idx
        norm_info = self.norm_data[clean_idx]
        offset = self.offsets[idx]  # Use precomputed offset

        clean_slice_x = norm_info["norm_clean_x"][:, :, offset : offset + NTURNS]
        clean_slice_y = norm_info["norm_clean_y"][:, :, offset : offset + NTURNS]

        # Use precomputed noise_factors indexed by idx.
        noise_factor = self.noise_factors[idx] * self.current_noise_multiplier
        noise_x = torch.randn(clean_slice_x.shape, dtype=clean_slice_x.dtype) * noise_factor
        noise_y = torch.randn(clean_slice_y.shape, dtype=clean_slice_y.dtype) * noise_factor

        noisy_slice_x = clean_slice_x + noise_x * norm_info["noise_scale_x"]
        noisy_slice_y = clean_slice_y + noise_y * norm_info["noise_scale_y"]

        return {
            "clean_x": clean_slice_x,
            "clean_y": clean_slice_y,
            "noisy_x": noisy_slice_x,
            "noisy_y": noisy_slice_y,
            "norm_info": norm_info,  # include normalisation information in each sample
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
            betas = norm_info["sqrt_betax"]
        else:
            min_val = norm_info["min_y"].item()
            max_val = norm_info["max_y"].item()
            betas = norm_info["sqrt_betay"]

        # Inverse min–max scaling.
        orig_data = (norm_data + 1) / 2 * (max_val - min_val) + min_val
        return orig_data * betas[:, None]


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
