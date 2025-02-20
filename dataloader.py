from typing import Tuple

import numpy as np
import pandas as pd
import tfs
import torch
from torch.utils.data import DataLoader, Dataset
from turn_by_turn import TbtData, TransverseData
from turn_by_turn.lhc import read_tbt, write_tbt

from config import (
    BATCH_SIZE,
    BEAM,
    NBPMS,
    NOISE_FACTOR,
    NTURNS,
    NUM_FILES,
    SAMPLE_INDEX,
    SEED,
)
from rdt_functions import get_model_dir, get_tbt_path


def load_clean_data() -> torch.Tensor:
    # Load zero-noise data
    sdds_data_path = get_tbt_path(beam=BEAM, nturns=NTURNS, index=-1)
    sdds_data = read_tbt(sdds_data_path)

    # Read twiss file for beta functions
    model_dat = tfs.read(get_model_dir(beam=BEAM) / "twiss.dat")
    sqrt_betax = np.sqrt(model_dat["BETX"].values)  # For X plane
    sqrt_betay = np.sqrt(model_dat["BETY"].values)  # For Y plane

    # Extract data; assume sdds_data.matrices[0] contains both 'X' and 'Y'
    # Original shapes: (NTURNS, NBPMS)
    x_data = sdds_data.matrices[0].X.to_numpy().T / sqrt_betax[None, :]
    y_data = sdds_data.matrices[0].Y.to_numpy().T / sqrt_betay[None, :]

    assert x_data.shape == y_data.shape == (NTURNS, NBPMS), "Data shape mismatch"

    # Concatenate along axis=1 so that each sample becomes (NTURNS, 2*NBPMS)
    xy_data = np.concatenate([x_data, y_data], axis=1)
    # Return a tensor; in __getitem__ we will transpose so that shape becomes (2*NBPMS, NTURNS)
    return torch.tensor(xy_data, dtype=torch.float32)


def write_data(data: torch.Tensor, noise_index: int) -> np.ndarray:
    model_dat = tfs.read(get_model_dir(beam=BEAM) / "twiss.dat", index="NAME")
    sqrt_betax = np.sqrt(
        model_dat["BETX"].values
    )  # Square root of beta x at each BPM (1D array)
    sqrt_betay = np.sqrt(
        model_dat["BETY"].values
    )  # Square root of beta y at each BPM (1D array)

    x_bpm_names = model_dat.index.to_list()
    y_bpm_names = model_dat.index.to_list()

    print(data.shape)

    assert data.shape == (2 * NBPMS, NTURNS), "Data shape mismatch"

    x_data = data[:NBPMS, :] * sqrt_betax[:, None]
    y_data = data[NBPMS:, :] * sqrt_betay[:, None]

    # Write out the denoised frequency and amplitude data
    out_path = get_tbt_path(beam=BEAM, nturns=NTURNS, index=noise_index)

    matrices = [
        TransverseData(
            X=pd.DataFrame(
                index=x_bpm_names,
                data=x_data,
                dtype=float,
            ),
            Y=pd.DataFrame(
                index=y_bpm_names,
                data=y_data,
                dtype=float,
            ),
        )
    ]
    out_data = TbtData(matrices=matrices, nturns=NTURNS)
    write_tbt(out_path, out_data)  # Write out the denoised data
    return out_path, out_data


class BPMSDataset(Dataset):
    def __init__(self, num_files, noise_factor=NOISE_FACTOR, base_seed=SEED):
        """
        Args:
            num_files (int): Number of samples in the dataset.
            noise_factor (float): Scale of the noise added in the original domain.
            base_seed (int): Base seed for reproducibility.
        """
        super().__init__()
        self.num_files = num_files
        self.noise_factor = noise_factor

        # 1) Load the clean data (shape: (NTURNS, 2*NBPMS)) and transpose => (2*NBPMS, NTURNS)
        raw_data = self.load_clean_data().T
        self.clean_data_x = raw_data[:NBPMS, :]   # (NBPMS, NTURNS)
        self.clean_data_y = raw_data[NBPMS:, :]   # (NBPMS, NTURNS)

        # 2) Compute arcsin of the original clean data for each plane
        #    so we can derive mean and std in the arcsin domain.
        self.eps = 1e-8
        arcsin_x = torch.arcsin(self.clean_data_x, self.eps)
        arcsin_y = torch.arcsin(self.clean_data_y, self.eps)

        # 3) Compute mean and std of arcsin-transformed data, per BPM (dim=1 => across turns)
        self.arcsin_mean_x = torch.mean(arcsin_x, dim=1)
        self.arcsin_std_x = torch.std(arcsin_x, dim=1)

        self.arcsin_mean_y = torch.mean(arcsin_y, dim=1)
        self.arcsin_std_y = torch.std(arcsin_y, dim=1)

        # 4) Prepare noise seeds for reproducibility
        rng = np.random.default_rng(base_seed)
        self.seeds = rng.integers(low=0, high=1_000_000, size=num_files)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        """
        Returns a dictionary:
        {
            "noisy": <arcsin-normalized noisy data>,
            "clean": <arcsin-normalized clean data>,
            "plane": <"X" or "Y">
        }
        """
        if idx % 2 == 0:
            plane = "X"
            clean = self.clean_data_x
            arcsin_mean = self.arcsin_mean_x
            arcsin_std = self.arcsin_std_x
        else:
            plane = "Y"
            clean = self.clean_data_y
            arcsin_mean = self.arcsin_mean_y
            arcsin_std = self.arcsin_std_y

        # 1) Generate noise in the original domain
        rng_i = np.random.default_rng(self.seeds[idx])
        noise_array = rng_i.normal(
            loc=0.0, scale=self.noise_factor, size=clean.shape
        )
        noise_tensor = torch.tensor(noise_array, dtype=torch.float32)

        # 2) Create noisy data in original domain
        noisy = clean + noise_tensor

        # 3) arcsin-transform both clean & noisy (clipping to [-1,1])
        arcsin_noisy = torch.arcsin(noisy, self.eps)
        arcsin_clean = torch.arcsin(clean, self.eps)

        # 4) Normalize arcsin data => (x - mean) / std
        arcsin_noisy_norm = (arcsin_noisy - arcsin_mean[:, None]) / (arcsin_std[:, None] + self.eps)
        arcsin_clean_norm = (arcsin_clean - arcsin_mean[:, None]) / (arcsin_std[:, None] + self.eps)

        return {
            "noisy": arcsin_noisy_norm,
            "clean": arcsin_clean_norm,
            "plane": plane
        }

    def load_clean_data(self) -> torch.Tensor:
        """
        Loads zero-noise data from SDDS, returns a (NTURNS, 2*NBPMS) float tensor.
        """
        sdds_data_path = get_tbt_path(beam=BEAM, nturns=NTURNS, index=-1)
        sdds_data = read_tbt(sdds_data_path)

        model_dat = tfs.read(get_model_dir(beam=BEAM) / "twiss.dat")
        sqrt_betax = np.sqrt(model_dat["BETX"].values)
        sqrt_betay = np.sqrt(model_dat["BETY"].values)

        x_data = sdds_data.matrices[0].X.to_numpy().T / sqrt_betax[None, :]
        y_data = sdds_data.matrices[0].Y.to_numpy().T / sqrt_betay[None, :]

        xy_data = np.concatenate([x_data, y_data], axis=1)
        return torch.tensor(xy_data, dtype=torch.float32)

    def denormalise(self, normalized_output: torch.Tensor, plane: str) -> torch.Tensor:
        """
        Reverses arcsin normalization:
          1) arcsin_out = normalized_output * std + mean
          2) out = sin(arcsin_out)
        """
        if plane == "X":
            arcsin_mean = self.arcsin_mean_x
            arcsin_std = self.arcsin_std_x
        elif plane == "Y":
            arcsin_mean = self.arcsin_mean_y
            arcsin_std = self.arcsin_std_y
        else:
            raise ValueError("Plane must be 'X' or 'Y'")

        arcsin_out = normalized_output * (arcsin_std[:, None] + self.eps) + arcsin_mean[:, None]
        return torch.sin(arcsin_out)



def load_data() -> Tuple[DataLoader, DataLoader, BPMSDataset]:
    """
    Loads the training and validation data (arcsin-based).
    """
    dataset = BPMSDataset(num_files=NUM_FILES)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

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


def build_sample_dict(sample_list: list, dataset: BPMSDataset) -> dict:
    """
    Given a list of batches, returns a dictionary with the X and Y samples at the SAMPLE_INDEX,
    both in noisy and clean versions. Also returns the combined noisy and clean samples.
    """
    x_sample = 2 * SAMPLE_INDEX
    y_sample = 2 * SAMPLE_INDEX + 1

    # Take the correct samples
    x_sample = sample_list[x_sample]
    y_sample = sample_list[y_sample]

    # Denormalise the samples
    x_noisy = dataset.denormalise(x_sample["noisy"], "X")
    x_clean = dataset.denormalise(x_sample["clean"], "X")
    x_denoised = dataset.denormalise(x_sample["denoised"], "X")

    y_noisy = dataset.denormalise(y_sample["clean"], "Y")
    y_clean = dataset.denormalise(y_sample["clean"], "Y")
    y_denoised = dataset.denormalise(y_sample["denoised"], "Y")

    return {
        "x": x_noisy,
        "y": y_noisy,
        "x_clean": x_clean,
        "y_clean": y_clean,
        "x_denoised": x_denoised,
        "y_denoised": y_denoised,
    }
