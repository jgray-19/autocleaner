import numpy as np
import pandas as pd
import tfs
import torch
from torch.utils.data import DataLoader, Dataset
from turn_by_turn import TbtData, TransverseData
from turn_by_turn.lhc import read_tbt, write_tbt

from config import BATCH_SIZE, BEAM, NBPMS, NOISE_FACTOR, NTURNS, NUM_FILES, SEED
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
    sqrt_betax = np.sqrt(model_dat["BETX"].values)  # Square root of beta x at each BPM (1D array)
    sqrt_betay = np.sqrt(model_dat["BETY"].values)  # Square root of beta y at each BPM (1D array)

    x_bpm_names = model_dat.index.to_list()
    y_bpm_names = model_dat.index.to_list()

    print(data.shape)

    assert data.shape == (2*NBPMS, NTURNS), "Data shape mismatch"

    x_data = data[:NBPMS, :] * sqrt_betax[:, None]
    y_data = data[NBPMS:, :] * sqrt_betay[:, None]
    
    # Write out the denoised frequency and amplitude data
    out_path = get_tbt_path(beam=BEAM, nturns=NTURNS, index=noise_index)

    matrices = [
        TransverseData(
            X=pd.DataFrame(
                index=x_bpm_names, data=x_data, dtype=float,
            ),
            Y=pd.DataFrame(
                index=y_bpm_names, data=y_data, dtype=float,
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

        # Load the clean data once (original shape: [NBPMS, NTURNS])
        raw_data = load_clean_data()  # shape: (NTURNS, 2*NBPMS)
        self.clean_data = raw_data.T   # Now shape: (2*NBPMS, NTURNS)

        # Compute global min and max for normalization
        self.data_min = torch.min(self.clean_data)
        self.data_max = torch.max(self.clean_data)

        self.clean_data_norm = self._minmax_normalize(self.clean_data)

        # Precompute a unique seed for each sample for reproducible noise
        rng = np.random.default_rng(base_seed)
        self.seeds = rng.integers(low=0, high=1_000_000, size=num_files)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        """
        Returns:
            noisy_data_norm (torch.Tensor): Noisy sample, shape (NBPMS, NTURNS), normalized to [-1, 1].
            clean_data_norm (torch.Tensor): Clean sample, shape (NBPMS, NTURNS), normalized to [-1, 1].
        """
        # Create an RNG instance for the given sample using its stored seed
        rng = np.random.default_rng(self.seeds[idx])
        # Generate noise on the fly (same shape as self.clean_data)
        noise_array = rng.normal(loc=0.0, scale=self.noise_factor, size=self.clean_data.shape)
        noise = torch.tensor(noise_array, dtype=torch.float32)
        
        # Add noise to the clean data (in the original domain)
        noisy_data = self.clean_data + noise

        # Normalize both clean and noisy data to [-1, 1] using the global stats
        noisy_data_norm = self._minmax_normalize(noisy_data)
        # clean_data_norm = self._minmax_normalize(self.clean_data)

        return noisy_data_norm, self.clean_data_norm

    def _minmax_normalize(self, x: torch.Tensor, eps=1e-8) -> torch.Tensor:
        """Normalize x to [-1, 1] using the precomputed global min and max."""
        return 2.0 * (x - self.data_min) / (self.data_max - self.data_min + eps) - 1.0

    def denormalise(self, normalized_output: torch.Tensor, eps=1e-8) -> torch.Tensor:
        """
        Converts a normalized tensor (in range [-1, 1]) back to its original scale.

        Args:
            normalized_output (torch.Tensor): Output from the model (assumed to be in [-1, 1]).
            eps (float): Small constant to prevent division by zero.

        Returns:
            torch.Tensor: Denormalized output in the original data scale.
        """
        return self.data_min + ((normalized_output + 1) / 2) * (self.data_max - self.data_min + eps)


def load_data():
    train_size = int(0.8 * NUM_FILES)
    dataset = BPMSDataset(num_files=NUM_FILES)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)

    return train_loader, val_loader, dataset
