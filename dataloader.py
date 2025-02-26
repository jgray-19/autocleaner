from typing import Union

import numpy as np
import pandas as pd
import tfs
import torch
from sklearn.preprocessing import QuantileTransformer
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
    SEED,
    TOTAL_TURNS,
    TRAIN_RATIO,
    get_model_dir,
    get_tbt_path,
)


def load_clean_data() -> torch.Tensor:
    # Load zero-noise data
    sdds_data_path = get_tbt_path(beam=BEAM, nturns=TOTAL_TURNS, index=-1)
    sdds_data = read_tbt(sdds_data_path)

    # Read twiss file for beta functions
    model_dat = tfs.read(get_model_dir(beam=BEAM) / "twiss.dat")
    sqrt_betax = np.sqrt(model_dat["BETX"].values)  # For X plane
    sqrt_betay = np.sqrt(model_dat["BETY"].values)  # For Y plane

    # Extract data; assume sdds_data.matrices[0] contains both 'X' and 'Y'
    x_data = sdds_data.matrices[0].X.to_numpy().T / sqrt_betax[None, :]
    y_data = sdds_data.matrices[0].Y.to_numpy().T / sqrt_betay[None, :]

    assert x_data.shape == y_data.shape == (NTURNS, NBPMS), "Data shape mismatch"

    # Concatenate along axis=1 so that each sample becomes (NTURNS, 2*NBPMS)
    xy_data = np.concatenate([x_data, y_data], axis=1)

    return torch.tensor(xy_data, dtype=torch.float32)


def write_data(data: torch.Tensor, noise_index: int=2) -> np.ndarray:
    model_dat = tfs.read(get_model_dir(beam=BEAM) / "twiss.dat", index="NAME")
    sqrt_betax = np.sqrt(model_dat["BETX"].values)
    sqrt_betay = np.sqrt(model_dat["BETY"].values)
    x_bpm_names = model_dat.index.to_list()
    y_bpm_names = model_dat.index.to_list()

    assert data.shape == (2, NBPMS, NTURNS), "Data shape mismatch"

    x_data = data[0, :, :] * sqrt_betax[:, None]
    y_data = data[1, :, :] * sqrt_betay[:, None]

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
    A dataset that produces samples with shape (n_planes, nbpms, nturns),
    where n_planes is 2 (X and Y). The normalization for each channel is
    computed over the entire 2D image (flattened) rather than per BPM.
    """
    def __init__(self, num_files, noise_factor=NOISE_FACTOR, base_seed=SEED):
        super().__init__()
        # Load full clean data; expected shape: (2*NBPMS, TOTAL_TURNS) where TOTAL_TURNS==1500
        raw = load_clean_data().T  # shape: (2*NBPMS, TOTAL_TURNS)
        # Split into channels for X and Y
        clean_data_x = raw[:NBPMS, :]  # shape: (NBPMS, TOTAL_TURNS)
        clean_data_y = raw[NBPMS:, :]  # shape: (NBPMS, TOTAL_TURNS)

        max_start = TOTAL_TURNS - NTURNS  # 500 turns (NTURNS is the window size)

        # Initialize QuantileTransformers to map data to a normal distribution.
        self.qt_x = QuantileTransformer(output_distribution='normal')
        self.qt_y = QuantileTransformer(output_distribution='normal')

        # Fit transformers on the full clean data for each channel.
        x_flat = clean_data_x.reshape(-1, 1) # shape: (NBPMS*TOTAL_TURNS, 1)
        x_norm_flat = self.qt_x.fit_transform(x_flat)
        norm_clean_x_full = x_norm_flat.reshape(NBPMS, TOTAL_TURNS)

        y_flat = clean_data_y.reshape(-1, 1)
        y_norm_flat = self.qt_y.fit_transform(y_flat)
        norm_clean_y_full = y_norm_flat.reshape(NBPMS, TOTAL_TURNS)

        # Create a full normalized clean tensor.
        full_clean_norm = torch.tensor(
            np.stack([norm_clean_x_full, norm_clean_y_full], axis=0),
            dtype=torch.float32
        )  # shape: (2, NBPMS, total_turns)

        # Prepare lists to store each sample's cropped clean and noisy data.
        self.clean_norm_list = []
        self.noisy_norm_list = []

        rng = np.random.default_rng(base_seed)
        for _ in range(num_files):
            # Choose a random window start index.
            window_start = np.random.randint(0, max_start)
            # Crop the normalized clean window for this sample.
            clean_window = full_clean_norm[:, :, window_start:window_start + NTURNS]
            self.clean_norm_list.append(clean_window)

            # Generate noisy data on the full raw clean data.
            noise_x = rng.normal(loc=0.0, scale=noise_factor, size=clean_data_x.shape)
            noise_y = rng.normal(loc=0.0, scale=noise_factor, size=clean_data_y.shape)

            # Create noisy data by adding noise.
            noisy_x = clean_data_x + noise_x
            noisy_y = clean_data_y + noise_y

            # Normalize the noisy data using the previously fitted transformers.
            noisy_x_flat = noisy_x.reshape(-1, 1)
            noisy_x_norm_flat = self.qt_x.transform(noisy_x_flat)
            noisy_x_norm_full = noisy_x_norm_flat.reshape(clean_data_x.shape[0], TOTAL_TURNS)

            noisy_y_flat = noisy_y.reshape(-1, 1)
            noisy_y_norm_flat = self.qt_y.transform(noisy_y_flat)
            noisy_y_norm_full = noisy_y_norm_flat.reshape(clean_data_y.shape[0], TOTAL_TURNS)

            noisy_full_norm = torch.tensor(
                np.stack([noisy_x_norm_full, noisy_y_norm_full], axis=0),
                dtype=torch.float32
            )  # shape: (2, NBPMS, total_turns)

            # Crop the same window from the noisy data.
            noisy_window = noisy_full_norm[:, :, window_start:window_start + NTURNS]
            self.noisy_norm_list.append(noisy_window)

        self.num_files = num_files

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        # Return a dict containing the noisy sample and the clean target,
        # both with shape (2, NBPMS, NTURNS).
        return {
            "noisy": self.noisy_norm_list[idx],
            "clean": self.clean_norm_list[idx],
        }

    def denormalise(self, normalized_output: torch.Tensor) -> torch.Tensor:
        """
        Inverse transforms a normalized output (of shape (2, NBPMS, NTURNS))
        back to the original scale, treating each channel as a whole 2D image.
        """
        # For channel X.
        norm_x = normalized_output[0].detach().cpu().numpy()
        norm_x_flat = norm_x.reshape(-1, 1)
        orig_x_flat = self.qt_x.inverse_transform(norm_x_flat)
        orig_x = orig_x_flat.reshape(NBPMS, NTURNS)

        # For channel Y.
        norm_y = normalized_output[1].detach().cpu().numpy()
        norm_y_flat = norm_y.reshape(-1, 1)
        orig_y_flat = self.qt_y.inverse_transform(norm_y_flat)
        orig_y = orig_y_flat.reshape(NBPMS, NTURNS)

        orig = torch.tensor(
            np.stack([orig_x, orig_y], axis=0),
            dtype=torch.float32
        )
        return orig

def load_data() -> Union[DataLoader, DataLoader, BPMSDataset]:
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
        num_workers=2,                 # Lower worker count can help
        pin_memory=True,
        persistent_workers=True,       # Reuse dataloader workers
        prefetch_factor=2,            # Adjust to taste
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader, dataset


def build_sample_dict(sample_list: list, dataset: BPMSDataset) -> dict:
    """
    Given a list of batches, returns a dictionary with the X and Y samples at SAMPLE_INDEX,
    both in noisy and clean versions after inversion.
    """
    # Take a sample from the sample list.
    sample = sample_list[0]  
    
    # 'noisy' is of shape (2, NBPMS, NTURNS) where index 0 is X and index 1 is Y
    denorm_sample = dataset.denormalise(sample["noisy"])
    denorm_clean = dataset.denormalise(sample["clean"])
    denorm_denoised = dataset.denormalise(sample["denoised"])

    # You can access channels directly:
    x_noisy = denorm_sample[0]  # shape: (NBPMS, NTURNS)
    x_clean = denorm_clean[0]  # shape: (NBPMS, NTURNS)
    x_denoised = denorm_denoised[0]  # shape: (NBPMS, NTURNS)

    y_noisy = denorm_sample[1]  # shape: (NBPMS, NTURNS)
    y_clean = denorm_clean[1]  # shape: (NBPMS, NTURNS)
    y_denoised = denorm_denoised[1]  # shape: (NBPMS, NTURNS)

    return {
        "x": x_noisy,
        "y": y_noisy,
        "x_clean": x_clean,
        "y_clean": y_clean,
        "x_denoised": x_denoised,
        "y_denoised": y_denoised,
    }