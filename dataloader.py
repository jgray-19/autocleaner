from typing import Union

import numpy as np
import pandas as pd
import tfs
import torch
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
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
    TRAIN_RATIO,
    get_model_dir,
    get_tbt_path,
)


def load_clean_data() -> torch.Tensor:
    # Load zero-noise data
    sdds_data_path = get_tbt_path(beam=BEAM, nturns=NTURNS, index=-1)
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
        # Load clean data.
        # load_clean_data() returns a tensor of shape (NTURNS, 2*NBPMS)
        # We transpose to get shape (2*NBPMS, NTURNS)
        raw = load_clean_data().T  # shape: (2*NBPMS, NTURNS)
        
        # Split into X and Y channels.
        clean_data_x = raw[:NBPMS, :]  # shape: (NBPMS, NTURNS)
        clean_data_y = raw[NBPMS:, :]  # shape: (NBPMS, NTURNS)
        
        self.raw_data = torch.stack([clean_data_x, clean_data_y], dim=0)

        # Initialize a QuantileTransformer for each channel.
        self.qt_x = QuantileTransformer(output_distribution='normal')
        self.qt_y = QuantileTransformer(output_distribution='normal')

        # For each channel, flatten the entire 2D image and fit the transformer.
        x_flat = clean_data_x.reshape(-1, 1)  # shape: (NBPMS*NTURNS, 1)
        x_norm_flat = self.qt_x.fit_transform(x_flat)
        norm_clean_x = x_norm_flat.reshape(NBPMS, NTURNS)

        y_flat = clean_data_y.reshape(-1, 1)  # shape: (NBPMS*NTURNS, 1)
        y_norm_flat = self.qt_y.fit_transform(y_flat)
        norm_clean_y = y_norm_flat.reshape(NBPMS, NTURNS)

        # Store the normalized clean data as a tensor with shape (2, NBPMS, NTURNS)
        self.clean_norm = torch.tensor(
            np.stack([norm_clean_x, norm_clean_y], axis=0),
            dtype=torch.float32
        )

        # Precompute noisy samples.
        self.noisy_norm = []
        rng = np.random.default_rng(base_seed)

        self.num_files = num_files
        for _ in range(num_files):
            # Generate noise for each channel.
            noise_x = rng.normal(loc=0.0, scale=noise_factor, size=clean_data_x.shape)
            noise_y = rng.normal(loc=0.0, scale=noise_factor, size=clean_data_y.shape)

            # Create noisy data by adding noise.
            noisy_x = clean_data_x + noise_x
            noisy_y = clean_data_y + noise_y

            # Normalize each channel using the already fitted transformers.
            # Flatten the 2D array, transform, then reshape back.
            noisy_x_flat = noisy_x.reshape(-1, 1)
            noisy_x_norm_flat = self.qt_x.transform(noisy_x_flat)
            noisy_x_norm = noisy_x_norm_flat.reshape(NBPMS, NTURNS)

            noisy_y_flat = noisy_y.reshape(-1, 1)
            noisy_y_norm_flat = self.qt_y.transform(noisy_y_flat)
            noisy_y_norm = noisy_y_norm_flat.reshape(NBPMS, NTURNS)

            # Stack to create a (2, NBPMS, NTURNS) tensor.
            sample = torch.tensor(
                np.stack([noisy_x_norm, noisy_y_norm], axis=0),
                dtype=torch.float32
            )
            self.noisy_norm.append(sample)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        # Return a dict containing the noisy sample and the clean target,
        # both with shape (2, NBPMS, NTURNS).
        return {
            "noisy": self.noisy_norm[idx],
            "clean": self.clean_norm
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



class BPMSDatasetNormalised(Dataset):
    """
    A dataset that produces samples with shape (2, NBPMS, NTURNS) where each channel
    (X and Y) is normalized to the range [-1, 1] using a MinMaxScaler.
    """
    def __init__(self, num_files, noise_factor=NOISE_FACTOR, base_seed=SEED):
        super().__init__()
        # Load clean data: expected shape (NTURNS, 2*NBPMS)
        raw = load_clean_data().T  # shape: (2*NBPMS, NTURNS)
        # Split into X and Y channels.
        clean_data_x = raw[:NBPMS, :]  # (NBPMS, NTURNS)
        clean_data_y = raw[NBPMS:, :]  # (NBPMS, NTURNS)
        
        # Initialize scalers for each channel with feature_range (-1, 1)
        self.scaler_x = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        # Initialize a QuantileTransformer for each channel.
        self.qt_x = QuantileTransformer(output_distribution='normal')
        self.qt_y = QuantileTransformer(output_distribution='normal')
        
        # Fit the scalers on the clean data (flattened).
        x_flat = clean_data_x.reshape(-1, 1)
        y_flat = clean_data_y.reshape(-1, 1)

        x_gauss_flat = self.qt_x.fit_transform(x_flat)
        y_gauss_flat = self.qt_y.fit_transform(y_flat)

        x_norm_flat = self.scaler_x.fit_transform(x_gauss_flat)
        y_norm_flat = self.scaler_y.fit_transform(y_gauss_flat)
        
        # Reshape back to the original 2D format.
        norm_clean_x = x_norm_flat.reshape(NBPMS, NTURNS)
        norm_clean_y = y_norm_flat.reshape(NBPMS, NTURNS)
        
        # Store the normalized clean data as a tensor of shape (2, NBPMS, NTURNS)
        self.clean_norm = torch.tensor(
            np.stack([norm_clean_x, norm_clean_y], axis=0),
            dtype=torch.float32
        )
        
        # Precompute noisy samples.
        self.noisy_norm = []
        rng = np.random.default_rng(base_seed)
        for _ in range(num_files):
            # Generate noise for each channel.
            noise_x = rng.normal(loc=0.0, scale=noise_factor, size=clean_data_x.shape)
            noise_y = rng.normal(loc=0.0, scale=noise_factor, size=clean_data_y.shape)
            
            # Create noisy data by adding noise.
            noisy_x = clean_data_x + noise_x
            noisy_y = clean_data_y + noise_y
            
            # Normalize each channel using the already fitted transformers.
            noisy_x_flat = noisy_x.reshape(-1, 1)
            noisy_y_flat = noisy_y.reshape(-1, 1)
            
            noisy_x_gauss_flat = self.qt_x.transform(noisy_x_flat)
            noisy_y_gauss_flat = self.qt_y.transform(noisy_y_flat)

            noisy_x_norm_flat = self.scaler_x.transform(noisy_x_gauss_flat)
            noisy_y_norm_flat = self.scaler_y.transform(noisy_y_gauss_flat)

            noisy_x_norm = noisy_x_norm_flat.reshape(NBPMS, NTURNS)
            noisy_y_norm = noisy_y_norm_flat.reshape(NBPMS, NTURNS)
            
            sample = torch.tensor(
                np.stack([noisy_x_norm, noisy_y_norm], axis=0),
                dtype=torch.float32
            )
            self.noisy_norm.append(sample)
            
        self.num_files = num_files

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        return {
            "noisy": self.noisy_norm[idx],
            "clean": self.clean_norm
        }
    
    def denormalise(self, normalized_output: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transforms a normalized output (shape (2, NBPMS, NTURNS)) back to the original scale.
        """
        norm_x = normalized_output[0].detach().cpu().numpy()
        norm_y = normalized_output[1].detach().cpu().numpy()
        
        gauss_x_flat = self.scaler_x.inverse_transform(norm_x.reshape(-1, 1))
        gauss_y_flat = self.scaler_y.inverse_transform(norm_y.reshape(-1, 1))

        orig_x_flat = self.qt_x.inverse_transform(gauss_x_flat)
        orig_y_flat = self.qt_y.inverse_transform(gauss_y_flat)


        orig_x = orig_x_flat.reshape(NBPMS, NTURNS)
        orig_y = orig_y_flat.reshape(NBPMS, NTURNS)
        orig = torch.tensor(
            np.stack([orig_x, orig_y], axis=0),
            dtype=torch.float32
        )
        return orig