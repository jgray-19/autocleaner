from typing import Union
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
    NUM_PLANES,
    SEED,
    TRAIN_RATIO,
    get_model_dir,
    get_tbt_path,
)
from sklearn.preprocessing import QuantileTransformer


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

    assert data.shape == (2 * NBPMS, NTURNS), "Data shape mismatch"

    x_data = data[:NBPMS, :] * sqrt_betax[:, None]
    y_data = data[NBPMS:, :] * sqrt_betay[:, None]

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
    def __init__(self, num_files, noise_factor=NOISE_FACTOR, base_seed=SEED):
        """
        Args:
            num_files (int): Number of samples (per plane) in the dataset.
            noise_factor (float): Scale of the noise added.
            base_seed (int): Seed for reproducibility.
        """
        super().__init__()
        # Each sample includes both planes.
        self.num_files = num_files * NUM_PLANES  
        self.noise_factor = noise_factor

        # Load clean data (shape: (NTURNS, 2*NBPMS)) and transpose => (2*NBPMS, NTURNS)
        self.raw_data = load_clean_data().T  
        # Split into X and Y channels (each of shape: (NBPMS, NTURNS))
        self.clean_data_x = self.raw_data[:NBPMS, :]
        self.clean_data_y = self.raw_data[NBPMS:, :]

        # Initialize QuantileTransformers
        self.qt_x = QuantileTransformer(output_distribution='normal')
        self.qt_y = QuantileTransformer(output_distribution='normal')

        # Fit and transform the clean data
        self.clean_data_norm_x = torch.tensor(self.qt_x.fit_transform(self.clean_data_x.T).T, dtype=torch.float32)
        self.clean_data_norm_y = torch.tensor(self.qt_y.fit_transform(self.clean_data_y.T).T, dtype=torch.float32)

        # 2) Precompute all possible noisy + transformed data
        self.noisy_data_norm_x = []
        self.noisy_data_norm_y = []
        
        rng = np.random.default_rng(base_seed)
        for i in range(num_files * 2):
            if i % 2 == 0:
                # X plane
                noise_array = rng.normal(
                    loc=0.0, scale=noise_factor, size=self.clean_data_x.shape
                )
                noisy = self.clean_data_x + noise_array
                noisy_norm = self.qt_x.transform(noisy.T).T  # offline transform
                self.noisy_data_norm_x.append(torch.tensor(noisy_norm, dtype=torch.float32))
            else:
                # Y plane
                noise_array = rng.normal(
                    loc=0.0, scale=noise_factor, size=self.clean_data_y.shape
                )
                noisy = self.clean_data_y + noise_array
                noisy_norm = self.qt_y.transform(noisy.T).T
                self.noisy_data_norm_y.append(torch.tensor(noisy_norm, dtype=torch.float32))

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        # Index into the precomputed data
        if idx % 2 == 0:
            plane = "X"
            noisy_norm = self.noisy_data_norm_x[idx // 2]  # or however you map idx
            clean_norm = self.clean_data_norm_x
        else:
            plane = "Y"
            noisy_norm = self.noisy_data_norm_y[idx // 2]
            clean_norm = self.clean_data_norm_y

        return {
            "noisy": noisy_norm, 
            "clean": clean_norm,
            "plane": plane
        }

    def denormalise(self, normalized_output: torch.Tensor, plane: str) -> torch.Tensor:
        """
        Inverts the Quantile Transform.
        """
        if plane == "X":
            qt = self.qt_x
        elif plane == "Y":
            qt = self.qt_y
        else:
            raise ValueError("Plane must be 'X' or 'Y'")
        return torch.tensor(qt.inverse_transform(normalized_output.T).T, dtype=torch.float32)

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
    # Take a sample from the sample list that is X and Y
    for sample in sample_list:
        if sample["plane"] == "X":
            x_sample = sample
        elif sample["plane"] == "Y":
            y_sample = sample

    # Denormalise the samples
    x_noisy = dataset.denormalise(x_sample["noisy"], "X")
    x_clean = dataset.denormalise(x_sample["clean"], "X")
    x_denoised = dataset.denormalise(x_sample["denoised"], "X")

    y_noisy = dataset.denormalise(y_sample["noisy"], "Y")
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
