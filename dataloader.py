import numpy as np
import pandas as pd
from pathlib import Path
import tfs
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, Dataset
from turn_by_turn import TbtData, TransverseData
from turn_by_turn.lhc import read_tbt, write_tbt

from config import (
    BATCH_SIZE,
    BEAM,
    DATA_SCALING,
    NBPMS,
    NOISE_FACTORS,
    NTURNS,
    NUM_FILES,
    NUM_SAME_OFFSET,
    NUM_SAME_NOISE,
    SEED,
    TOTAL_TURNS,
    TRAIN_RATIO,
    USE_OFFSETS,
    get_model_dir,
    get_tbt_path,
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


def write_data(data: torch.Tensor, noise_index: int = 2) -> tuple[Path, TbtData]:
    data = data.detach().cpu().numpy()

    model_dat = tfs.read(get_model_dir(beam=BEAM) / "twiss.dat", index="NAME")
    sqrt_betax = np.sqrt(model_dat["BETX"].values)
    sqrt_betay = np.sqrt(model_dat["BETY"].values)
    x_bpm_names = model_dat.index.to_list()
    y_bpm_names = model_dat.index.to_list()

    # assert data.shape == (2, NBPMS, NTURNS), "Data shape mismatch"
    print("Writing datashape with:", data.shape)

    x_data = data[0, :, :] * sqrt_betax[:, None]
    y_data = data[1, :, :] * sqrt_betay[:, None]

    _, _, nturns = data.shape
    out_path = get_tbt_path(beam=BEAM, nturns=nturns, index=noise_index)

    matrices = [
        TransverseData(
            X=pd.DataFrame(index=x_bpm_names, data=x_data, dtype=float),
            Y=pd.DataFrame(index=y_bpm_names, data=y_data, dtype=float),
        )
    ]
    out_data = TbtData(matrices=matrices, nturns=nturns)
    write_tbt(out_path, out_data)
    return out_path, out_data


class BPMSDataset(Dataset):
    """
    A dataset that produces samples with shape (n_planes, nbpms, nturns),
    where n_planes is 2 (X and Y). The normalization for each channel is
    computed over the entire 2D image (flattened) rather than per BPM.
    """

    def __init__(self, num_files, noise_factors=NOISE_FACTORS, base_seed=SEED):
        super().__init__()
        # Load clean data.
        clean_data_x, clean_data_y = load_clean_data()  # shape: (NBPMS, TOTAL_TURNS)
        if NUM_SAME_NOISE > 1 and NUM_SAME_OFFSET > 1:
            raise ValueError("Not sure why you're doing this.")

        # To normalise the noise, so that it is uniform in real space, we need to know the beta functions.
        model_dat = tfs.read(get_model_dir(beam=BEAM) / "twiss.dat", index="NAME")
        sqrt_betax = np.sqrt(model_dat["BETX"].values)
        sqrt_betay = np.sqrt(model_dat["BETY"].values)

        if DATA_SCALING == "qt":
            # Initialize a QuantileTransformer for each channel.
            self.qt_x = QuantileTransformer(output_distribution="normal")
            self.qt_y = QuantileTransformer(output_distribution="normal")

            # For each channel, flatten the entire 2D image and fit the transformer.
            x_flat = clean_data_x.reshape(-1, 1)  # shape: (NBPMS*TOTAL_TURNS, 1)
            x_norm_flat = self.qt_x.fit_transform(x_flat)
            norm_clean_x = x_norm_flat.reshape(NBPMS, TOTAL_TURNS)

            y_flat = clean_data_y.reshape(-1, 1)  # shape: (NBPMS*TOTAL_TURNS, 1)
            y_norm_flat = self.qt_y.fit_transform(y_flat)
            norm_clean_y = y_norm_flat.reshape(NBPMS, TOTAL_TURNS)
        elif DATA_SCALING == "meanstd":
            # Compute per-BPM mean and standard deviation for the clean data.
            # clean_data_x and clean_data_y have shape (NBPMS, TOTAL_TURNS)
            self.mean_x = torch.mean(clean_data_x)  # shape: (NBPMS, 1)
            self.std_x = torch.std(clean_data_x)  # shape: (NBPMS, 1)
            norm_clean_x = (clean_data_x - self.mean_x) / self.std_x

            self.mean_y = torch.mean(clean_data_y)  # shape: (NBPMS, 1)
            self.std_y = torch.std(clean_data_y)  # shape: (NBPMS, 1)
            norm_clean_y = (clean_data_y - self.mean_y) / self.std_y
        elif DATA_SCALING == "minmax":
            # Compute per-BPM minimum and maximum for the clean data.
            self.min_x = torch.min(
                clean_data_x
            )  # - (max(NOISE_FACTORS) * 5 / sqrt_betax.min()) # shape: (NBPMS, 1)
            self.max_x = torch.max(
                clean_data_x
            )  # + (max(NOISE_FACTORS) * 5 / sqrt_betax.min())  # shape: (NBPMS, 1)
            # Scale clean data to [-1, 1]:
            norm_clean_x = (
                2 * (clean_data_x - self.min_x) / (self.max_x - self.min_x) - 1
            )

            self.min_y = torch.min(
                clean_data_y
            )  # - (max(NOISE_FACTORS) * 5 / sqrt_betay.min())
            self.max_y = torch.max(
                clean_data_y
            )  # + (max(NOISE_FACTORS) * 5 / sqrt_betay.min())
            norm_clean_y = (
                2 * (clean_data_y - self.min_y) / (self.max_y - self.min_y) - 1
            )
        else:
            raise ValueError(f"Unknown scaling type: {DATA_SCALING}")

        # Create a full normalized clean tensor.
        self.clean_norm = torch.tensor(
            np.stack([norm_clean_x, norm_clean_y], axis=0), dtype=torch.float32
        )

        # Precompute noisy samples.
        self.noisy_norm = []
        self.offsets = []
        rng = np.random.default_rng(base_seed)

        # If not using offsets always set to 0
        if not USE_OFFSETS:
            offset = 0
        for i in range(num_files):
            # Generate noise for each channel.
            if i % NUM_SAME_NOISE == 0:
                noise_idx = i // NUM_SAME_NOISE
                factor_idx = noise_idx % len(noise_factors)
                # noise_x = rng.normal(loc=0.0, scale=noise_factor, size=clean_data_x.shape)
                # noise_y = rng.normal(loc=0.0, scale=noise_factor, size=clean_data_y.shape)
                noise_x = (
                    noise_factors[factor_idx]
                    * rng.standard_normal(clean_data_x.shape)
                    / sqrt_betax[:, None]
                )
                noise_y = (
                    noise_factors[factor_idx]
                    * rng.standard_normal(clean_data_y.shape)
                    / sqrt_betay[:, None]
                )
            # print("Avg noise", noise_x.flatten().mean(), noise_y.flatten().mean())

            # Create noisy data by adding noise.
            noisy_x = clean_data_x + noise_x
            noisy_y = clean_data_y + noise_y

            # Normalize each channel using the already fitted transformers.
            # Flatten the 2D array, transform, then reshape back.
            if DATA_SCALING == "qt":
                noisy_x_flat = noisy_x.reshape(-1, 1)
                noisy_x_norm_flat = self.qt_x.transform(noisy_x_flat)
                noisy_x_norm = noisy_x_norm_flat.reshape(NBPMS, TOTAL_TURNS)

                noisy_y_flat = noisy_y.reshape(-1, 1)
                noisy_y_norm_flat = self.qt_y.transform(noisy_y_flat)
                noisy_y_norm = noisy_y_norm_flat.reshape(NBPMS, TOTAL_TURNS)
            elif DATA_SCALING == "meanstd":
                # Normalize the noisy data using the same per-BPM mean and std.
                noisy_x_norm = (noisy_x - self.mean_x) / self.std_x
                noisy_y_norm = (noisy_y - self.mean_y) / self.std_y
            elif DATA_SCALING == "minmax":
                noisy_x_norm = (
                    2 * (noisy_x - self.min_x) / (self.max_x - self.min_x) - 1
                )
                noisy_y_norm = (
                    2 * (noisy_y - self.min_y) / (self.max_y - self.min_y) - 1
                )
            else:
                raise ValueError(f"Unknown scaling type: {DATA_SCALING}")

            # Stack to create a (2, NBPMS, TOTAL_TURNS) tensor.
            sample = torch.tensor(
                np.stack([noisy_x_norm, noisy_y_norm], axis=0), dtype=torch.float32
            )
            self.noisy_norm.append(sample)

            # Update the offset every NUM_SAME_OFFSET
            if USE_OFFSETS and i % NUM_SAME_OFFSET == 0:
                offset = rng.integers(0, TOTAL_TURNS - NTURNS + 1)

            # Store the offset
            self.offsets.append(offset)

        self.num_files = num_files

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        # Use the stored offset for this sample
        start_idx = self.offsets[idx]
        end_idx = start_idx + NTURNS

        noisy_sample = self.noisy_norm[idx][:, :, start_idx:end_idx]
        clean_sample = self.clean_norm[:, :, start_idx:end_idx]
        return {"noisy": noisy_sample, "clean": clean_sample}

    def get_full_noisy(self, idx):
        return self.noisy_norm[idx]

    def denormalise(self, normalized_output: torch.Tensor) -> torch.Tensor:
        """
        Inverse transforms a normalized output (of shape (2, NBPMS, NTURNS))
        back to the original scale, treating each channel as a whole 2D image.
        """
        if DATA_SCALING == "qt":
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

            # Convert back to tensor.
            orig_x = torch.tensor(orig_x, dtype=torch.float32)
            orig_y = torch.tensor(orig_y, dtype=torch.float32)

        elif DATA_SCALING == "meanstd":
            # For channel X:
            norm_x = normalized_output[0].detach().cpu()
            orig_x = norm_x * self.std_x + self.mean_x  # Broadcasting over turns

            # For channel Y:
            norm_y = normalized_output[1].detach().cpu()
            orig_y = norm_y * self.std_y + self.mean_y  # Broadcasting over turns

        elif DATA_SCALING == "minmax":
            # For channel X:
            norm_x = normalized_output[0].detach().cpu()
            orig_x = (norm_x + 1) / 2 * (self.max_x - self.min_x) + self.min_x

            # For channel Y:
            norm_y = normalized_output[1].detach().cpu()
            orig_y = (norm_y + 1) / 2 * (self.max_y - self.min_y) + self.min_y

        else:
            raise ValueError(f"Unknown scaling type: {DATA_SCALING}")

        # Combine channels back together.
        orig = torch.stack([orig_x, orig_y], dim=0)  # shape: (2, NBPMS, NTURNS)
        return orig


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
    denorm_full_sample = dataset.denormalise(sample["noisy_full"])
    denorm_clean = dataset.denormalise(sample["clean"])
    denorm_denoised = dataset.denormalise(sample["denoised"])

    return denorm_sample, denorm_full_sample, denorm_clean, denorm_denoised
