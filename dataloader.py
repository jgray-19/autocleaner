from pathlib import Path
import re

import numpy as np
import pandas as pd
import tfs
import torch
from lhcng.config import DATA_DIR
from lhcng.model import get_model_dir
from lhcng.tracking import get_tbt_path
from torch.utils.data import DataLoader, Dataset
from turn_by_turn import TbtData, TransverseData
from turn_by_turn.lhc import read_tbt, write_tbt

from config import (
    BATCH_SIZE,
    BEAM,
    NBPMS,
    NONOISE_INDEX,
    NOISE_FACTORS,
    NTURNS,
    NUM_FILES,
    NUM_SAME_OFFSET,
    SEED,
    TOTAL_TURNS,
    TRAIN_RATIO,
    USE_OFFSETS,
)


TBT_FILENAME_PATTERN = re.compile(
    r"^tbt_b(?P<beam>\d+)__(?:(?:c(?P<coupling>[^_]+))_)?"
    r"t(?P<tune_x>[-+0-9.eE]+)_(?P<tune_y>[-+0-9.eE]+)"
    r"_t(?P<nturns>\d+)_k(?P<kick_amp>[-+0-9.eE]+)_(?P<index>.+)\.sdds$"
)


def _parse_coupling_knob(raw_value: str | None) -> bool | float:
    if raw_value is None:
        return False
    if raw_value == "True":
        return True
    if raw_value == "False":
        return False
    return float(raw_value)


def get_twiss_path(model_dir: Path) -> Path:
    for filename in ("twiss.dat", "twiss.tfs.bz2"):
        twiss_path = model_dir / filename
        if twiss_path.exists():
            return twiss_path
    raise FileNotFoundError(f"Could not find a twiss file in {model_dir}")


def parse_tbt_path_metadata(tbt_path: Path) -> dict:
    match = TBT_FILENAME_PATTERN.match(tbt_path.name)
    if match is None:
        raise ValueError(f"Unrecognized TBT filename format: {tbt_path.name}")

    return {
        "beam": int(match.group("beam")),
        "coupling_knob": _parse_coupling_knob(match.group("coupling")),
        "tunes": [
            float(match.group("tune_x")),
            float(match.group("tune_y")),
        ],
        "nturns": int(match.group("nturns")),
        "kick_amp": float(match.group("kick_amp")),
        "index": match.group("index"),
    }


def discover_clean_data_paths() -> list[Path]:
    exact_match_paths = []
    fallback_paths = []
    pattern = f"tbt_b{BEAM}__*_t*_k*_{NONOISE_INDEX}.sdds"
    for clean_path in sorted(DATA_DIR.glob(pattern)):
        try:
            metadata = parse_tbt_path_metadata(clean_path)
        except ValueError:
            continue
        if metadata["beam"] != BEAM or metadata["nturns"] < NTURNS:
            continue
        if metadata["nturns"] == TOTAL_TURNS:
            exact_match_paths.append(clean_path)
        else:
            fallback_paths.append(clean_path)
    if exact_match_paths:
        return exact_match_paths
    if fallback_paths:
        print(
            "Warning: No clean sources matched TOTAL_TURNS exactly; falling back to"
            f" {len(fallback_paths)} file(s) with at least {NTURNS} turns."
        )
        return fallback_paths
    raise FileNotFoundError(
        f"Could not find any clean TBT files matching {pattern} in {DATA_DIR}"
    )


def load_clean_data(
    sdds_data_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    sdds_data = read_tbt(sdds_data_path)
    metadata = parse_tbt_path_metadata(sdds_data_path)

    # Read twiss file for beta functions
    model_dir = get_model_dir(
        beam=metadata["beam"],
        coupling_knob=metadata["coupling_knob"],
        tunes=metadata["tunes"],
    )
    model_dat = tfs.read(
        get_twiss_path(model_dir)
    )
    sqrt_betax = np.sqrt(model_dat["BETX"].values)  # For X plane
    sqrt_betay = np.sqrt(model_dat["BETY"].values)  # For Y plane

    # Extract data; assume sdds_data.matrices[0] contains both 'X' and 'Y'
    x_data = sdds_data.matrices[0].X.to_numpy() / sqrt_betax[:, None]
    y_data = sdds_data.matrices[0].Y.to_numpy() / sqrt_betay[:, None]

    expected_shape = (NBPMS, metadata["nturns"])
    assert x_data.shape == y_data.shape == expected_shape, "Data shape mismatch"

    # Return x_data and y_data as separate tensors
    return (
        torch.tensor(x_data, dtype=torch.float32),
        torch.tensor(y_data, dtype=torch.float32),
        sqrt_betax,
        sqrt_betay,
    )


def write_data(x_data: torch.Tensor, y_data: torch.Tensor, noise_index: int = 2) -> tuple[Path, TbtData]:
    model_dat = tfs.read(get_twiss_path(get_model_dir(beam=BEAM)), index="NAME")
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
    def __init__(self, clean_paths, num_samples, noise_factors=NOISE_FACTORS, base_seed=SEED):
        super().__init__()
        if not clean_paths:
            raise ValueError("BPMSDataset requires at least one clean TBT file.")

        self.sources = []
        for clean_path in clean_paths:
            metadata = parse_tbt_path_metadata(clean_path)
            clean_x, clean_y, sqrt_betax, sqrt_betay = load_clean_data(clean_path)
            sqrt_betax_tensor = torch.tensor(sqrt_betax, dtype=torch.float32)
            sqrt_betay_tensor = torch.tensor(sqrt_betay, dtype=torch.float32)
            min_x = torch.min(clean_x)
            max_x = torch.max(clean_x)
            min_y = torch.min(clean_y)
            max_y = torch.max(clean_y)

            norm_clean_x = 2 * (clean_x - min_x) / (max_x - min_x) - 1
            norm_clean_y = 2 * (clean_y - min_y) / (max_y - min_y) - 1
            min_max_scale_x = 2 / (max_x - min_x)
            min_max_scale_y = 2 / (max_y - min_y)

            self.sources.append(
                {
                    "path": clean_path,
                    "total_turns": metadata["nturns"],
                    "sqrt_betax": sqrt_betax,
                    "sqrt_betay": sqrt_betay,
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y,
                    "norm_clean_x": norm_clean_x.unsqueeze(0),
                    "norm_clean_y": norm_clean_y.unsqueeze(0),
                    "noise_scale_x": min_max_scale_x / sqrt_betax_tensor,
                    "noise_scale_y": min_max_scale_y / sqrt_betay_tensor,
                }
            )

        self.num_samples = num_samples
        self.noise_factors = noise_factors
        self.base_seed = base_seed
        self.sample_source_indices = [i % len(self.sources) for i in range(num_samples)]

        # Precompute per-sample offsets using the source-specific turn count.
        self.offsets = []
        rng_offsets = np.random.default_rng(base_seed)
        for i, source_idx in enumerate(self.sample_source_indices):
            source_total_turns = self.sources[source_idx]["total_turns"]
            max_start = source_total_turns - NTURNS
            if max_start < 0:
                raise ValueError(
                    f"Source {self.sources[source_idx]['path']} has only"
                    f" {source_total_turns} turns, which is fewer than NTURNS={NTURNS}."
                )
            if USE_OFFSETS and max_start > 0:
                if i % NUM_SAME_OFFSET == 0:
                    offset = int(rng_offsets.integers(0, max_start + 1))
                self.offsets.append(offset)
            else:
                self.offsets.append(0)

        # Pre-store an RNG for each sample.
        self.rngs = [np.random.default_rng(base_seed + i) for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        source_idx = self.sample_source_indices[idx]
        source = self.sources[source_idx]
        start_idx = self.offsets[idx]
        end_idx = start_idx + NTURNS

        # Get the precomputed normalized clean data slice.
        clean_slice_norm_x = source["norm_clean_x"][:, :, start_idx:end_idx]
        clean_slice_norm_y = source["norm_clean_y"][:, :, start_idx:end_idx]

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
        noise_x = torch.tensor(noise_x, dtype=torch.float32) * source["noise_scale_x"][:, None]
        noise_y = torch.tensor(noise_y, dtype=torch.float32) * source["noise_scale_y"][:, None]

        # The noisy normalized data is simply the clean normalized slice plus the scaled noise.
        noisy_norm_x = clean_slice_norm_x + noise_x.unsqueeze(0)
        noisy_norm_y = clean_slice_norm_y + noise_y.unsqueeze(0)

        return {
            "noisy_x": noisy_norm_x,  # shape: (1, NBPMS, NTURNS)
            "noisy_y": noisy_norm_y,
            "clean_x": clean_slice_norm_x,
            "clean_y": clean_slice_norm_y,
            "source_idx": torch.tensor(source_idx, dtype=torch.long),
        }

    def denormalise(self, norm_data: np.ndarray, plane: str, source_idx: int = 0) -> np.ndarray:
        """
        Inverse transforms a normalized output for a specific plane ('x' or 'y')
        back to the original scale, treating the channel as a whole 2D image.
        """
        if plane not in ["x", "y"]:
            raise ValueError("Plane must be 'x' or 'y'.")

        source = self.sources[source_idx]
        min_val = source[f"min_{plane}"].numpy()
        max_val = source[f"max_{plane}"].numpy()
        betas = source[f"sqrt_beta{plane}"]
        orig_data = (norm_data + 1) / 2 * (max_val - min_val) + min_val
        return orig_data * betas[:, None]


def load_data() -> tuple[DataLoader, DataLoader, BPMSDataset]:
    """Loads the training and validation data."""
    clean_paths = discover_clean_data_paths()
    print(f"Discovered {len(clean_paths)} clean source file(s) for training.")
    for clean_path in clean_paths:
        print(f"  - {clean_path.name}")

    train_num_samples = int(TRAIN_RATIO * NUM_FILES)
    val_num_samples = NUM_FILES - train_num_samples

    if len(clean_paths) == 1:
        print(
            "Warning: Only one clean source file found; validation will reuse the same"
            " underlying orbit."
        )
        train_paths = clean_paths
        val_paths = clean_paths
    else:
        rng = np.random.default_rng(SEED)
        shuffled_indices = rng.permutation(len(clean_paths))
        train_source_count = int(np.floor(TRAIN_RATIO * len(clean_paths)))
        train_source_count = max(1, min(len(clean_paths) - 1, train_source_count))
        train_paths = [clean_paths[i] for i in shuffled_indices[:train_source_count]]
        val_paths = [clean_paths[i] for i in shuffled_indices[train_source_count:]]

    train_dataset = BPMSDataset(
        clean_paths=train_paths,
        num_samples=max(train_num_samples, 1),
        noise_factors=NOISE_FACTORS,
        base_seed=SEED,
    )
    val_dataset = BPMSDataset(
        clean_paths=val_paths,
        num_samples=max(val_num_samples, 1),
        noise_factors=NOISE_FACTORS,
        base_seed=SEED + 10_000,
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
    return train_loader, val_loader, val_dataset


def build_sample_dict(sample: dict[str, np.ndarray], dataset: BPMSDataset) -> dict:
    """
    Given a list of batches, returns a dictionary with the X and Y samples,
    both in noisy and clean versions after inversion, for each plane.
    """
    sample_dict = {}
    source_idx = int(sample["source_idx"])
    for plane in ["x", "y"]:
        sample_dict[f"noisy_{plane}"] = dataset.denormalise(
            sample[f"noisy_{plane}"], plane=plane, source_idx=source_idx
        )
        sample_dict[f"clean_{plane}"] = dataset.denormalise(
            sample[f"clean_{plane}"], plane=plane, source_idx=source_idx
        )
        sample_dict[f"recon_{plane}"] = dataset.denormalise(
            sample[f"recon_{plane}"], plane=plane, source_idx=source_idx
        )

    return sample_dict
