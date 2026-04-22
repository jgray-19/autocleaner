import re
from pathlib import Path

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
    NOISE_FACTORS,
    NONOISE_INDEX,
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


def _load_twiss_table(model_dir: Path) -> pd.DataFrame:
    return tfs.read(get_twiss_path(model_dir), index="NAME")


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
    model_dat = _load_twiss_table(model_dir)
    sqrt_betax = np.sqrt(model_dat["BETX"].to_numpy())
    sqrt_betay = np.sqrt(model_dat["BETY"].to_numpy())

    x_data = sdds_data.matrices[0].X.to_numpy() / sqrt_betax[:, None]
    y_data = sdds_data.matrices[0].Y.to_numpy() / sqrt_betay[:, None]

    expected_shape = (len(model_dat.index), metadata["nturns"])
    assert x_data.shape == y_data.shape == expected_shape, "Data shape mismatch"

    # Return x_data and y_data as separate tensors
    return (
        torch.tensor(x_data, dtype=torch.float32),
        torch.tensor(y_data, dtype=torch.float32),
        sqrt_betax,
        sqrt_betay,
    )


def write_data(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    noise_index: int | str = 2,
    reference_tbt_path: Path | None = None,
) -> tuple[Path, TbtData]:
    if reference_tbt_path is None:
        model_dir = get_model_dir(beam=BEAM)
        expected_turns = NTURNS
        out_path = get_tbt_path(beam=BEAM, nturns=NTURNS, index=noise_index)
    else:
        metadata = parse_tbt_path_metadata(Path(reference_tbt_path))
        model_dir = get_model_dir(
            beam=metadata["beam"],
            coupling_knob=metadata["coupling_knob"],
            tunes=metadata["tunes"],
        )
        expected_turns = metadata["nturns"]
        out_path = get_tbt_path(
            beam=metadata["beam"],
            nturns=expected_turns,
            coupling_knob=metadata["coupling_knob"],
            tunes=metadata["tunes"],
            kick_amp=metadata["kick_amp"],
            index=noise_index,
        )

    model_dat = _load_twiss_table(model_dir)
    x_bpm_names = model_dat.index.to_list()
    y_bpm_names = model_dat.index.to_list()
    sqrt_betax = np.sqrt(model_dat["BETX"].to_numpy())
    sqrt_betay = np.sqrt(model_dat["BETY"].to_numpy())

    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    assert x_data.shape == (len(x_bpm_names), expected_turns), "Data shape mismatch"
    assert y_data.shape == (len(y_bpm_names), expected_turns), "Data shape mismatch"
    print("Writing datashape with:", x_data.shape, y_data.shape)

    x_data = x_data * sqrt_betax[:, None]
    y_data = y_data * sqrt_betay[:, None]

    matrices = [
        TransverseData(
            X=pd.DataFrame(index=x_bpm_names, data=x_data, dtype=float),
            Y=pd.DataFrame(index=y_bpm_names, data=y_data, dtype=float),
        )
    ]
    out_data = TbtData(matrices=matrices, nturns=expected_turns)
    write_tbt(out_path, out_data)
    return out_path, out_data


class BPMSDataset(Dataset):
    """
    A compact dataset class that:
      - Assumes min-max normalization.
      - Precomputes the normalized clean data.
      - Uses deterministic per-sample specs for source, window offset, and RNG seed.
      - Precomputes a combined noise scaling factor that divides the min-max scale (2/(max-min))
        by the per-BPM beta function, so noise can be injected directly in the normalized space.
    """
    def __init__(self, clean_paths, sample_specs, noise_factors=NOISE_FACTORS):
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

        self.noise_factors = noise_factors
        self.sample_specs = sample_specs

        for spec in self.sample_specs:
            source_idx = spec["source_idx"]
            source_total_turns = self.sources[source_idx]["total_turns"]
            max_start = source_total_turns - NTURNS
            if max_start < 0:
                raise ValueError(
                    f"Source {self.sources[source_idx]['path']} has only"
                    f" {source_total_turns} turns, which is fewer than NTURNS={NTURNS}."
                )
            if not 0 <= spec["offset"] <= max_start:
                raise ValueError(
                    f"Sample offset {spec['offset']} is outside the valid range"
                    f" [0, {max_start}] for {self.sources[source_idx]['path']}."
                )

    def __len__(self):
        return len(self.sample_specs)

    def __getitem__(self, idx):
        spec = self.sample_specs[idx]
        source_idx = spec["source_idx"]
        source = self.sources[source_idx]
        start_idx = spec["offset"]
        end_idx = start_idx + NTURNS

        # Get the precomputed normalized clean data slice.
        clean_slice_norm_x = source["norm_clean_x"][:, :, start_idx:end_idx]
        clean_slice_norm_y = source["norm_clean_y"][:, :, start_idx:end_idx]

        # Determine the noise factor deterministically.
        factor_idx = spec["noise_factor_idx"]
        noise_factor = self.noise_factors[factor_idx]

        rng = np.random.default_rng(spec["rng_seed"])

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


def _build_sample_specs(clean_paths: list[Path], num_samples: int) -> list[dict]:
    rng_offsets = np.random.default_rng(SEED)
    sample_specs = []
    current_offset = 0

    for i in range(num_samples):
        source_idx = i % len(clean_paths)
        metadata = parse_tbt_path_metadata(clean_paths[source_idx])
        max_start = metadata["nturns"] - NTURNS
        if max_start < 0:
            raise ValueError(
                f"Source {clean_paths[source_idx]} has only {metadata['nturns']} turns,"
                f" which is fewer than NTURNS={NTURNS}."
            )
        if USE_OFFSETS and max_start > 0:
            if i % NUM_SAME_OFFSET == 0:
                current_offset = int(rng_offsets.integers(0, max_start + 1))
            offset = current_offset
        else:
            offset = 0

        sample_specs.append(
            {
                "source_idx": source_idx,
                "offset": offset,
                "noise_factor_idx": i % len(NOISE_FACTORS),
                "rng_seed": SEED + i,
            }
        )

    return sample_specs


def _split_sample_specs(
    sample_specs: list[dict], train_ratio: float
) -> tuple[list[dict], list[dict]]:
    grouped_specs = {}
    for spec in sample_specs:
        grouped_specs.setdefault(spec["source_idx"], []).append(spec)

    rng = np.random.default_rng(SEED)
    train_specs = []
    val_specs = []

    for source_idx in sorted(grouped_specs):
        source_specs = grouped_specs[source_idx]
        if len(source_specs) == 1:
            train_specs.extend(source_specs)
            continue

        shuffled_indices = rng.permutation(len(source_specs))
        train_count = int(np.floor(train_ratio * len(source_specs)))
        train_count = max(1, min(len(source_specs) - 1, train_count))

        train_specs.extend(source_specs[i] for i in shuffled_indices[:train_count])
        val_specs.extend(source_specs[i] for i in shuffled_indices[train_count:])

    rng.shuffle(train_specs)
    rng.shuffle(val_specs)
    return train_specs, val_specs


def load_data() -> tuple[DataLoader, DataLoader, BPMSDataset]:
    """Loads the training and validation data."""
    clean_paths = discover_clean_data_paths()
    print(f"Discovered {len(clean_paths)} clean source file(s) for training.")
    for clean_path in clean_paths:
        print(f"  - {clean_path.name}")

    all_sample_specs = _build_sample_specs(clean_paths, max(NUM_FILES, 1))
    if len(clean_paths) == 1:
        print(
            "Warning: Only one clean source file found; training and validation will"
            " split windows from the same underlying orbit."
        )
    train_specs, val_specs = _split_sample_specs(all_sample_specs, TRAIN_RATIO)

    train_dataset = BPMSDataset(
        clean_paths=clean_paths,
        sample_specs=train_specs,
        noise_factors=NOISE_FACTORS,
    )
    val_dataset = BPMSDataset(
        clean_paths=clean_paths,
        sample_specs=val_specs,
        noise_factors=NOISE_FACTORS,
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
