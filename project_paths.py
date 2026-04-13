from __future__ import annotations

from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
DEFAULT_TUNES = [0.28, 0.31]

DATA_DIR.mkdir(exist_ok=True)


def get_folder_suffix(
    beam: int,
    coupling_knob: bool | float = False,
    tunes: list[float] | tuple[float, float] = DEFAULT_TUNES,
) -> str:
    assert beam in [1, 2], "Beam must be 1 or 2"
    coupling = ""
    if coupling_knob is not False:
        coupling = f"_c{coupling_knob}"
    return f"b{beam}_{coupling}_t{tunes[0]}_{tunes[1]}"


def get_model_dir(
    beam: int,
    coupling_knob: bool | float = False,
    tunes: list[float] | tuple[float, float] = DEFAULT_TUNES,
) -> Path:
    model_dir = CURRENT_DIR / ("model_" + get_folder_suffix(beam, coupling_knob, tunes))
    model_dir.mkdir(exist_ok=True)
    return model_dir


def get_tfs_path(
    beam: int,
    nturns: int,
    coupling_knob: bool | float = False,
    tunes: list[float] | tuple[float, float] = DEFAULT_TUNES,
    kick_amp: float = 1e-3,
) -> Path:
    suffix = get_file_suffix(beam, nturns, coupling_knob, tunes, kick_amp)
    return DATA_DIR / f"{suffix}.tfs.bz2"


def get_tbt_path(
    beam: int,
    nturns: int,
    coupling_knob: bool | float = False,
    tunes: list[float] | tuple[float, float] = DEFAULT_TUNES,
    kick_amp: float = 1e-3,
    index: int | str = 0,
) -> Path:
    suffix = get_file_suffix(beam, nturns, coupling_knob, tunes, kick_amp)
    return DATA_DIR / f"tbt_{suffix}_{index}.sdds"


def get_file_suffix(
    beam: int,
    nturns: int,
    coupling_knob: bool | float = False,
    tunes: list[float] | tuple[float, float] = DEFAULT_TUNES,
    kick_amp: float = 1e-3,
) -> str:
    return get_folder_suffix(beam, coupling_knob, tunes) + f"_t{nturns}_k{kick_amp}"

