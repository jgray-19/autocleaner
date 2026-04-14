import shutil
from pathlib import Path

import tfs
import turn_by_turn as tbt
from config import BEAM, HARPY_INPUT, NONOISE_INDEX, NTURNS
from dataloader import parse_tbt_path_metadata
from lhcng.config import (
    PLOT_DIR,
    ANALYSIS_DIR,
    FREQ_OUT_DIR,
)
from lhcng.analysis import get_rdts_from_optics_analysis, run_harpy
from project_paths import DATA_DIR, get_model_dir

def format_noise(noise):
    if noise >= 1e-3:
        return f"{noise * 1e3:.1f} mm"
    elif noise >= 1e-6:
        return f"{noise * 1e6:.1f} µm"
    else:
        return f"{noise * 1e9:.1f} nm"


def rdt_plots_dir(noise):
    d = PLOT_DIR / f"{noise:1.1e}_{NTURNS}t_rdt"
    d.mkdir(exist_ok=True)
    return d


def _run_harpy_compat(tbt_file: Path, clean: bool, turn_bits: int) -> None:
    metadata = parse_tbt_path_metadata(tbt_file)
    model_dir = get_model_dir(
        beam=metadata["beam"],
        coupling_knob=metadata["coupling_knob"],
        tunes=metadata["tunes"],
    )

    compatibility_calls = [
        lambda: run_harpy(beam=BEAM, tbt_file=tbt_file, clean=clean, turn_bits=turn_bits),
        lambda: run_harpy(BEAM, tbt_file, clean=clean, turn_bits=turn_bits),
        lambda: run_harpy(tbt_file, beam=BEAM, clean=clean, turn_bits=turn_bits),
        lambda: run_harpy(
            tbt_file,
            beam=BEAM,
            model_dir=model_dir,
            tunes=HARPY_INPUT.tunes,
            natdeltas=HARPY_INPUT.natdeltas,
            clean=clean,
            turn_bits=turn_bits,
        ),
        lambda: run_harpy(
            beam=BEAM,
            tbt_path=tbt_file,
            model_dir=model_dir,
            tunes=HARPY_INPUT.tunes,
            natdeltas=HARPY_INPUT.natdeltas,
            clean=clean,
            turn_bits=turn_bits,
        ),
    ]

    last_error = None
    for call in compatibility_calls:
        try:
            call()
            return
        except TypeError as error:
            last_error = error

    raise TypeError(
        f"Could not find a compatible run_harpy signature for {tbt_file}. "
        f"Last error: {last_error}"
    )


def run_harpy_analysis(tbt_file, rdts, clean=False, turn_bits=16):
    """Run Harpy and return both the RDT dataframes and the frequency/amplitude data."""
    print(f"Running Harpy for {tbt_file} (clean={clean})")
    _run_harpy_compat(tbt_file=tbt_file, clean=clean, turn_bits=turn_bits)
    analysis_folder = ANALYSIS_DIR / tbt_file.stem
    analysis_folder.mkdir(exist_ok=True)

    rdts_df = get_rdts_from_optics_analysis(
        beam=BEAM,
        tbt_path=FREQ_OUT_DIR / tbt_file.name,
    )
    # Load frequency/amplitude data for both X and Y planes
    freqx = tfs.read(FREQ_OUT_DIR / f"{tbt_file.name}.freqsx")
    ampsx = tfs.read(FREQ_OUT_DIR / f"{tbt_file.name}.ampsx")
    freqy = tfs.read(FREQ_OUT_DIR / f"{tbt_file.name}.freqsy")
    ampsy = tfs.read(FREQ_OUT_DIR / f"{tbt_file.name}.ampsy")
    freq_amp = {"freqx": freqx, "ampsx": ampsx, "freqy": freqy, "ampsy": ampsy}
    return rdts_df, freq_amp


def _find_clean_tbt_path(nturns: int = NTURNS) -> Path:
    pattern = f"tbt_b{BEAM}__*_t*_k*_{NONOISE_INDEX}.sdds"
    matching_paths = []
    for candidate in sorted(DATA_DIR.glob(pattern)):
        try:
            metadata = parse_tbt_path_metadata(candidate)
        except ValueError:
            continue
        if metadata["beam"] == BEAM and metadata["nturns"] == nturns:
            matching_paths.append(candidate)

    if not matching_paths:
        raise FileNotFoundError(
            f"Could not find a clean TBT file for beam {BEAM} with {nturns} turns in {DATA_DIR}."
        )
    return matching_paths[0]


def process_tbt_data(noise):
    tbt_path_nonoise = _find_clean_tbt_path()

    if noise == 0.0:
        return tbt_path_nonoise

    noise_tag = f"{noise:.1e}".replace("+", "")
    tbt_file_noisy = tbt_path_nonoise.name.replace("zero_noise", f"noisy_{noise_tag}")
    tbt_path_noisy = tbt_path_nonoise.parent / tbt_file_noisy

    clean_tbt = tbt.read_tbt(tbt_path_nonoise)
    tbt.write_tbt(tbt_path_noisy, clean_tbt, noise=noise)
    print(f"Written tbt file to {tbt_path_noisy}")

    # Now get the clean path (by copying the noisy)
    tbt_file_clean = tbt_path_nonoise.name.replace("zero_noise", f"harpy_cleaned_{noise_tag}")
    tbt_path_clean = tbt_path_nonoise.parent / tbt_file_clean

    if tbt_path_clean.exists():
        tbt_path_clean.unlink()
    shutil.copy(tbt_path_noisy, tbt_path_clean)
    print(f"Written tbt file to {tbt_path_clean}")

    return tbt_path_noisy, tbt_path_clean
