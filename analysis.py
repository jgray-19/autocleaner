import shutil
import tfs
import turn_by_turn as tbt
from config import BEAM, NONOISE_INDEX, NTURNS, PLOT_DIR
from rdt_constants import (
    ANALYSIS_DIR,
    FREQ_OUT_DIR,
)
from rdt_functions import (
    get_rdts_from_optics_analysis,
    get_tbt_path,
    run_harpy,
)

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


def run_harpy_analysis(tbt_file, rdts, clean=False, turn_bits=16):
    """Run Harpy and return both the RDT dataframes and the frequency/amplitude data."""
    print(f"Running Harpy for {tbt_file} (clean={clean})")
    run_harpy(beam=BEAM, tbt_file=tbt_file, clean=clean, turn_bits=turn_bits)
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


def process_tbt_data(noise):
    tbt_path_nonoise = get_tbt_path(beam=BEAM, nturns=NTURNS, index=NONOISE_INDEX)
    if not tbt_path_nonoise.exists():
        raise FileNotFoundError(f"Could not find file {tbt_path_nonoise}")

    if noise == 0.0:
        return tbt_path_nonoise

    tbt_file_noisy = tbt_path_nonoise.name.replace("zero_noise", f"{noise}")
    tbt_path_noisy = tbt_path_nonoise.parent / tbt_file_noisy

    clean_tbt = tbt.read_tbt(tbt_path_nonoise)
    tbt.write_tbt(tbt_path_noisy, clean_tbt, noise=noise)

    # Now get the clean path (by copying the noisy)
    tbt_file_clean = tbt_path_nonoise.name.replace("zero_noise", "harpy_cleaned")
    tbt_path_clean = tbt_path_nonoise.parent / tbt_file_clean

    if tbt_path_clean.exists():
        tbt_path_clean.unlink()
    shutil.copy(tbt_path_noisy, tbt_path_clean)

    return tbt_path_noisy, tbt_path_clean