import shutil
import tfs
import turn_by_turn as tbt
from config import NONOISE_INDEX, TOTAL_TURNS, SAMPLE_INDEX, HARPY_CLEAN_INDEX
from lhcng.config import (
    PLOT_DIR,
    ANALYSIS_DIR,
    FREQ_OUT_DIR,
)
from lhcng.analysis import get_rdts_from_optics_analysis, run_harpy
from lhcng.tracking import get_tbt_path
from lhcng.model import get_model_dir

def format_noise(noise):
    if noise >= 1e-3:
        return f"{noise * 1e3:.1f} mm"
    elif noise >= 1e-6:
        return f"{noise * 1e6:.1f} µm"
    else:
        return f"{noise * 1e9:.1f} nm"


def rdt_plots_dir(noise):
    d = PLOT_DIR / f"{noise:1.1e}_{TOTAL_TURNS}t_rdt"
    d.mkdir(exist_ok=True)
    return d


def run_harpy_analysis(
    beam, coupling_knob, tunes, kick_amp, index, rdts, clean=False, turn_bits=16
):
    """Run Harpy and return both the RDT dataframes and the frequency/amplitude data."""
    tbt_file = get_tbt_path(beam, TOTAL_TURNS, coupling_knob, kick_amp, tunes, index=index)
    model_dir = get_model_dir(beam, coupling_knob, tunes)

    print(f"Running Harpy for {tbt_file} (clean={clean})")
    run_harpy(
        beam=beam,
        model_dir=model_dir,
        tbt_file=tbt_file,
        tunes=tunes,
        clean=clean,
        turn_bits=turn_bits,
    )
    analysis_folder = ANALYSIS_DIR / tbt_file.stem
    analysis_folder.mkdir(exist_ok=True)

    rdts_df = get_rdts_from_optics_analysis(
        beam=beam,
        tbt_path=FREQ_OUT_DIR / tbt_file.name,
        model_dir=model_dir,
    )
    # Load frequency/amplitude data for both X and Y planes
    freqx = tfs.read(FREQ_OUT_DIR / f"{tbt_file.name}.freqsx")
    ampsx = tfs.read(FREQ_OUT_DIR / f"{tbt_file.name}.ampsx")
    freqy = tfs.read(FREQ_OUT_DIR / f"{tbt_file.name}.freqsy")
    ampsy = tfs.read(FREQ_OUT_DIR / f"{tbt_file.name}.ampsy")
    freq_amp = {"freqx": freqx, "ampsx": ampsx, "freqy": freqy, "ampsy": ampsy}
    return rdts_df, freq_amp


def process_tbt_data(beam, coupling_knob, tunes, kick_amp, noise):
    assert beam in [1, 2], "Beam not 1 or 2"

    tbt_path_nonoise = get_tbt_path(
        beam=beam,
        nturns=TOTAL_TURNS,
        coupling_knob=coupling_knob,
        tunes=tunes,
        kick_amp=kick_amp,
        index=NONOISE_INDEX,
    )
    if not tbt_path_nonoise.exists():
        raise FileNotFoundError(f"Could not find file {tbt_path_nonoise}")

    if noise == 0.0:
        return tbt_path_nonoise

    tbt_path_noisy = get_tbt_path(
        beam=beam,
        nturns=TOTAL_TURNS,
        coupling_knob=coupling_knob,
        tunes=tunes,
        kick_amp=kick_amp,
        index=SAMPLE_INDEX,
    )

    clean_tbt = tbt.read_tbt(tbt_path_nonoise)
    tbt.write_tbt(tbt_path_noisy, clean_tbt, noise=noise)
    print(f"Written tbt file to {tbt_path_noisy}")

    # Now get the clean path (by copying the noisy)
    tbt_path_clean = get_tbt_path(
        beam=beam,
        nturns=TOTAL_TURNS,
        coupling_knob=coupling_knob,
        tunes=tunes,
        kick_amp=kick_amp,
        index=HARPY_CLEAN_INDEX,
    )

    if tbt_path_clean.exists():
        tbt_path_clean.unlink()
    shutil.copy(tbt_path_noisy, tbt_path_clean)
    print(f"Written tbt file to {tbt_path_clean}")

    return tbt_path_noisy, tbt_path_clean
