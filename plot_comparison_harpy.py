import shutil
from pathlib import Path

import numpy as np
import turn_by_turn as tbt
from matplotlib import pyplot as plt
from turn_by_turn import madng
import tfs
from config import NTURNS, BEAM, DENOISED_INDEX

from rdt_constants import (
    ANALYSIS_DIR,
    DATA_DIR,
    FREQ_OUT_DIR,
)
from rdt_functions import (
    get_rdts_from_optics_analysis,
    run_tracking,
    get_tbt_path,
    run_harpy,
)

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
TFS_PATH = get_tbt_path(beam=BEAM, nturns=NTURNS, index=DENOISED_INDEX)

def format_noise(noise):
    if noise >= 1e-3:
        return f"{noise * 1e3:.1f} mm"
    elif noise >= 1e-6:
        return f"{noise * 1e6:.1f} µm"
    else:
        return f"{noise * 1e9:.1f} nm"

def rdt_plots_dir(noise):
    d = PLOTS_DIR / f"{noise:1.1e}_{NTURNS}t_rdt"
    d.mkdir(exist_ok=True)
    return d

def run_harpy_analysis(tbt_file, rdts, clean=False):
    """Run Harpy and return both the RDT dataframes and the frequency/amplitude data."""
    print(f"Running Harpy for {tbt_file} (clean={clean})")
    run_harpy(files=[tbt_file], output_dir=FREQ_OUT_DIR, beam=BEAM, clean=clean)
    analysis_folder = ANALYSIS_DIR / tbt_file.name
    analysis_folder.mkdir(exist_ok=True)

    rdts_df = get_rdts_from_optics_analysis(
        files=[FREQ_OUT_DIR / tbt_file.name],
        output_dir=analysis_folder,
        rdts=rdts,
        beam=BEAM,
    )
    # Load frequency/amplitude data for both X and Y planes
    freqx = tfs.read(FREQ_OUT_DIR / f"{tbt_file.name}.freqsx")
    ampsx = tfs.read(FREQ_OUT_DIR / f"{tbt_file.name}.ampsx")
    freqy = tfs.read(FREQ_OUT_DIR / f"{tbt_file.name}.freqsy")
    ampsy = tfs.read(FREQ_OUT_DIR / f"{tbt_file.name}.ampsy")
    freq_amp = {"freqx": freqx, "ampsx": ampsx, "freqy": freqy, "ampsy": ampsy}
    return rdts_df, freq_amp

def process_tbt_data(noise, tfs_path):
    tbt_name = f"tbt_data_{noise:1.1e}_{NTURNS}t_b{BEAM}.sdds"
    tbt_file = DATA_DIR / tbt_name

    if not tbt_file.exists():
        print(
            f"Tracking or Harpy is enabled, retrieving TBT data for noise level {noise}"
        )
        if tfs_path.exists():
            print("Reading TFS file")
            tbt_data = madng.read_tbt(tfs_path)
        else:
            print("Running MAD-NG Tracking")
            run_tracking(BEAM, NTURNS, tfs_path, kick_amp=1e-4)  # 2 mm
            tbt_data = madng.read_tbt(tfs_path)
        tbt.write(tbt_file, tbt_data, noise=noise)
        del tbt_data
    if noise == 0.0:
        return tbt_file
    else:
        clean_name = tbt_name.replace("tbt_data", "tbt_data_clean")
        tbt_file_clean = DATA_DIR / clean_name
        if tbt_file_clean.exists() or tbt_file_clean.is_symlink():
            tbt_file_clean.unlink()
        # tbt_file_clean.symlink_to(tbt_file.name)
        shutil.copy(tbt_file, tbt_file_clean)
        return tbt_file, tbt_file_clean


COLOURS = [
    "#0072B2",  # Blue
    "#D55E00",  # Red
    "#009E73",  # Green
    "#56B4E9",  # Sky Blue
    "#E69F00",  # Orange
    "#F0E442",  # Yellow
    "#CC79A7",  # Pink
    "#000000",  # Black
]
plt.rcParams.update({"axes.prop_cycle": plt.cycler(color=COLOURS)})

# Get the RDT strings and processing function
rdts = [  # Normal Sextupole
    "f1200_x",
    "f3000_x",
    "f1002_x",
    "f1020_x",
    "f0111_y",
    "f0120_y",
    "f1011_y",
    "f1020_y",
]

# Define noise levels
noise_levels = [1e-4, 2.5e-4, 5e-4, 1e-3]

# RDTs to plot separately
rdts_to_plot = ["f3000_x", "f1011_y"]

for rdt_str in rdts_to_plot:
    plt.rcParams.update({"font.size": 22})
    fig, axs = plt.subplots(1, 4, figsize=(30, 12))  # 1 row, 4 columns
    fig.suptitle(f"{rdt_str}", fontsize=28, fontweight="bold")
    for i, noise in enumerate(noise_levels):
        # Process noisy data
        tbt_file, tbt_file_clean = process_tbt_data(noise, TFS_PATH)
        tbt_file_zero = process_tbt_data(0.0, TFS_PATH)
        # Example usage (update the paths as needed):
        autoencoder_path = "conv_autoencoder.pth"  # path to your saved model
        auto_cleaned_file = denoise_tbt(autoencoder_path, tbt_file)
        print("Cleaned file written to:", auto_cleaned_file)

        # Run Harpy analysis for the three cases
        rdt_dfs, _ = run_harpy_analysis(tbt_file, rdts=rdts)
        rdt_dfs_clean, _ = run_harpy_analysis(tbt_file_clean, rdts=rdts, clean=True)
        rdt_dfs_zero, _ = run_harpy_analysis(tbt_file_zero, rdts=rdts)
        rdt_dfs_auto, _ = run_harpy_analysis(auto_cleaned_file, rdts=rdts)

        omc_calc = np.abs(rdt_dfs[rdt_str]["AMP"])
        omc_calc_clean = np.abs(rdt_dfs_clean[rdt_str]["AMP"])
        omc_calc_zero = np.abs(rdt_dfs_zero[rdt_str]["AMP"])
        omc_calc_auto = np.abs(rdt_dfs_auto[rdt_str]["AMP"])

        diff = (omc_calc - omc_calc_zero) / omc_calc_zero
        diff_clean = (omc_calc_clean - omc_calc_zero) / omc_calc_zero
        diff_auto = (omc_calc_auto - omc_calc_zero) / omc_calc_zero

        data_dict = {
            "Zero Noise": {
                "x_data": rdt_dfs_zero[rdt_str]["S"],
                "y_data": omc_calc_zero,
                "color": COLOURS[1],
                "linestyle": "solid",
                "alpha": 1.0,
            },
            "Cleaned": {
                "x_data": rdt_dfs_clean[rdt_str]["S"],
                "y_data": omc_calc_clean,
                "color": COLOURS[2],
                "linestyle": "dashed",
                "alpha": 0.8,
                "avg_err": diff_clean.abs().mean(),
            },
            "Autoencoder Denoised": {
                "x_data": rdt_dfs_auto[rdt_str]["S"],
                "y_data": omc_calc_auto,
                "color": COLOURS[6],
                "linestyle": "dotted",
                "alpha": 0.9,
                "avg_err": diff_auto.abs().mean(),
            },
        }

        axs[i].set_title(f"Noise Level: {format_noise(noise)}")
        for label, props in data_dict.items():
            avg_err_label = (
                f"\n(Avg err reduction: {diff.abs().mean() - props['avg_err']:.2%})"
                if "avg_err" in props
                else ""
            )
            axs[i].plot(
                props["x_data"] / 1e3,
                props["y_data"],
                label=f"{label}{avg_err_label}",
                color=props["color"],
                linestyle=props["linestyle"],
                alpha=props["alpha"],
                marker="x" if label == "Cleaned" else None,
            )
        if i == 0:
            axs[i].set_ylabel("RDT Amplitude [$m^{-1/2}$]")
        if i > 0:
            axs[i].set_yticklabels([])

        if rdt_str == "f3000_x":
            axs[i].set_ylim(10, 30)
        elif rdt_str == "f1011_y":
            axs[i].set_ylim(0, 70)
        axs[i].set_xlabel("s [km]")
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(
        rdt_plots_dir(noise) / f"{rdt_str}_multi_noise.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

print("Script finished")
