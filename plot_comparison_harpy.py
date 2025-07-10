import numpy as np
from matplotlib import pyplot as plt
from config import PLOT_DIR
from tbt_denoiser import denoise_tbt
from analysis import run_harpy_analysis, process_tbt_data, format_noise

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
# noise_levels = [1e-4, 1e-3] #
plot_dir = PLOT_DIR / "harpy"
plot_dir.mkdir(exist_ok=True)


# RDTs to plot separately
rdts_to_plot = ["f3000_x", "f1011_y"]

rdt_dfs = {}
for i, noise in enumerate(noise_levels):
    noise_dfs = {}
    # Process noisy data
    tbt_file, tbt_file_clean = process_tbt_data(noise)
    tbt_file_zero = process_tbt_data(0.0)
    # Example usage (update the paths as needed):
    autoencoder_path = "conv_autoencoder.pth"  # path to your saved model
    auto_cleaned_file = denoise_tbt(autoencoder_path, tbt_file)
    print("Cleaned file written to:", auto_cleaned_file)

    # Run Harpy analysis for the three cases
    noise_dfs["noisy"], _ = run_harpy_analysis(tbt_file, rdts=rdts, turn_bits=12)
    # noise_dfs["clean"], _ = run_harpy_analysis(tbt_file_clean   , rdts=rdts, clean=True, turn_bits=12)
    noise_dfs["zero"], _ = run_harpy_analysis(tbt_file_zero, rdts=rdts, turn_bits=12)
    noise_dfs["auto"], _ = run_harpy_analysis(
        auto_cleaned_file, rdts=rdts, turn_bits=12
    )
    rdt_dfs[noise] = noise_dfs

plt.rcParams.update({"font.size": 22})
for amp_or_phase in ["AMP", "PHASE"]:
    for rdt_str in rdts_to_plot:
        fig, axs = plt.subplots(1, 4, figsize=(30, 12))  # 1 row, 4 columns
        # fig, axs = plt.subplots(1, 2, figsize=(15, 12))  # 1 row, 2 columns
        fig.suptitle(f"{rdt_str}", fontsize=28, fontweight="bold")
        for i, noise in enumerate(noise_levels):
            omc_calc = np.abs(rdt_dfs[noise]["noisy"][rdt_str][amp_or_phase])
            # omc_calc_clean = np.abs(rdt_dfs[noise]["clean"][rdt_str][amp_or_phase])
            omc_calc_zero = np.abs(rdt_dfs[noise]["zero"][rdt_str][amp_or_phase])
            omc_calc_auto = np.abs(rdt_dfs[noise]["auto"][rdt_str][amp_or_phase])

            diff = (omc_calc - omc_calc_zero) / omc_calc_zero
            # diff_clean = (omc_calc_clean - omc_calc_zero) / omc_calc_zero
            diff_auto = (omc_calc_auto - omc_calc_zero) / omc_calc_zero

            data_dict = {
                "Zero Noise": {
                    "x_data": rdt_dfs[noise]["zero"][rdt_str]["S"],
                    "y_data": omc_calc_zero,
                    "color": COLOURS[1],
                    "linestyle": "solid",
                    "alpha": 1.0,
                },
                # "Cleaned": {
                #     "x_data": rdt_dfs[noise]["clean"][rdt_str]["S"],
                #     "y_data": omc_calc_clean,
                #     "color": COLOURS[2],
                #     "linestyle": "dashed",
                #     "alpha": 0.7,
                #     "avg_err": diff_clean.abs().mean(),
                # },
                "Autoencoder Denoised": {
                    "x_data": rdt_dfs[noise]["auto"][rdt_str]["S"],
                    "y_data": omc_calc_auto,
                    "color": COLOURS[6],
                    "linestyle": "dotted",
                    "alpha": 0.9,
                    "avg_err": diff_auto.abs().mean(),
                },
            }

            axs[i].set_title(f"Noise Level: {format_noise(noise)}")
            for label, props in data_dict.items():
                avg_err_label = ""
                if "avg_err" in props:
                    if label == "Autoencoder Denoised":
                        avg_err_label = f"\n(Avg err: {props['avg_err']:.2%}\nReduction in err: {diff.abs().mean() - props['avg_err']:.2%})"
                    else:
                        avg_err_label = f"\n(Reduction in Avg err: {diff.abs().mean() - props['avg_err']:.2%})"
                axs[i].plot(
                    props["x_data"] / 1e3,
                    props["y_data"],
                    label=f"{label}{avg_err_label}",
                    color=props["color"],
                    linestyle=props["linestyle"],
                    alpha=props["alpha"],
                    marker="x" if label == "Autoencoder Denoised" else None,
                )
            if i == 0:
                if amp_or_phase == "AMP":
                    axs[i].set_ylabel("RDT Amplitude [$m^{-1/2}$]")
                else:
                    axs[i].set_ylabel("RDT Phase")
            if i > 0:
                axs[i].set_yticklabels([])

            if amp_or_phase == "AMP":
                if rdt_str == "f3000_x":
                    axs[i].set_ylim(10, 30)
                elif rdt_str == "f1011_y":
                    axs[i].set_ylim(0, 70)
            axs[i].set_xlabel("s [km]")
            axs[i].legend()

        plt.tight_layout()
        plt.savefig(
            plot_dir / f"{rdt_str}_multi_noise_{amp_or_phase}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

print("Script finished")
