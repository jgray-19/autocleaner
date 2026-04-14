import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from lhcng.config import PLOT_DIR
from tbt_denoiser import denoise_tbt
from analysis import run_harpy_analysis, process_tbt_data, format_noise
from config import MODEL_SAVE_PATH

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

noise_levels = [1e-4, 2.5e-4, 5e-4, 1e-3]
plot_dir = PLOT_DIR / "harpy"
plot_dir.mkdir(exist_ok=True, parents=True)

rdts_to_plot = ["f3000_x", "f1011_y"]


def _extract_series(rdt_df, rdt_str, value_key):
    values = np.asarray(rdt_df[rdt_str][value_key])
    if value_key == "AMP":
        values = np.abs(values)
    return values


def _mean_relative_error(values, baseline):
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.divide(
            values - baseline,
            baseline,
            out=np.zeros_like(values, dtype=float),
            where=baseline != 0,
        )
    return np.abs(rel_err).mean()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Harpy outputs for noisy, zero-noise, and denoised TBT data."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default=MODEL_SAVE_PATH,
        help="Path to model weights or a Lightning checkpoint (.pth or .ckpt).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find trained autoencoder weights at {model_path.resolve()}."
        )

    tbt_file_zero = process_tbt_data(0.0)
    rdt_dfs = {}
    for noise in noise_levels:
        noise_dfs = {}
        tbt_file_noisy, _ = process_tbt_data(noise)
        auto_cleaned_file = denoise_tbt(str(model_path), tbt_file_noisy)
        print("Cleaned file written to:", auto_cleaned_file)

        noise_dfs["noisy"], _ = run_harpy_analysis(tbt_file_noisy, rdts=rdts, turn_bits=12)
        noise_dfs["zero"], _ = run_harpy_analysis(tbt_file_zero, rdts=rdts, turn_bits=12)
        noise_dfs["auto"], _ = run_harpy_analysis(auto_cleaned_file, rdts=rdts, turn_bits=12)
        rdt_dfs[noise] = noise_dfs

    plt.rcParams.update({"font.size": 22})
    for amp_or_phase in ["AMP", "PHASE"]:
        for rdt_str in rdts_to_plot:
            fig, axs = plt.subplots(1, len(noise_levels), figsize=(7.5 * len(noise_levels), 12))
            if len(noise_levels) == 1:
                axs = [axs]

            fig.suptitle(f"{rdt_str}", fontsize=28, fontweight="bold")
            for i, noise in enumerate(noise_levels):
                noisy_values = _extract_series(rdt_dfs[noise]["noisy"], rdt_str, amp_or_phase)
                zero_values = _extract_series(rdt_dfs[noise]["zero"], rdt_str, amp_or_phase)
                auto_values = _extract_series(rdt_dfs[noise]["auto"], rdt_str, amp_or_phase)

                noisy_err = _mean_relative_error(noisy_values, zero_values)
                auto_err = _mean_relative_error(auto_values, zero_values)

                data_dict = {
                    "Noisy": {
                        "x_data": rdt_dfs[noise]["noisy"][rdt_str]["S"],
                        "y_data": noisy_values,
                        "color": COLOURS[0],
                        "linestyle": "dashed",
                        "alpha": 0.85,
                        "avg_err": noisy_err,
                    },
                    "Zero Noise": {
                        "x_data": rdt_dfs[noise]["zero"][rdt_str]["S"],
                        "y_data": zero_values,
                        "color": COLOURS[1],
                        "linestyle": "solid",
                        "alpha": 1.0,
                    },
                    "Autoencoder Denoised": {
                        "x_data": rdt_dfs[noise]["auto"][rdt_str]["S"],
                        "y_data": auto_values,
                        "color": COLOURS[6],
                        "linestyle": "dotted",
                        "alpha": 0.9,
                        "avg_err": auto_err,
                        "reduction": noisy_err - auto_err,
                    },
                }

                axs[i].set_title(f"Noise Level: {format_noise(noise)}")
                for label, props in data_dict.items():
                    metric_label = ""
                    if "avg_err" in props:
                        metric_label = f"\n(Mean rel err: {props['avg_err']:.2%})"
                    if "reduction" in props:
                        metric_label = (
                            f"{metric_label}\n(Error reduction: {props['reduction']:.2%})"
                        )

                    axs[i].plot(
                        props["x_data"] / 1e3,
                        props["y_data"],
                        label=f"{label}{metric_label}",
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

            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(
                plot_dir / f"{rdt_str}_multi_noise_{amp_or_phase.lower()}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    print("Script finished")


if __name__ == "__main__":
    main()
