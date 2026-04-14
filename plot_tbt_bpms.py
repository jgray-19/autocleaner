import argparse
from pathlib import Path

from matplotlib import pyplot as plt
from lhcng.config import PLOT_DIR
from turn_by_turn.lhc import read_tbt

from analysis import process_tbt_data
from config import NBPMS

COLOURS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#56B4E9",
    "#E69F00",
    "#CC79A7",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot turn-by-turn data over all turns for a range of BPMs."
    )
    parser.add_argument(
        "tbt_path",
        nargs="?",
        type=Path,
        help="Path to the input TBT .sdds file. Defaults to the zero-noise file.",
    )
    parser.add_argument(
        "--start-bpm",
        type=int,
        default=110,
        help="First BPM row to plot, using zero-based row indexing.",
    )
    parser.add_argument(
        "--end-bpm",
        type=int,
        default=114,
        help="Last BPM row to plot, using zero-based row indexing.",
    )
    return parser.parse_args()


def _validate_bpm_range(start_bpm: int, end_bpm: int):
    if start_bpm < 0 or end_bpm < 0:
        raise ValueError("BPM indices must be non-negative.")
    if end_bpm < start_bpm:
        raise ValueError("end-bpm must be greater than or equal to start-bpm.")
    if end_bpm >= NBPMS:
        raise ValueError(f"BPM index out of range. Expected < {NBPMS}, got {end_bpm}.")


def _get_default_tbt_path() -> Path:
    return process_tbt_data(0.0)


def _plot_plane(ax, plane_df, start_bpm: int, end_bpm: int, plane_label: str):
    bpm_slice = plane_df.iloc[start_bpm : end_bpm + 1]
    turns = range(plane_df.shape[1])

    for i, (bpm_name, row) in enumerate(bpm_slice.iterrows()):
        ax.plot(
            turns,
            row.to_numpy() * 1e3,
            label=f"{bpm_name} (row {start_bpm + i})",
            color=COLOURS[i % len(COLOURS)],
            linewidth=1.5,
        )

    ax.set_title(f"{plane_label} Plane")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Amplitude [mm]")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()


def main():
    args = parse_args()
    _validate_bpm_range(args.start_bpm, args.end_bpm)

    tbt_path = args.tbt_path if args.tbt_path is not None else _get_default_tbt_path()
    if not tbt_path.exists():
        raise FileNotFoundError(f"Could not find TBT file at {tbt_path.resolve()}.")

    tbt_data = read_tbt(tbt_path)
    matrix = tbt_data.matrices[0]

    fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    _plot_plane(axs[0], matrix.X, args.start_bpm, args.end_bpm, "X")
    _plot_plane(axs[1], matrix.Y, args.start_bpm, args.end_bpm, "Y")

    fig.suptitle(
        f"TBT data for BPM rows {args.start_bpm}-{args.end_bpm}\n{tbt_path.name}",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir = PLOT_DIR / "tbt"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{tbt_path.stem}_bpms_{args.start_bpm}_{args.end_bpm}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
