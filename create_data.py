from __future__ import annotations

import random
from math import pi
from pathlib import Path

import tfs
import turn_by_turn as tbt
from turn_by_turn import convert_to_tbt
from xtrack_tools import create_xsuite_environment, line_to_dataframes, run_tracking_without_ac_dipole

from config import BEAM, NONOISE_INDEX, TOTAL_TURNS
from project_paths import DEFAULT_TUNES, get_model_dir, get_tbt_path, get_tfs_path

NTURNS = TOTAL_TURNS
# Reuse the smaller kick that already produced the working 1000-turn clean file.
ACTION = 4e-7
TUNES = DEFAULT_TUNES

TURN_ANGLES = {
    1000: 0.0,
    3000: pi / 4,
}

random.seed(42)


def get_sequence_path(model_dir: Path, beam: int) -> Path:
    sequence_path = model_dir / f"lhcb{beam}_saved.seq"
    if not sequence_path.exists():
        raise FileNotFoundError(
            f"Could not find saved sequence for beam {beam} at {sequence_path}."
        )
    return sequence_path


def get_tracking_angle(nturns: int) -> float:
    return TURN_ANGLES.get(nturns, 0.0)


def main() -> None:
    angle = get_tracking_angle(NTURNS)
    model_dir = get_model_dir(BEAM, tunes=TUNES)
    sequence_path = get_sequence_path(model_dir, BEAM)
    tfs_path = get_tfs_path(BEAM, NTURNS, tunes=TUNES, kick_amp=ACTION)
    tbt_path = get_tbt_path(
        beam=BEAM,
        nturns=NTURNS,
        tunes=TUNES,
        kick_amp=ACTION,
        index=NONOISE_INDEX,
    )
    json_path = model_dir / f"lhcb{BEAM}.json"

    print(f"Loading xsuite environment from {sequence_path}")
    env = create_xsuite_environment(
        sequence_file=sequence_path,
        seq_name=f"lhcb{BEAM}",
        json_file=json_path,
    )
    line = env[f"lhcb{BEAM}"]
    twiss_table = line.twiss(method="4d")

    print("Running xsuite tracking")
    print(f"Using kick angle {angle:.6f} rad for {NTURNS} turns")
    tracked_line = run_tracking_without_ac_dipole(
        line=line,
        tws=twiss_table,
        flattop_turns=NTURNS,
        action_list=[ACTION],
        angle_list=[angle],
    )

    print(f"Writing tracking dataframe to {tfs_path}")
    tracking_df = line_to_dataframes(tracked_line)[0]
    tfs.write(tfs_path, tracking_df)

    print(f"Writing turn-by-turn data to {tbt_path}")
    tbt.write_tbt(tbt_path, convert_to_tbt(tracked_line))


if __name__ == "__main__":
    main()
