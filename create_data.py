# Convert a MAD-NG output into a fake measurement, then use OMC3 to calculate this fake measurement.
import random
import shutil
from pathlib import Path
import concurrent.futures  # added for parallel execution

import tfs
import turn_by_turn as tbt
from turn_by_turn import madng

from lhcng.model import create_model_dir, get_model_dir
from lhcng.tracking import run_tracking, get_tfs_path, get_tbt_path
from config import NONOISE_INDEX

create_model = False
do_track = True

nturns = 2000
random.seed(42)

# Create a list of tunes for the fake measurement
tune_list = [
    [0.19, 0.18],
    [0.28, 0.29],
    [0.31, 0.32],
    [0.43, 0.49],
]
coupling = [
    False,
    1e-4,
    1e-3,
    # 3e-3,
]

def delete_unwanted_files(beam, nturns, coupling_knob, tunes):
    # Delete the TFS file using pathlib instead of os
    tfs_file = Path(get_tfs_path(beam, nturns, coupling_knob, tunes))
    if tfs_file.exists():
        tfs_file.unlink()

    # Delete the .ini files in the model directory using Path.iterdir()
    model_dir = Path(get_model_dir(beam, coupling_knob, tunes))
    for file in model_dir.iterdir():
        if file.suffix == ".ini":
            file.unlink()

    # Delete the saved sequence files using Path.unlink()
    madx_seq = model_dir / f"lhcb{beam}_saved.seq"
    mad_seq = model_dir / f"lhcb{beam}.mad"
    if madx_seq.exists():
        madx_seq.unlink()
    if mad_seq.exists():
        mad_seq.unlink()
    
    # Delete the macros folder (unchanged as it's already using shutil)
    macros_dir = model_dir / "macros"
    if macros_dir.exists():
        shutil.rmtree(macros_dir)

def process_configuration(args):
    beam, tunes, cknob = args
    model_dir = get_model_dir(beam, cknob, tunes)
    create_model_dir(beam, nat_tunes=tunes, coupling_knob=cknob)

    tfs_path = get_tfs_path(beam, nturns, coupling_knob=cknob, tunes=tunes)
    tbt_file = get_tbt_path(
        beam=beam,
        nturns=nturns,
        coupling_knob=cknob,
        tunes=tunes,
        index=NONOISE_INDEX,
    )

    print("Running MAD-NG Tracking")
    tfs_data = run_tracking(beam, nturns, model_dir, tunes=tunes, kick_amp=1e-4)

    # Save the TFS data
    print("Saving TFS data")
    tfs.write(tfs_path, tfs_data)
    del tfs_data  # Free up memory

    # Convert the TFS data to TBT data
    print("Converting TFS to TBT")
    tbt_data = madng.read_tbt(tfs_path)
    tbt.write(tbt_file, tbt_data)
    del tbt_data

    delete_unwanted_files(beam, nturns, cknob, tunes)

if __name__ == '__main__':
    configs = []
    for beam in [1, 2]:
        for tunes in tune_list:
            for cknob in coupling:
                configs.append((beam, tunes, cknob))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_configuration, configs)
