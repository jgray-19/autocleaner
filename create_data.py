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

# x -> 0.26 to 0.32
# y -> 0.26 to 0.32
tune_list = [
    # [0.07, 0.06],
    [0.19, 0.18],
    [0.26, 0.27],
    [0.27, 0.28],
    [0.28, 0.29],
    [0.29, 0.31],
    [0.32, 0.31],
    [0.31, 0.32],
    [0.43, 0.49],
    [0.74, 0.73],
    [0.72, 0.69],
    [0.71, 0.68],
    [0.69, 0.68],
]
coupling = [
    False,
    1e-4,
    1e-3,
    # 3e-3,
]

kick_amps = [5e-5, 1e-4, 2e-4]

def delete_unwanted_files(beam, nturns, coupling_knob, tunes, kick_amp):
    # Delete the TFS file using pathlib instead of os
    tfs_file = Path(get_tfs_path(beam, nturns, coupling_knob, tunes, kick_amp))
    if tfs_file.exists():
        tfs_file.unlink()
    
    # Loop through the model directory and delete all the files, including the macro directory
    model_dir = Path(get_model_dir(beam, coupling_knob, tunes))
    if model_dir.exists():
        for file in model_dir.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir() and file.name == "macros":
                shutil.rmtree(file)
        # Now delete the model directory itself
        model_dir.rmdir()
    # Uncomment the following lines if you want to delete more than just the whole directory


    # Delete the .ini files in the model directory using Path.iterdir()
    # model_dir = Path(get_model_dir(beam, coupling_knob, tunes))
    # for file in model_dir.iterdir():
    #     if file.suffix == ".ini":
    #         file.unlink()

    # Delete the saved sequence files using Path.unlink()
    # madx_seq = model_dir / f"lhcb{beam}_saved.seq"
    # mad_seq = model_dir / f"lhcb{beam}_saved.mad"
    # if madx_seq.exists():
    #     madx_seq.unlink()
    # if mad_seq.exists():
    #     mad_seq.unlink()
    
    # Delete the macros folder (unchanged as it's already using shutil)
    # macros_dir = model_dir / "macros"
    # if macros_dir.exists():
    #     shutil.rmtree(macros_dir)

def process_configuration(args):
    beam, tunes, cknob, kick = args
    model_dir = get_model_dir(beam, cknob, tunes)
    create_model_dir(beam, nat_tunes=tunes, coupling_knob=cknob)

    tfs_path = get_tfs_path(beam, nturns, coupling_knob=cknob, tunes=tunes, kick_amp=kick)
    tbt_file = get_tbt_path(
        beam=beam,
        nturns=nturns,
        coupling_knob=cknob,
        tunes=tunes,
        kick_amp=kick,
        index=NONOISE_INDEX,
    )

    print("Running MAD-NG Tracking")
    tfs_data = run_tracking(beam, nturns, model_dir, tunes=tunes, kick_amp=kick)

    # Save the TFS data
    print("Saving TFS data")
    tfs.write(tfs_path, tfs_data)
    del tfs_data  # Free up memory

    # Convert the TFS data to TBT data
    print("Converting TFS to TBT")
    tbt_data = madng.read_tbt(tfs_path)
    tbt.write(tbt_file, tbt_data)
    del tbt_data

    delete_unwanted_files(beam, nturns, cknob, tunes, kick)

if __name__ == '__main__':
    configs = []
    # for beam in [1, 2]:
    for beam in [1]:
        for tunes in tune_list:
            for cknob in coupling:
                for kick in kick_amps:
                    # tbt_file = get_tbt_path(
                    #         beam=beam,
                    #         nturns=nturns,
                    #         coupling_knob=cknob,
                    #         tunes=tunes,
                    #         kick_amp=kick,
                    #         index=NONOISE_INDEX,
                    #     )
                    # if not tbt_file.exists():
                    configs.append((beam, tunes, cknob, kick))
                    print(f"Processing beam {beam}, tunes {tunes}, coupling {cknob} kick {kick}")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_configuration, configs)
