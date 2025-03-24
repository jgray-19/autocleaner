# Convert a MAD-NG output into a fake measurement, then use OMC3 to calculate this fake measurement.
import random

import tfs
import turn_by_turn as tbt
from turn_by_turn import madng

from lhcng.model import create_model_dir, get_model_dir
from lhcng.tracking import run_tracking, get_tfs_path, get_tbt_path
from config import NONOISE_INDEX

BEAM = 1
create_model = False
do_track = True

nturns = 1000
random.seed(42)

# Create the MAD-X model
model_dir = get_model_dir(BEAM)
if create_model or not model_dir.exists():
    create_model_dir(BEAM)

# Get the TFS and TBT paths
tfs_path = get_tfs_path(BEAM, nturns)
tbt_file = get_tbt_path(beam=BEAM, nturns=nturns, index=NONOISE_INDEX)

# Run the tracking
print("Running MAD-NG Tracking")
tfs_data = run_tracking(BEAM, nturns, kick_amp=1e-4)

# Save the TFS data
print("Saving TFS data")
tfs.write(tfs_path, tfs_data)
del tfs_data # Free up memory

# Convert the TFS data to TBT data
print("Converting TFS to TBT")
tbt_data = madng.read_tbt(tfs_path)
tbt.write(tbt_file, tbt_data)