from matplotlib import pyplot as plt

from config import NTURNS, BEAM, SAMPLE_INDEX
from rdt_constants import NORMAL_SEXTUPOLE_RDTS
from rdt_functions import (
    get_rdts_from_optics_analysis,
    get_tbt_path,
    run_harpy,
)

cleaned_path = get_tbt_path(beam=BEAM, nturns=NTURNS, index=-2)
print("Cleaned data read in")
run_harpy(BEAM, cleaned_path, clean=False)
print("Done running Harpy on cleaned data")

hcleaned = get_tbt_path(beam=BEAM, nturns=NTURNS, index=SAMPLE_INDEX)
print("Noisy data read in")
run_harpy(BEAM, hcleaned, clean=True)
print("Done running Harpy on noisy data (also cleaned this one)")

nonoise_path = get_tbt_path(beam=BEAM, nturns=NTURNS, index=-1)
print("No noise data read in")
run_harpy(BEAM, nonoise_path, clean=False)
print("Done running Harpy on no noise data")

print("Running optics analysis on the data")
cleaned_dfs = get_rdts_from_optics_analysis(beam=BEAM, tbt_path=cleaned_path)
hcleaned_dfs = get_rdts_from_optics_analysis(beam=BEAM, tbt_path=hcleaned)
nonoise_rdts = get_rdts_from_optics_analysis(beam=BEAM, tbt_path=nonoise_path)
print("Done running optics analysis")

for rdt in NORMAL_SEXTUPOLE_RDTS:
    plt.figure()
    plt.plot(nonoise_rdts[rdt]["S"], nonoise_rdts[rdt]["AMP"], label="No Noise")
    plt.plot(hcleaned_dfs[rdt]["S"], hcleaned_dfs[rdt]["AMP"], label="Harpy Cleaned")
    plt.plot(cleaned_dfs[rdt]["S"], cleaned_dfs[rdt]["AMP"], label="Autoencoder Cleaned")
    plt.title(rdt)
    plt.legend()
plt.show()
