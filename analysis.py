from matplotlib import pyplot as plt

from config import NTURNS, BEAM, SAMPLE_INDEX
from rdt_constants import NORMAL_SEXTUPOLE_RDTS
from rdt_functions import (
    get_rdts_from_optics_analysis,
    get_tbt_path,
    run_harpy,
)

cleaned_path = get_tbt_path(beam=BEAM, nturns=NTURNS, index=-2)
run_harpy(BEAM, cleaned_path, clean=False)

hcleaned = get_tbt_path(beam=BEAM, nturns=NTURNS, index=SAMPLE_INDEX)
run_harpy(BEAM, hcleaned, clean=True)

nonoise_path = get_tbt_path(beam=BEAM, nturns=NTURNS, index=-1)
run_harpy(BEAM, nonoise_path, clean=False)

cleaned_dfs = get_rdts_from_optics_analysis(beam=BEAM, tbt_path=cleaned_path)
hcleaned_dfs = get_rdts_from_optics_analysis(beam=BEAM, tbt_path=hcleaned)
nonoise_rdts = get_rdts_from_optics_analysis(beam=BEAM, tbt_path=nonoise_path)

for rdt in NORMAL_SEXTUPOLE_RDTS:
    plt.figure()
    plt.plot(nonoise_rdts[rdt]["S"], nonoise_rdts[rdt]["AMP"], label="No Noise")
    plt.plot(hcleaned_dfs[rdt]["S"], hcleaned_dfs[rdt]["AMP"], label="Harpy Cleaned")
    plt.plot(cleaned_dfs[rdt]["S"], cleaned_dfs[rdt]["AMP"], label="Autoencoder Cleaned")
    plt.title(rdt)
    plt.legend()
plt.show()
