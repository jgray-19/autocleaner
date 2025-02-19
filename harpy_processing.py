import tfs
from rdt_functions import run_harpy
from rdt_constants import FREQ_OUT_DIR

def run_harpy_analysis(BEAM, noise):
    """
    Runs Harpy analysis on the provided noisy TBT data.

    Args:
        BEAM (int): Beam number (1 or 2).
        noise (np.ndarray): Noisy TBT data.

    Returns:
        harpy_amps_df_x (DataFrame): DataFrame containing amplitude data from Harpy.
        harpy_freqs_df_x (DataFrame): DataFrame containing frequency data from Harpy.
    """
    print(f"Running Harpy on {noisy_tbt_path} for Beam {BEAM}...")
    
    # Run Harpy
    run_harpy(BEAM, noisy_tbt_path, clean=True)

    # Define output paths for Harpy results
    harpy_amps_path_x = FREQ_OUT_DIR / f"{noisy_tbt_path.stem}.sdds.ampsx"
    harpy_freqs_path_x = FREQ_OUT_DIR / f"{noisy_tbt_path.stem}.sdds.freqsx"

    # Read Harpy results
    harpy_amps_df_x = tfs.read(harpy_amps_path_x)
    harpy_freqs_df_x = tfs.read(harpy_freqs_path_x)

    print("Harpy analysis completed.")

    return harpy_amps_df_x, harpy_freqs_df_x
