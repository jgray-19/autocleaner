import scipy.signal
import torch
import torch.nn as nn
from generic_parser.tools import DotDict
from scipy.signal import find_peaks

from fft_processing import windowed_padded_rfft_torch


def smooth_fft_amps(
    fft_amps: torch.Tensor, window_length: int = 11, polyorder: int = 5
) -> torch.Tensor:
    """
    Smooths the FFT amplitudes using a Savitzky-Golay filter.

    Args:
        fft_amps (torch.Tensor): FFT amplitudes to be smoothed.
        window_length (int): The length of the filter window (must be a positive odd integer).
        polyorder (int): The order of the polynomial used to fit the samples (must be less than window_length).

    Returns:
        torch.Tensor: Smoothed FFT amplitudes.
    """
    smoothed_amps = []
    for amp in fft_amps:
        real_part = amp.detach().numpy()
        smoothed_real = scipy.signal.savgol_filter(real_part, window_length, polyorder)
        smoothed_amp = torch.tensor(smoothed_real, dtype=torch.float32)
        smoothed_amps.append(smoothed_amp)

    return torch.stack(smoothed_amps)


def compute_fft_loss(
    outputs: torch.Tensor,
    peak_weights: torch.Tensor,
    zero_noise_amps_logspace: torch.Tensor,
    harpy_input: DotDict,
    criterion: nn.Module,
    tolerance: float = 0.002,
) -> torch.Tensor:
    """
    Computes the FFT-based loss by applying weights dynamically based on proximity to target peaks.

    Args:
        outputs (torch.Tensor): Model outputs.
        target_peaks (list): List of target peak frequencies for each BPM.
        zero_noise_amps (torch.Tensor): Amplitudes of the zero-noise spectrum.
        harpy_input (DotDict): Input settings for the FFT computation.
        criterion (nn.Module): Loss function.
        tolerance (float): Frequency tolerance for matching peaks.

    Returns:
        torch.Tensor: Computed FFT loss.
    """
    # Compute FFT of the outputs
    _, outputs_amps = windowed_padded_rfft_torch(
        harpy_input, outputs, harpy_input["tunes"]
    )
    outputs_amps = torch.abs(outputs_amps)

    # Normalize and apply log10
    outputs_amps = outputs_amps / outputs_amps.max(dim=1, keepdim=True)[0]
    outputs_amps = torch.log10(outputs_amps)
    outputs_amps = smooth_fft_amps(outputs_amps, polyorder=5)

    # Remove any values below -6
    # limit = -6
    # outputs_amps[outputs_amps < limit] = limit
    # zero_noise_amps_logspace[zero_noise_amps_logspace < limit] = limit

    # Compute the loss
    loss = criterion(
        (outputs_amps + 10) * peak_weights,
        (zero_noise_amps_logspace + 10) * peak_weights,
    )

    # Check for any nans and then throw an error
    if torch.isnan(loss).any():
        raise ValueError("Loss contains NaN values")
    return loss


def find_significant_peaks(
    zero_noise_amps: torch.Tensor,
    zero_noise_freqs: torch.Tensor,
    nbpms: int,
    prominence: float,
    weighting: int,
) -> list:
    """
    Find significant peaks in the zero-noise spectrum and produce a matrix of 1s and 100s,
    where 1s represents non-significant peaks and 100s represent significant peaks.

    Args:
        zero_noise_amps (torch.Tensor): Zero-noise amplitudes.
        zero_noise_freqs (torch.Tensor): Zero-noise frequencies.
        nbpms (int): Number of BPMs.

    Returns:
        list: List of significant peaks for each BPM.
    """
    peak_weights = torch.ones_like(zero_noise_amps)  # Initialize peak weights
    for i in range(nbpms):
        # Find all peaks in the zero-noise spectrum
        peaks, properties = find_peaks(
            zero_noise_amps[i].detach().numpy(),
            height=-10,  # Only consider peaks with log10 amplitude > -10
            prominence=prominence,  # Adjust prominence as needed
        )
        print(f"Found {len(peaks)} significant peaks for BPM {i}")

        # Update peak weights
        peak_weights[i, peaks] = weighting
    return peak_weights
