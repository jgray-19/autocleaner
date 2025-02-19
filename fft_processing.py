from typing import Tuple

import numpy as np
import torch
from omc3.harpy.frequency import get_freq_mask, windowing
from config import NTURNS, HARPY_INPUT

window = windowing(NTURNS, window="hann")
window = torch.tensor(window, dtype=torch.float64)


def windowed_padded_rfft_torch(
        matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the spectra using specified windowing function and zero-padding with PyTorch tensors.

    Args:
        matrix: `torch.Tensor` of TbT matrix (BPMs x turns).
        tunes: List of tunes [x, y, z].
        svd: Reduced (U_matrix, np.dot(S_matrix, V_matrix)) of original TbT matrix, defaults to `None`.

    Returns:
        Tuple of `torch.Tensor` for frequencies and coefficients.
    """
    matrix = matrix.T # Transpose matrix to have turns as rows
    tunes = HARPY_INPUT["tunes"]

    # Define lengths for padding and output
    padded_len = 2 ** HARPY_INPUT["turn_bits"]
    output_len = 2 ** HARPY_INPUT["output_bits"]
    sub_bins = padded_len // output_len 

    # Get frequency mask
    HARPY_INPUT["turn_bits"] = int(np.log2(padded_len))
    freq_mask = get_freq_mask(HARPY_INPUT, tunes, 2 / matrix.shape[1])
    n_bins = np.sum(freq_mask).item() // sub_bins
    n_bpms = matrix.shape[0]

    # Common frequency array
    common_freqs = torch.fft.rfftfreq(padded_len * 2, d=1.0)

    # Apply windowing function
    windowed_matrix = matrix * window

    # Perform FFT
    coefs = torch.fft.rfft(windowed_matrix, n=padded_len * 2)
    coefs = coefs[:, freq_mask]

    # Reshape coefficients and find maximum indices
    indices = np.indices((n_bpms, n_bins))[1] * sub_bins

    reshaped_coefs = torch.reshape(coefs, (n_bpms, n_bins, sub_bins))
    max_indices = np.argmax(np.abs(reshaped_coefs.detach().numpy()), axis=2)
    del reshaped_coefs
    argsmax = indices + max_indices

    # Calculate coefficients and frequencies
    coefficients = 2 * coefs[np.arange(n_bpms)[:, None], argsmax]
    del coefs
    frequencies = common_freqs[freq_mask]
    frequencies = frequencies[argsmax]

    return frequencies, coefficients

def calculate_fft_and_amps(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs, amps = windowed_padded_rfft_torch(data.T)
    return freqs, normalize_and_log(amps)

# Function to normalize and log amplitudes
def normalize_and_log(amps):
    amps = torch.abs(amps)
    amps = amps / amps.max(dim=1, keepdim=True)[0]
    return torch.log10(amps)
