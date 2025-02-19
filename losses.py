import torch
import torch.nn as nn
from omc3.harpy.frequency import windowing

from config import NTURNS, HARPY_INPUT, FFT_WEIGHT, BATCH_SIZE, NUM_CHANNELS

default_window = windowing(NTURNS, window="hann")
default_window = torch.tensor(default_window, dtype=torch.float64)

class HybridMSERelativeLoss(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-5):
        """
        alpha: Weighting factor between absolute MSE and relative error
        eps:   Small constant for numerical stability
        """
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        # Standard MSE
        mse_part = (pred - target).pow(2)

        # Relative error => dividing by |target|^2 (or + eps)
        rel_part = mse_part / (target.pow(2).clamp_min(self.eps))

        return self.alpha * mse_part.mean() + (1 - self.alpha) * rel_part.mean()

class DedicatedMSLELoss(nn.Module):
    def __init__(self):
        """
        Dedicated Mean Squared Logarithmic Error Loss for turn-by-turn data.
        """
        super(DedicatedMSLELoss, self).__init__()

    def forward(self, pred, target):
        # Shift values from [-1,1] to [-0.999 1.001] so that log1p is well-defined.
        pred_shifted = pred + 1.0e-3
        target_shifted = target + 1.0e-3
        
        # Apply log1p transformation.
        log_pred = torch.log1p(pred_shifted)
        log_target = torch.log1p(target_shifted)
        
        # Compute the Mean Squared Error between the log-transformed values.
        loss = torch.mean((log_pred - log_target) ** 2)
        return loss
    
class SafeMSLELoss(nn.Module):
    def __init__(self):
        """
        Safe version of the Mean Squared Logarithmic Error Loss.
        """
        super(SafeMSLELoss, self).__init__()

    def forward(self, pred, target):
        # Shift values from [-1,1] to [0,2] so that log1p is well-defined.
        pred_shifted = pred + 1.0
        target_shifted = target + 1.0
        
        # Apply log1p transformation.
        log_pred = torch.log1p(pred_shifted)
        log_target = torch.log1p(target_shifted)
        
        # Compute the Mean Squared Error between the log-transformed values.
        loss = torch.mean((log_pred - log_target) ** 2)
        return loss

def fast_fft_log_mse(
    reconstructed: torch.Tensor,
    clean: torch.Tensor,
    fft_criterion: torch.nn.Module,
    window: torch.Tensor = default_window,
    n_fft: int = 2 ** HARPY_INPUT["output_bits"],
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute an MSE loss in log(FFT) space over a batch of data.
    
    Args:
        reconstructed (torch.Tensor): shape (batch, BPMs, T)
        clean (torch.Tensor): shape (batch, BPMs, T)
        fft_criterion (nn.Module): e.g. nn.MSELoss()
        window (torch.Tensor, optional): 1D window of shape (T,) to apply before FFT.
        n_fft (int, optional): Zero-padding length. If None, no extra padding is used.
        eps (float): Small offset to avoid log(0).
    
    Returns:
        torch.Tensor: Scalar loss (averaged over batch and BPMs).
    """
    B, C, T = reconstructed.shape
    # 1) Flatten (B × N, T) so we can apply a single batched FFT
    rec_flat = reconstructed.reshape(B * C, T)
    clean_flat = clean.reshape(B * C, T)

    # 2) Multiply by window
    rec_flat = rec_flat * window
    clean_flat = clean_flat * window

    # 3) Perform batched RFFT
    rec_fft = torch.fft.rfft(rec_flat, n=n_fft, dim=-1)
    clean_fft = torch.fft.rfft(clean_flat, n=n_fft, dim=-1)

    # 4) Convert to log10 of magnitude
    rec_log = torch.log10(torch.clamp(rec_fft.abs(), min=eps))
    clean_log = torch.log10(torch.clamp(clean_fft.abs(), min=eps))
    
    # 5) Compute MSE loss
    fft_loss = fft_criterion(rec_log, clean_log)
    return fft_loss


class CombinedTimeFreqLoss(torch.nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.time_loss_func = nn.MSELoss()
        self.fft_criterion = nn.MSELoss()
        # self.target = target  # This will be the clean data without any batch dimension, shape (BPMs, T)
        # target = target * default_window
        # fft_target = torch.fft.rfft(target, n=2 ** HARPY_INPUT["output_bits"], dim=-1)
        # self.clean_log = torch.log10(torch.clamp(fft_target.abs(), min=1e-8))

    def forward(self, pred, target):
        # Time-domain portion
        time_loss = self.time_loss_func(pred, target)

        # Frequency-domain portion (log10 amplitude comparison)
        if FFT_WEIGHT == 0:
            return time_loss

        # Repeat the clean_log based on the batch size of pred
        # batch_size = pred.shape[0]
        # clean_log_repeated = self.clean_log.repeat(batch_size, 1)

        freq_loss = fast_fft_log_mse(
            pred, target, fft_criterion=self.fft_criterion,
        )

        # Weighted combination
        total_loss = time_loss + FFT_WEIGHT * freq_loss
        return total_loss