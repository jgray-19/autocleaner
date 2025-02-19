import torch
import torch.nn as nn
import torch.nn.functional as F
from omc3.harpy.frequency import windowing

from config import NTURNS, HARPY_INPUT, FFT_WEIGHT, ALPHA

WINDOW = windowing(NTURNS, window="hann")
WINDOW = torch.tensor(WINDOW, dtype=torch.float64)

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

@torch.jit.script
def fast_fft_log_mse(
    reconstructed: torch.Tensor,
    clean: torch.Tensor,
    window: torch.Tensor,
    n_fft: int = 2 ** HARPY_INPUT["output_bits"],
    eps: float = 1e-8
) -> torch.Tensor:
    B, N, T = reconstructed.shape

    # Apply window function
    rec_flat = reconstructed.reshape(B * N, T) * window
    clean_flat = clean.reshape(B * N, T) * window

    # Perform FFT (TorchScript does not support complex tensors)
    rec_fft = torch.fft.rfft(rec_flat, n=n_fft, dim=-1)
    clean_fft = torch.fft.rfft(clean_flat, n=n_fft, dim=-1)

    # Extract the real magnitude of FFT (TorchScript does NOT support complex tensors)
    rec_mag = torch.view_as_real(rec_fft).pow(2).sum(-1).sqrt()
    clean_mag = torch.view_as_real(clean_fft).pow(2).sum(-1).sqrt()

    # Compute log10(FFT magnitude)
    rec_log = torch.log10(torch.clamp(rec_mag, min=eps))
    clean_log = torch.log10(torch.clamp(clean_mag, min=eps))

    # Compute MSE loss
    return torch.nn.functional.mse_loss(rec_log, clean_log)

class CombinedTimeFreqLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.time_loss_func = correlation_loss

    def forward(self, pred, target):
        # Time-domain portion
        time_loss = self.time_loss_func(pred, target)

        # Frequency-domain portion (log10 amplitude comparison)
        if FFT_WEIGHT == 0:
            return time_loss

        freq_loss = fast_fft_log_mse(pred, target, WINDOW)

        # Weighted combination
        total_loss = time_loss + FFT_WEIGHT * freq_loss
        return total_loss
    
def correlation_loss(pred, target, eps=1e-8):
    # Flatten (batch, channels, length) => (batch*channels, length)
    B, C, T = pred.shape
    pred_flat = pred.view(B*C, T)
    tgt_flat  = target.view(B*C, T)

    # Zero-mean
    pred_centered = pred_flat - pred_flat.mean(dim=-1, keepdim=True)
    tgt_centered  = tgt_flat  - tgt_flat.mean(dim=-1, keepdim=True)

    # Cosine similarity or correlation
    numerator = (pred_centered * tgt_centered).sum(dim=-1)
    denom = torch.sqrt((pred_centered**2).sum(dim=-1) * (tgt_centered**2).sum(dim=-1)) + eps
    corr = numerator / denom  # correlation in [-1, 1]
    
    # Convert correlation to a loss => 1 - corr
    loss = 1 - corr.mean()
    return loss

def combined_mse_correlation_loss(pred, target, alpha=ALPHA):
    loss_mse = torch.mean((pred - target)**2)
    loss_corr = correlation_loss(pred, target)
    return alpha * loss_mse + (1 - alpha) * loss_corr

# class MSLELoss(nn.Module):
#     def forward(self, pred, target):
#         pred = torch.log1p(pred + 1)
#         target = torch.log1p(target + 1)
#         # Normalise all bpms for that turn using max absolute value
#         pred = pred / (torch.max(torch.abs(pred), dim=0, keepdim=True)[0] + 1e-8)
#         target = target / (torch.max(torch.abs(target), dim=0, keepdim=True)[0] + 1e-8)
#         return torch.mean((pred - target) ** 2)