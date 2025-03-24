import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ALPHA

class CorrelationLoss(nn.Module):
    """
    Computes 1 - Pearson correlation between prediction and target.
    
    For each sample in the batch, we:
      1. Flatten its data
      2. Remove mean
      3. Compute correlation = dot(pred, target)/[||pred||*||target||]
    Then average 1 - correlation across the batch.
    
    This loss encourages signals to match in shape (phase, relative amplitude)
    but doesn't penalize absolute scale as strongly as MSE.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 2, NBPMS, NTURNS)
            target: (B, 2, NBPMS, NTURNS)
        Returns:
            Scalar loss = average(1 - corr_i) over the batch i in [1..B].
        """
        B = pred.size(0)
        # Flatten each sample to a vector.
        p = pred.view(B, -1)
        t = target.view(B, -1)
        # Remove means per sample.
        p_centered = p - p.mean(dim=1, keepdim=True)
        t_centered = t - t.mean(dim=1, keepdim=True)
        # Compute components.
        numerator = torch.sum(p_centered * t_centered, dim=1)
        denom = torch.sqrt(torch.sum(p_centered**2, dim=1) * torch.sum(t_centered**2, dim=1)) + self.eps
        # 1 - Pearson correlation per sample.
        corr = numerator / denom
        return torch.mean(1 - corr)


class CombinedCorrelationLoss(nn.Module):
    def __init__(self):
        """
        Correlation loss that gradually transitions to MSE loss
        
        Args:
            epochs: Total training epochs
            fade_start: Fraction of training when correlation weight starts decreasing
            fade_end: Fraction of training when correlation weight reaches minimum
        """
        super().__init__()
        
    def correlation_coefficient(self, x, y):
        """Calculate batch-wise Pearson correlation coefficient"""
        # x, y shape: (B, 1, ...)
        B = x.size(0)
        # Flatten out the channel dimension.
        x_flat = x.view(B, -1)
        y_flat = y.view(B, -1)
        x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
        y_centered = y_flat - y_flat.mean(dim=1, keepdim=True)
        numerator = torch.sum(x_centered * y_centered, dim=1)
        x_norm = torch.sqrt(torch.sum(x_centered**2, dim=1))
        y_norm = torch.sqrt(torch.sum(y_centered**2, dim=1))
        epsilon = 1e-8
        corr = numerator / (x_norm * y_norm + epsilon)
        return corr.mean()
    def forward(self, pred, target):
        """Calculate the weighted loss"""
        # MSE component
        mse_loss = torch.mean((pred - target)**2)
        
        # Correlation component (1-corr so 0 is perfect correlation)
        corr = self.correlation_coefficient(pred, target)
        corr_loss = 1.0 - corr

        if corr_loss < 1e-8:
            corr_loss = 0
        
        # Apply weighting
        total_loss = ALPHA * 10 * corr_loss + (1-ALPHA) * mse_loss
        
        return total_loss


def fft_loss_per_bpm(pred, target, n_fft=1024, eps=1e-8):
    """
    Computes an FFT loss for each BPM over the turns dimension.
    
    Args:
        pred: Tensor of shape (B, 2, NBPMS, NTURNS)
        target: Tensor of shape (B, 2, NBPMS, NTURNS)
        n_fft: Number of FFT points.
        eps: Small constant to avoid log(0).
        
    Returns:
        Scalar loss computed as the mean squared error of the log magnitude spectra.
    """
    B, C, NBPMS, NTURNS = pred.shape
    # Reshape to combine batch, channel, and BPM dimensions.
    pred_flat = pred.view(B * C * NBPMS, NTURNS)
    target_flat = target.view(B * C * NBPMS, NTURNS)
    
    # Compute FFT along the time dimension.
    pred_fft = torch.fft.rfft(pred_flat, n=n_fft, dim=-1)
    target_fft = torch.fft.rfft(target_flat, n=n_fft, dim=-1)
    
    # Compute magnitude spectra.
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    
    # Optionally, compute the log magnitude to emphasize relative differences.
    pred_log = torch.log10(torch.clamp(pred_mag, min=eps))
    target_log = torch.log10(torch.clamp(target_mag, min=eps))
    
    # Compute mean squared error between the log spectra.
    loss = F.mse_loss(pred_log, target_log)
    return loss

class SSPLoss(nn.Module):
    """
    Implements the Surface Similarity Parameter (SSP) loss function for oscillatory spatio-temporal data,
    as introduced in:
      M. Wedler et al. (2022), "Surface Similarity Parameter: A New Machine Learning Loss Metric for
      Oscillatory Spatio-Temporal Data", arXiv:2204.06843

    The SSP measures how similar two signals (pred, target) are by computing their L2 norm difference
    in the Fourier domain, normalized by the sum of their individual L2 norms in the same domain.
    The result is in [0, 1], where 0 => perfect match, 1 => very dissimilar.

    For 2D data shaped (B, C, H, W):
      1) We perform an rFFT2 (real 2D FFT) of pred and target.
      2) Compute the L2 difference of the complex spectra for each sample.
      3) Normalize by the sum of the individual spectral L2 norms.
      4) Average across the batch.

    If your data are strictly real and have shape (B, C, H, W), rFFT2 suffices. If you need a full
    complex FFT, you can switch to fft2 or handle multi-dimensional signals (e.g., 1D, 3D) similarly.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps (float): Small epsilon to avoid division by zero.
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   Tensor of shape (B, 1, H, W), real-valued signals.
            target: Tensor of shape (B, 1, H, W), real-valued signals.

        Returns:
            A scalar tensor representing the average SSP loss over the batch.
        """
        p = pred.squeeze(1)  # shape: (B, H, W)
        t = target.squeeze(1)

        # Compute next power-of-two sizes for H and W
        def next_power_of_two(x: int) -> int:
            return 1 << (x - 1).bit_length()

        new_H = next_power_of_two(p.size(1))
        new_W = next_power_of_two(p.size(2))

        # Compute 2D FFTs with automatic zero-padding to power-of-two dimensions.
        p_fft = torch.fft.rfft2(p, s=(new_H, new_W), norm="ortho")
        t_fft = torch.fft.rfft2(t, s=(new_H, new_W), norm="ortho")

        # Compute L2 norm differences per sample.
        diff = p_fft - t_fft
        diff_l2 = torch.norm(diff, dim=(1, 2))
        p_l2 = torch.norm(p_fft, dim=(1, 2))
        t_l2 = torch.norm(t_fft, dim=(1, 2))
        ssp = diff_l2 / (p_l2 + t_l2 + self.eps)
        return ssp.mean()
