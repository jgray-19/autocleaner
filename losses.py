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
        loss_sum = 0.0
        
        # Process each sample in the batch
        for i in range(B):
            # Flatten from (2, NBPMS, NTURNS) -> (2*NBPMS*NTURNS,)
            p = pred[i].view(-1)
            t = target[i].view(-1)
            
            # Remove mean
            p_centered = p - p.mean()
            t_centered = t - t.mean()
            
            # Dot product for numerator
            numerator = (p_centered * t_centered).sum()
            
            # Product of norms for denominator
            denom = torch.sqrt((p_centered**2).sum() * (t_centered**2).sum()) + self.eps
            
            # correlation in [-1, 1]
            corr = numerator / denom
            
            # Convert correlation to a loss => 1 - corr
            loss_i = 1 - corr
            loss_sum += loss_i
        
        # Average across the batch
        return loss_sum / B


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
        # Reshape to handle multi-dimensional data
        batch_size = x.size(0)
        n_channels = x.size(1)
        x_flat = x.reshape(batch_size, n_channels, -1)
        y_flat = y.reshape(batch_size, n_channels, -1)
        
        # Center the variables
        x_centered = x_flat - x_flat.mean(dim=-1, keepdim=True)
        y_centered = y_flat - y_flat.mean(dim=-1, keepdim=True)
        
        # Calculate correlation
        xy_cov = torch.sum(x_centered * y_centered, dim=-1)
        x_var = torch.sqrt(torch.sum(x_centered**2, dim=-1))
        y_var = torch.sqrt(torch.sum(y_centered**2, dim=-1))
        
        # Handle potential division by zero with small epsilon
        epsilon = 1e-8
        corr = xy_cov / (x_var * y_var + epsilon)
        
        # Average across batch and channels
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
            pred:   Tensor of shape (B, C, H, W), real-valued signals.
            target: Tensor of shape (B, C, H, W), real-valued signals.

        Returns:
            A scalar tensor representing the average SSP loss over the batch.
        """
        B = pred.shape[0]
        # We'll accumulate SSP values per sample, then average.
        ssp_sum = 0.0

        for i in range(B):
            # Extract single sample: (C, H, W)
            p = pred[i]
            t = target[i]

            # 1) Compute rFFT2 (real 2D FFT)
            #    -> shape (C, H, W//2+1), dtype=torch.complex64 (or complex128)
            p_fft = torch.fft.rfft2(p, norm="ortho")
            t_fft = torch.fft.rfft2(t, norm="ortho")

            # 2) Compute the L2 norm of the difference in the Fourier domain
            diff = p_fft - t_fft        # shape (C, H, W//2+1)
            diff_l2 = torch.sum(torch.abs(diff)**2).sqrt()  # scalar

            # 3) Compute the L2 norms of p_fft and t_fft individually
            p_l2 = torch.sum(torch.abs(p_fft)**2).sqrt()
            t_l2 = torch.sum(torch.abs(t_fft)**2).sqrt()

            # 4) Surface Similarity Parameter for this sample
            denom = p_l2 + t_l2 + self.eps
            ssp_i = diff_l2 / denom   # in [0, 1]

            ssp_sum += ssp_i

        # Average across batch
        return ssp_sum / B