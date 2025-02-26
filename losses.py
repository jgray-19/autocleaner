import torch
import torch.nn as nn
from omc3.harpy.frequency import windowing

from config import NTURNS, HARPY_INPUT, ALPHA

WINDOW = windowing(NTURNS, window="hann")
WINDOW = torch.tensor(WINDOW, dtype=torch.float64)

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

class CombinedMSECorrelationLoss(nn.Module):
    """
    A loss that combines Mean Squared Error (MSE) and a correlation-based metric:
      loss = alpha * MSE + (1 - alpha) * (1 - corr).

    MSE enforces amplitude matching, 
    while correlation focuses on signal shape/phase alignment.
    """
    def __init__(self, eps=1e-8):
        """
        Args:
            alpha (float): Weight given to MSE vs. correlation. 
                           alpha=1 => pure MSE, alpha=0 => pure correlation loss.
            eps (float):   Small constant to avoid division by zero in correlation.
        """
        super().__init__()
        self.alpha = ALPHA
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined loss over the entire batch.

        Expects pred and target to have the same shape, e.g. 
          (B, 2, NBPMS, NTURNS) or (B, Channels, H, W).
        """
        # 1) MSE
        mse = torch.mean((pred - target) ** 2)

        # 2) Pearson Correlation
        # Flatten entire batch for a single correlation measure:
        p = pred.view(-1)
        t = target.view(-1)

        p_centered = p - p.mean()
        t_centered = t - t.mean()
        numerator = (p_centered * t_centered).sum()
        denom = torch.sqrt((p_centered**2).sum() * (t_centered**2).sum()) + self.eps
        corr = numerator / denom   # in [-1, 1]

        # 3) Convert correlation to 1 - corr for the correlation-loss portion
        corr_loss = 1 - corr

        # 4) Combined
        loss = self.alpha * mse + (1 - self.alpha) * corr_loss
        return loss