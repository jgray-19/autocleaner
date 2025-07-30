import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ALPHA, NBPMS, NTURNS


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
        denom = (
            torch.sqrt(
                torch.sum(p_centered**2, dim=1) * torch.sum(t_centered**2, dim=1)
            )
            + self.eps
        )
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
        mse_loss = torch.mean((pred - target) ** 2)

        # Correlation component (1-corr so 0 is perfect correlation)
        corr = self.correlation_coefficient(pred, target)
        corr_loss = 1.0 - corr

        if corr_loss < 1e-8:
            corr_loss = 0

        # Apply weighting
        total_loss = ALPHA * 10 * corr_loss + (1 - ALPHA) * mse_loss

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

    Now also
    1. Pre-computes the padded FFT sizes in __init__ (so the bit_length logic
       runs only once).
    2. Uses torch.fft.rfftn + torch.linalg.norm on the complex tensor directly
       (all in optimized C/CUDA kernels) instead of hand-rolling real/imag ops.
    3. Flattens all non-batch dims before the norm reduction to keep it a single
       contiguous reduction per sample.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            H, W:  the height and width of your input patches (must be constant
                   across calls).
            eps:   small constant to avoid div by zero.
        """
        super().__init__()
        self.eps = eps
        # next power-of-two
        self.new_H = 1 << (NBPMS - 1).bit_length()
        self.new_W = 1 << (NTURNS - 1).bit_length()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
          pred, target: (B, C, H, W), real-valued.
        Returns:
          scalar SSP loss.
        """
        # compute batched rFFT in one shot over last two dims
        p_fft = torch.fft.rfftn(
            pred, s=(self.new_H, self.new_W), dim=(-2, -1), norm="ortho"
        )
        t_fft = torch.fft.rfftn(
            target, s=(self.new_H, self.new_W), dim=(-2, -1), norm="ortho"
        )

        # pointwise difference
        diff = p_fft - t_fft

        # flatten spatial+channel dims → (B, N), N = C * new_H * (new_W//2+1)
        diff = diff.flatten(1)
        p_vec = p_fft.flatten(1)
        t_vec = t_fft.flatten(1)

        # diff_l2 = torch.linalg.norm(diff, dim=1)
        # p_l2 = torch.linalg.norm(p_vec, dim=1)
        # t_l2 = torch.linalg.norm(t_vec, dim=1)

        # Frobenius norm over each row (complex-aware, manual for MPS/compat)
        diff_l2 = torch.sqrt((diff.real**2 + diff.imag**2).sum(dim=1))
        p_l2 = torch.sqrt((p_vec.real**2 + p_vec.imag**2).sum(dim=1))
        t_l2 = torch.sqrt((t_vec.real**2 + t_vec.imag**2).sum(dim=1))
        den = torch.clamp(p_l2 + t_l2, min=1e-8)
        ssp = diff_l2 / den
        return ssp.mean()

class ResidualSpectralLoss(nn.Module):
    """
    Loss = a·MSE_time(cleaned, clean)
         + β·MSE_spectrum(residual_pred, residual_true)
         + λ·||residual_pred||_2^2
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.1, lambda_id: float = 1e-3):
        """
        Args:
            alpha:    weight for the time-domain MSE on the cleaned signal
            beta:     weight for the spectral-domain MSE on the residual
            lambda_id: weight for the L2 identity penalty on the predicted residual
        """
        super().__init__()
        self.time_mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
        self.lambda_id = lambda_id

    def forward(
        self, noisy: torch.Tensor, clean: torch.Tensor, cleaned: torch.Tensor
    ) -> torch.Tensor:
        # 1) time-domain MSE on the cleaned output
        loss_time = self.time_mse(cleaned, clean)

        # 2) compute true & predicted residuals
        r_true = noisy - clean
        r_pred = noisy - cleaned

        # 3) FFT & magnitude of residuals
        R_true = torch.fft.rfft(r_true, dim=-1)
        R_pred = torch.fft.rfft(r_pred, dim=-1)
        mag_true = torch.sqrt(R_true.real**2 + R_true.imag**2 + 1e-12)
        mag_pred = torch.sqrt(R_pred.real**2 + R_pred.imag**2 + 1e-12)

        # ||mag_pred - mag_true||_F / (||mag_true||_F + eps)
        diff = mag_pred - mag_true
        num = torch.norm(diff, p="fro")
        denom = torch.norm(mag_true, p="fro")
        loss_spec = num / denom

        # 5) L2 identity penalty on predicted residual
        loss_id = self.lambda_id * torch.mean(r_pred**2)

        # 6) combine all terms
        return self.alpha * loss_time + self.beta * loss_spec + loss_id
