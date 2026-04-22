import glob
import os
from typing import Callable

import pytorch_lightning as pl
import torch
import torch.optim as optim

from config import ALPHA, MIN_LR, MODEL_TYPE, NUM_EPOCHS, RESIDUALS, SCHEDULER
from losses import CombinedCorrelationLoss, CorrelationLoss, SSPLoss, fft_loss_per_bpm
from ml_models.conv_2d import (
    Conv2DAutoencoder,
    Conv2DAutoencoderLeaky,
    Conv2DAutoencoderLeakyFourier,
    Conv2DAutoencoderLeakyNoFC,
    DeepConvAutoencoder,
    SineConv2DAutoencoder,
)
from ml_models.fno import FNO2d
from ml_models.unet import (
    UNetAutoencoder,
    UNetAutoencoderFixedDepth,
    UNetAutoencoderFixedDepthCheckpoint,
)


class LitAutoencoder(pl.LightningModule):
    def __init__(self, model, learning_rate, weight_decay, loss_type):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.corr_loss = None
        self.combined_corr_loss = None
        self.ssp = None
        self.loss_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor
        ]
        if loss_type == "mse":
            self.loss_fn = self.mse_loss_wrapper
        elif loss_type == "noise_norm_mse":
            self.loss_fn = self.noise_normalized_mse_loss
        elif loss_type == "corr":
            self.corr_loss = CorrelationLoss()
            self.loss_fn = self.corr_loss_wrapper
        elif loss_type == "combined":
            self.combined_corr_loss = CombinedCorrelationLoss()
            self.loss_fn = self.combined_corr_loss_wrapper
        elif loss_type == "ssp":
            self.ssp = SSPLoss()
            self.loss_fn = self.ssp_loss_wrapper
        elif loss_type == "comb_ssp":
            self.ssp = SSPLoss()
            self.loss_fn = self.combined_ssp_loss
        elif loss_type == "comb_ssp_norm":
            self.ssp = SSPLoss()
            self.loss_fn = self.combined_ssp_norm_loss
        elif loss_type == "fft":
            self.loss_fn = self.combined_fft_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def mse_loss_wrapper(self, pred, target, noise_var=None):
        return torch.nn.functional.mse_loss(pred, target)

    def corr_loss_wrapper(self, pred, target, noise_var=None):
        assert self.corr_loss is not None
        return self.corr_loss(pred, target)

    def combined_corr_loss_wrapper(self, pred, target, noise_var=None):
        assert self.combined_corr_loss is not None
        return self.combined_corr_loss(pred, target)

    def ssp_loss_wrapper(self, pred, target, noise_var=None):
        assert self.ssp is not None
        return self.ssp(pred, target)

    def combined_fft_loss(self, pred, target, noise_var=None):
        # Standard time-domain MSE
        mse_loss = torch.mean((pred - target) ** 2)
        # Frequency-domain loss using your FFT-based function (adjust hyperparameters if needed)
        fft_loss = fft_loss_per_bpm(pred, target)
        return ALPHA * mse_loss + (1 - ALPHA) * fft_loss

    def combined_ssp_loss(self, pred, target, noise_var=None):
        assert self.ssp is not None
        ssp_loss = self.ssp(pred, target)
        mse_loss = torch.nn.functional.mse_loss(pred, target)
        return (1 - ALPHA) * ssp_loss + ALPHA *  mse_loss

    def combined_ssp_norm_loss(self, pred, target, noise_var):
        assert self.ssp is not None
        assert noise_var is not None
        ssp_loss = self.ssp(pred, target)
        norm_mse_loss = self.noise_normalized_mse_loss(pred, target, noise_var)
        return (1 - ALPHA) * ssp_loss + ALPHA * norm_mse_loss

    def noise_normalized_mse_loss(self, pred, target, noise_var):
        squared_error = (pred - target) ** 2
        return torch.mean(squared_error / noise_var)

    def forward(self, x):
        return self.model(x)

    def get_batch_loss(self, batch):
        # Concatenate along batch dimension (assuming shape (B, 1, NBPMS, NTURNS))
        combined_noisy = torch.cat([batch["noisy_x"], batch["noisy_y"]], dim=0)
        combined_clean = torch.cat([batch["clean_x"], batch["clean_y"]], dim=0)
        combined_noise_var = None
        if self.loss_type in {"noise_norm_mse", "comb_ssp_norm"}:
            combined_noise_var = torch.cat(
                [batch["noise_var_x"], batch["noise_var_y"]], dim=0
            )
        combined_batch_size = combined_noisy.size(0)

        # Optionally shuffle the combined batch here (if your DataLoader doesn’t already shuffle individual samples)
        perm = torch.randperm(combined_batch_size)
        combined_noisy = combined_noisy[perm]
        combined_clean = combined_clean[perm]
        if combined_noise_var is not None:
            combined_noise_var = combined_noise_var[perm]

        # Process through the model
        combined_recon = self(combined_noisy)

        pred_for_loss = combined_noisy - combined_recon if RESIDUALS else combined_recon

        loss = self.loss_fn(pred_for_loss, combined_clean, combined_noise_var)

        return loss, combined_batch_size

    def training_step(self, batch, batch_idx):
        loss, combined_batch_size = self.get_batch_loss(batch)
        self.log("train_loss", loss, batch_size=combined_batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, combined_batch_size = self.get_batch_loss(batch)
        self.log("val_loss", loss, batch_size=combined_batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if SCHEDULER:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, eta_min=MIN_LR, T_max=NUM_EPOCHS
            )
            return [optimizer], [scheduler]
        return optimizer


def find_newest_file(directory_path):
    # Get a list of all files in the directory
    files = glob.glob(os.path.join(directory_path, "*.ckpt"))

    # Find the newest file
    return max(files, key=os.path.getctime)


def get_model():
    # Initialize or Load Model
    if MODEL_TYPE == "sine":
        return SineConv2DAutoencoder()
    elif MODEL_TYPE == "conv":
        return Conv2DAutoencoder()
    elif MODEL_TYPE == "leaky":
        return Conv2DAutoencoderLeaky()
    elif MODEL_TYPE == "nofc":
        return Conv2DAutoencoderLeakyNoFC()
    elif MODEL_TYPE == "fourier":
        return Conv2DAutoencoderLeakyFourier()
    elif MODEL_TYPE == "deep":
        return DeepConvAutoencoder()
    elif MODEL_TYPE == "unet":
        return UNetAutoencoder()
    elif MODEL_TYPE == "unet_fixed":
        return UNetAutoencoderFixedDepth()
    elif MODEL_TYPE == "unet_fixed_checkpoint":
        return UNetAutoencoderFixedDepthCheckpoint()
    elif MODEL_TYPE == "fno":
        return FNO2d()
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
