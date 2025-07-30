import glob
import os

import pytorch_lightning as pl
import torch
import torch.optim as optim

from config import (
    ALPHA,
    LEARNING_RATE,
    MIN_LR,
    MODEL_TYPE,
    NUM_DECAY_EPOCHS,
    RESIDUALS,
    SCHEDULER,
    IDENTITY_PENALTY,
)
from losses import (
    CombinedCorrelationLoss,
    CorrelationLoss,
    SSPLoss,
    fft_loss_per_bpm,
    ResidualSpectralLoss,
)
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
    ModifiedUNetFixed,
    UNetAutoencoder,
    UNetAutoencoderFixedDepth,
    UNetAutoencoderFixedDepthCheckpoint,
    GatedUNetAutoencoder,
)
from schedulers import HalvingCosineLR


class LitAutoencoder(pl.LightningModule):
    def __init__(self, model, learning_rate, weight_decay, loss_type):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.need_residual = False
        if loss_type == "mse":
            self.loss_fn = torch.nn.functional.mse_loss
        elif loss_type == "corr":
            self.loss_fn = CorrelationLoss()
        elif loss_type == "combined":
            self.loss_fn = CombinedCorrelationLoss()
        elif loss_type == "ssp":
            self.loss_fn = SSPLoss()
        elif loss_type == "comb_ssp":
            self.ssp = SSPLoss()
            self.mse = torch.nn.functional.mse_loss
            self.loss_fn = self.combined_ssp_loss
        elif loss_type == "comb_ssp_resid":
            self.ssp = SSPLoss()
            self.mse = torch.nn.functional.mse_loss
            self.loss_fn = self.combined_ssp_loss_resid
            self.need_residual = True
        elif loss_type == "residual":
            self.loss_fn = ResidualSpectralLoss(
                alpha=ALPHA, 
                beta=1-ALPHA, 
                lambda_id=IDENTITY_PENALTY
            )
            self.need_residual = True
        elif loss_type == "fft":
            self.loss_fn = self.combined_fft_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def combined_fft_loss(self, pred, target):
        # Standard time-domain MSE
        mse_loss = torch.mean((pred - target) ** 2)
        # Frequency-domain loss using your FFT-based function (adjust hyperparameters if needed)
        fft_loss = fft_loss_per_bpm(pred, target)
        return ALPHA * mse_loss + (1 - ALPHA) * fft_loss

    def combined_ssp_loss(self, pred, target):
        ssp_loss = self.ssp(pred, target)
        mse_loss = self.mse(pred, target)
        return ALPHA * mse_loss + (1 - ALPHA) * ssp_loss

    def combined_ssp_loss_resid(self, noisy, clean, cleaned):
        # 2) compute true & predicted residuals
        mse_loss = self.mse(cleaned, clean)
        r_true = noisy - clean
        r_pred = noisy - cleaned
        ssp_loss = self.ssp(r_pred, r_true)
        return ALPHA * mse_loss + (1 - ALPHA) * ssp_loss

    def forward(self, x):
        return self.model(x)

    def reconstruct(self, noisy):
        """Reconstruct the input from noisy data."""

        # Forward pass through the model
        recon = self.model(noisy)
        if RESIDUALS:
            # If using residuals, return the noisy input minus the reconstruction
            return noisy - recon
        else:
            # Otherwise, just return the reconstruction
            return recon

    def get_batch_loss(self, batch):
        # Concatenate along batch dimension (assuming shape (B, 1, NBPMS, NTURNS))
        combined_noisy = torch.cat([batch["noisy_x"], batch["noisy_y"]], dim=0)
        combined_clean = torch.cat([batch["clean_x"], batch["clean_y"]], dim=0)
        combined_batch_size = combined_noisy.size(0)

        # Process through the model
        combined_recon = self.reconstruct(combined_noisy)
        if self.need_residual:
            loss = self.loss_fn(combined_noisy, combined_clean, combined_recon)
        else:
            loss = self.loss_fn(combined_recon, combined_clean)

        return loss, combined_batch_size

    def training_step(self, batch, batch_idx):
        loss, combined_batch_size = self.get_batch_loss(batch)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=combined_batch_size,
        )
        self.log("lr", current_lr, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, combined_batch_size = self.get_batch_loss(batch)
        self.log("val_loss", loss, batch_size=combined_batch_size)
        return loss

    def configure_optimizers(self):
        optimiser = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if SCHEDULER:
            # The scheduler is step-based.
            # NUM_DECAY_EPOCHS is used for N here, assuming it represents
            # the number of steps for the decay cycle.
            # You might need to adjust this based on your training setup.
            # For example, N = NUM_DECAY_EPOCHS * len(self.trainer.datamodule.train_dataloader())
            # However, len of dataloader is not available at init.
            # A large number for N is a safe bet if you are unsure.
            # The user note on resuming from checkpoint is handled by PyTorch Lightning
            # automatically when `trainer.fit(ckpt_path=...)` is used, as it restores
            # the scheduler's state, including `last_epoch`.
            scheduler = HalvingCosineLR(
                optimiser,
                a=LEARNING_RATE,
                b=MIN_LR,
                n=NUM_DECAY_EPOCHS,
            )
            return {
                "optimizer": optimiser,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "lr",
                },
            }
        else:
            return {"optimizer": optimiser}


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
    elif MODEL_TYPE == "unet_modified":
        return ModifiedUNetFixed()
    elif MODEL_TYPE == "fno":
        return FNO2d()
    elif MODEL_TYPE == "gated_unet":
        return GatedUNetAutoencoder()
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
