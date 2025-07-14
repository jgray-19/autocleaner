import glob
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import (
    ALPHA,
    LEARNING_RATE,
    MIN_LR,
    MODEL_TYPE,
    NUM_CONSTANT_LR_EPOCHS,
    NUM_DECAY_EPOCHS,
    RESIDUALS,
    SCHEDULER,
    USE_MASK,
)
from losses import (
    CombinedCorrelationLoss,
    CorrelationLoss,
    SSPLoss,
    fft_loss_per_bpm,
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
)


class LitAutoencoder(pl.LightningModule):
    def __init__(self, model, learning_rate, weight_decay, loss_type):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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
            # self.loss_fn = self.combined_ssp_loss
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

    # def combined_ssp_loss(self, pred, target):

    def forward(self, x):
        return self.model(x)

    def get_batch_loss(self, batch):
        # Apply mask or residual logic to get recon_x and recon_y
        if RESIDUALS:
            pred_x = self(batch["noisy_x"])
            pred_y = self(batch["noisy_y"])

            recon_x = batch["noisy_x"] - pred_x
            recon_y = batch["noisy_y"] - pred_y
        elif USE_MASK:
            mask_x = self(batch["noisy_x"])
            mask_y = self(batch["noisy_y"])

            recon_x = mask_x * batch["noisy_x"]
            recon_y = mask_y * batch["noisy_y"]
        else:
            recon_x = self(batch["noisy_x"])
            recon_y = self(batch["noisy_y"])

        # Now compute SNR-dependent blend for both x and y
        loss_total = 0.0
        for recon, clean, noisy in zip(
            [recon_x, recon_y],
            [batch["clean_x"], batch["clean_y"]],
            [batch["noisy_x"], batch["noisy_y"]],
        ):
            loss_mse = F.mse_loss(recon, clean)

            loss_ssp = self.ssp(recon, clean)
            print(f"Loss MSE: {loss_mse:.4f}, Loss SSP: {loss_ssp:.4f}")
            loss_total += ALPHA * loss_mse + (1 - ALPHA) * loss_ssp
        loss = 0.5 * loss_total
        return loss, batch["noisy_x"].size(0) + batch["noisy_y"].size(0)

    def training_step(self, batch, batch_idx):
        loss, combined_batch_size = self.get_batch_loss(batch)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train_loss", loss, batch_size=combined_batch_size)
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
            scheduler1 = optim.lr_scheduler.CosineAnnealingLR(
                optimiser,
                T_max=NUM_CONSTANT_LR_EPOCHS,
                eta_min=(MIN_LR + LEARNING_RATE) / 2,
            )
            scheduler3 = optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=NUM_DECAY_EPOCHS, eta_min=MIN_LR
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimiser,
                schedulers=[scheduler1, scheduler3],
                milestones=[NUM_CONSTANT_LR_EPOCHS],
            )
            return {
                "optimizer": optimiser,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # call .step() every epoch
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
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
