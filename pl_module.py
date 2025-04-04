import glob
import os

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import Callback

from config import (
    ALPHA,
    MIN_LR,
    MODEL_TYPE,
    NUM_EPOCHS,
    RESIDUALS,
    SCHEDULER,
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
    UNetAutoencoder,
    UNetAutoencoderFixedDepth,
    UNetAutoencoderFixedDepthCheckpoint,
)


class NoiseAnnealingCallback(Callback): 
    def __init__(self, initial_multiplier=1.0, final_multiplier=0.1, max_epochs=NUM_EPOCHS): 
        super().__init__()
        self.initial_multiplier = initial_multiplier 
        self.final_multiplier = final_multiplier 
        self.max_epochs = max_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        # Compute a new multiplier (for example, a linear schedule)
        current_epoch = trainer.current_epoch
        new_multiplier = self.initial_multiplier - (
            (self.initial_multiplier - self.final_multiplier) * (current_epoch / self.max_epochs)
        )

        # Update the dataset's noise multiplier.
        train_dataloader_subset = trainer.train_dataloader.dataset
        train_dataloader_subset.dataset.update_noise_multiplier(new_multiplier)
        pl_module.log("noise_multiplier", new_multiplier)


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
            self.loss_fn = self.combined_ssp_loss
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
        return ssp_loss + mse_loss

    def forward(self, x):
        return self.model(x)

    def get_batch_loss(self, batch):
        # Concatenate along batch dimension (assuming shape (B, 1, NBPMS, NTURNS))
        combined_noisy = torch.cat([batch["noisy_x"], batch["noisy_y"]], dim=0)
        combined_clean = torch.cat([batch["clean_x"], batch["clean_y"]], dim=0)
        combined_batch_size = combined_noisy.size(0)

        # Optionally shuffle the combined batch here (if your DataLoader doesn’t already shuffle individual samples)
        perm = torch.randperm(combined_batch_size)
        combined_noisy = combined_noisy[perm]
        combined_clean = combined_clean[perm]

        # Process through the model
        combined_recon = self(combined_noisy)

        if RESIDUALS:
            loss = self.loss_fn(combined_noisy - combined_recon, combined_clean)
        else:
            loss = self.loss_fn(combined_recon, combined_clean)

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
