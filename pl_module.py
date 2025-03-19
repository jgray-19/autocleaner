import pytorch_lightning as pl
import torch
import torch.optim as optim

from config import NUM_EPOCHS, SCHEDULER, ALPHA, MIN_LR, RESIDUALS
from losses import CombinedCorrelationLoss, CorrelationLoss, fft_loss_per_bpm, SSPLoss

import os
import glob


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
        mse_loss = torch.nn.functional.mse_loss(pred, target)
        return ssp_loss + mse_loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        noisy = batch["noisy"]
        clean = batch["clean"]
        reconstructed = self(noisy)
        if RESIDUALS:
            loss = self.loss_fn(noisy-reconstructed, clean)
        else:
            loss = self.loss_fn(reconstructed, clean)

        self.log("train_loss", loss, batch_size=noisy.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        noisy = batch["noisy"]
        clean = batch["clean"]
        reconstructed = self(noisy)
        if RESIDUALS:
            loss = self.loss_fn(noisy-reconstructed, clean)
        else:
            loss = self.loss_fn(reconstructed, clean)


        self.log("val_loss", loss, batch_size=noisy.size(0))
        return loss


    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if SCHEDULER:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=MIN_LR, T_max=NUM_EPOCHS)
            return [optimizer], [scheduler]
        return optimizer

def find_newest_file(directory_path):
    # Get a list of all files in the directory
    files = glob.glob(os.path.join(directory_path, '*.ckpt'))
    
    # Find the newest file
    return max(files, key=os.path.getctime)