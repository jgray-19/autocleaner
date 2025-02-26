import pytorch_lightning as pl
import torch.optim as optim
from losses import CombinedMSECorrelationLoss, CorrelationLoss
import torch

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
            self.loss_fn = CombinedMSECorrelationLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        noisy = batch["noisy"]
        clean = batch["clean"]
        reconstructed = self(noisy)
        loss = self.loss_fn(reconstructed, clean)
        
        self.log("train_loss", loss, batch_size=noisy.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        noisy = batch["noisy"]
        clean = batch["clean"]
        reconstructed = self(noisy)
        loss = self.loss_fn(reconstructed, clean)
        
        self.log("val_loss", loss, batch_size=noisy.size(0))
        return loss


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
