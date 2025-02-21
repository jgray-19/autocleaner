import pytorch_lightning as pl
import torch.optim as optim
# from losses import combined_mse_correlation_loss
import torch

class LitAutoencoder(pl.LightningModule):
    def __init__(self, model, learning_rate, weight_decay):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # self.loss_fn = combined_mse_correlation_loss
        self.loss_fn = torch.nn.functional.mse_loss

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
