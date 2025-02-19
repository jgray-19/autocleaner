# training.py
from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import new config weight
from config import (
    BPM_DIFF_WEIGHT,
    LEARNING_RATE,
    NUM_EPOCHS,
    TURN_DIFF_WEIGHT,
    WEIGHT_DECAY,
)
from dataloader import BPMSDataset
from fft_processing import calculate_fft_and_amps

# Existing imports
from losses import CombinedTimeFreqLoss  # , DedicatedMSLELoss, SafeMSLELoss
from visualisation import plot_loss, update_live_plot


def loss_calculation(
    model: nn.Module,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    combined_loss_fn: Callable,
    intra_bpm_loss_fn: Callable,
    intra_turn_loss_fn: Callable,
):
    """
    Computes a hybrid loss:
      - Time-domain loss (MSLE + BPM difference losses)
      - Frequency-domain loss via MSE of log(FFT amplitudes)
    """
    # 1) Forward pass
    reconstructed = model(noisy)  # shape: (batch, BPMs, NTURNS)

    # 2) Flatten BEFORE subtraction (batch, BPMs * NTURNS)
    clean_flat = clean.reshape(clean.shape[0], -1)
    rec_flat = reconstructed.reshape(reconstructed.shape[0], -1)

    # 3) Compute intra-BPM difference loss
    diff_bpm_clean = clean_flat[:, 1:] - clean_flat[:, :-1]  # Shifted differences
    diff_bpm_rec = rec_flat[:, 1:] - rec_flat[:, :-1]
    intra_bpm_loss = intra_bpm_loss_fn(diff_bpm_rec, diff_bpm_clean)

    # 4) Inter-turn smoothness loss
    diff_turn_clean = clean[:, :, 1:] - clean[:, :, :-1]  # (batch, BPMs, NTURNS-1)
    diff_turn_rec = reconstructed[:, :, 1:] - reconstructed[:, :, :-1]
    intra_turn_loss = intra_turn_loss_fn(diff_turn_rec, diff_turn_clean)

    # Weighted sum with diff_loss_intra
    tbt_fft_loss = combined_loss_fn(reconstructed, clean)
    combined = (
        tbt_fft_loss
        + BPM_DIFF_WEIGHT * intra_bpm_loss
        + TURN_DIFF_WEIGHT * intra_turn_loss
    )
    return combined


def train(model, train_loader, optimizer, loss_fn):
    model.train()
    epoch_loss = 0

    for noisy, clean in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model, noisy, clean)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def validate(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            val_loss += loss_fn(model, noisy, clean).item()
    return val_loss / len(val_loader)


def train_model(model, train_loader, val_loader, dataset: BPMSDataset):
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_loss_values = []
    val_loss_values = []

    combined_loss_fn = CombinedTimeFreqLoss(dataset.clean_data_norm)

    # clean_flat = dataset.clean_data_norm.reshape(dataset.clean_data_norm.shape[0], -1)
    # diff_bpm_clean = clean_flat[:, 1:] - clean_flat[:, :-1]
    # bpm_diff_loss_fn = nn.MSELoss()
    # def intra_bpm_loss_fn(diff_bpm_rec):
        # diff_bpm_clean = 
        # return bpm_diff_loss_fn(diff_bpm_rec, diff_bpm_clean)
        
    
    intra_bpm_loss_fn = nn.MSELoss()
    intra_turn_loss_fn = nn.MSELoss()

    def loss_fn(model, noisy, clean):
        return loss_calculation(
            model, noisy, clean, combined_loss_fn, intra_bpm_loss_fn, intra_turn_loss_fn
        )

    try:
        for epoch in range(NUM_EPOCHS):
            epoch_loss = train(model, train_loader, optimizer, loss_fn)
            val_loss = validate(model, val_loader, loss_fn)
            train_loss_values.append(epoch_loss)
            val_loss_values.append(val_loss)

            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - "
                f"Train Loss: {train_loss_values[-1]:.4g}, "
                f"Val Loss: {val_loss_values[-1]:.4g}"
            )

        # Save final loss plot
        plot_loss(train_loss_values, val_loss_values)

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save(model.state_dict(), "interrupted_model.pth")

    return model
