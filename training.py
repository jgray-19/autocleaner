# training.py
import torch
import torch.nn as nn
import torch.optim as optim

# Import new config weight
from config import (
    BPM_DIFF_WEIGHT,
    LEARNING_RATE,
    NUM_EPOCHS,
    TURN_DIFF_WEIGHT,
    WEIGHT_DECAY,
)

# Existing imports
from losses import combined_mse_correlation_loss  # , MSLELoss
from visualisation import plot_loss

combined_loss_fn = combined_mse_correlation_loss
intra_bpm_loss_fn = nn.MSELoss()
intra_turn_loss_fn = nn.MSELoss()


def loss_calculation(
    model: nn.Module,
    noisy: torch.Tensor,
    clean: torch.Tensor,
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


def train(model, train_loader, optimizer):
    model.train()
    epoch_loss = 0

    for noisy, clean in train_loader:
        optimizer.zero_grad()
        loss = loss_calculation(model, noisy, clean)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def validate(model, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            val_loss += loss_calculation(model, noisy, clean).item()
    return val_loss / len(val_loader)


def train_model(model, train_loader, val_loader):
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    train_loss_values = []
    val_loss_values = []

    try:
        for epoch in range(NUM_EPOCHS):
            epoch_loss = train(model, train_loader, optimizer)
            val_loss = validate(model, val_loader)
            train_loss_values.append(epoch_loss)
            val_loss_values.append(val_loss)

            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - "
                f"Train Loss: {epoch_loss:.4e}, "
                f"Val Loss: {val_loss:.4e}"
            )

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save(model.state_dict(), "interrupted_model.pth")
    plot_loss(train_loss_values, val_loss_values)

    return model
