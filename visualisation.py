import matplotlib.pyplot as plt
from fft_processing import calculate_fft_and_amps
from config import CONFIG_NAME, NBPMS, PLOT_DIR, NTURNS

if not PLOT_DIR.exists():
    PLOT_DIR.mkdir(parents=True)


def plot_loss(train_loss_values, val_loss_values, filename="training_loss.png"):
    """
    Plots the training and validation loss over epochs.

    Args:
        train_loss_values (list): List of training loss values per epoch.
        val_loss_values (list): List of validation loss values per epoch.
        filename (str): Filename to save the loss plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_loss_values)), train_loss_values, label="Training Loss")
    plt.plot(range(len(val_loss_values)), val_loss_values, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(PLOT_DIR / (CONFIG_NAME + filename))
    print(f"Loss plot saved as {filename}.")


def update_live_plot(
    epoch, train_loss_values, val_loss_values, ax, train_loss_line, val_loss_line
):
    """
    Updates the live plot for training and validation loss.

    Args:
        epoch (int): Current epoch number.
        train_loss_values (list): List of training loss values.
        val_loss_values (list): List of validation loss values.
        ax (matplotlib axis): The axis object for the plot.
        train_loss_line (matplotlib line): Training loss line plot object.
        val_loss_line (matplotlib line): Validation loss line plot object.
    """
    if epoch % 10 == 0:
        train_loss_line.set_data(range(len(train_loss_values)), train_loss_values)
        val_loss_line.set_data(range(len(val_loss_values)), val_loss_values)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)


def get_fft_at_device(tbt_data, device_index):
    freqs, amps = calculate_fft_and_amps(tbt_data)
    freqs = freqs[device_index].detach().numpy()
    amps = amps[device_index].detach().numpy()
    return freqs, amps


def plot_denoised_data(
    denoised,
    noisy,
    clean,
    bpm_index,
    spectrum_id="denoised_spectrum_xy.png",
    tbt_id="tbt_data.png",
):
    """
    Plots FFT spectra for both X and Y data at a given BPM index.

    Args:
        denoised (torch.Tensor): Denoised data tensor of shape (2*NBPMS, NTURNS).
        noisy (torch.Tensor): Noisy data tensor.
        clean (torch.Tensor): Clean data tensor.
        bpm_index (int): BPM index (0 to NBPMS-1) for the X plane. The corresponding Y data is at index bpm_index + NBPMS.
    """
    # Separate channels for X and Y
    assert denoised.shape == (2, NBPMS, NTURNS)
    assert noisy.shape == (2, NBPMS, NTURNS)
    assert clean.shape == (2, NBPMS, NTURNS)

    denoised_x = denoised[0, bpm_index, :]
    denoised_y = denoised[1, bpm_index, :]
    noisy_x = noisy[0, bpm_index, :]
    noisy_y = noisy[1, bpm_index, :]
    clean_x = clean[0, bpm_index, :]
    clean_y = clean[1, bpm_index, :]

    # Compute FFT (using your existing helper, for example)
    denoised_freqs_x, denoised_amps_x = get_fft_at_device(denoised_x.unsqueeze(0), 0)
    denoised_freqs_y, denoised_amps_y = get_fft_at_device(denoised_y.unsqueeze(0), 0)
    noisy_freqs_x, noisy_amps_x = get_fft_at_device(noisy_x.unsqueeze(0), 0)
    noisy_freqs_y, noisy_amps_y = get_fft_at_device(noisy_y.unsqueeze(0), 0)
    clean_freqs_x, clean_amps_x = get_fft_at_device(clean_x.unsqueeze(0), 0)
    clean_freqs_y, clean_amps_y = get_fft_at_device(clean_y.unsqueeze(0), 0)

    # Create subplots for X and Y
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(denoised_freqs_x, denoised_amps_x, label="Denoised X", alpha=0.8)
    axs[0].plot(
        noisy_freqs_x, noisy_amps_x, label="Noisy X", linestyle="dotted", alpha=0.7
    )
    axs[0].plot(
        clean_freqs_x, clean_amps_x, label="Clean X", linestyle="dashed", alpha=0.7
    )
    axs[0].set_xlabel("Frequency")
    axs[0].set_ylabel("Normalized Amplitude (log scale)")
    axs[0].set_title("X Plane")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.6)

    axs[1].plot(denoised_freqs_y, denoised_amps_y, label="Denoised Y", alpha=0.8)
    axs[1].plot(
        noisy_freqs_y, noisy_amps_y, label="Noisy Y", linestyle="dotted", alpha=0.7
    )
    axs[1].plot(
        clean_freqs_y, clean_amps_y, label="Clean Y", linestyle="dashed", alpha=0.7
    )
    axs[1].set_xlabel("Frequency")
    axs[1].set_ylabel("Normalized Amplitude (log scale)")
    axs[1].set_title("Y Plane")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / (CONFIG_NAME + spectrum_id))
    print(f"Denoised spectrum plot saved as {spectrum_id}.")

    # Also plot the tbt data for the selected device
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(denoised_x, label="Denoised X", alpha=0.8)
    axs[0].plot(noisy_x, label="Noisy X", linestyle="dotted", alpha=0.7)
    axs[0].plot(clean_x, label="Clean X", linestyle="dashed", alpha=0.7)
    axs[0].set_xlabel("Turns")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("X Plane")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.6)

    axs[1].plot(denoised_y, label="Denoised Y", alpha=0.8)
    axs[1].plot(noisy_y, label="Noisy Y", linestyle="dotted", alpha=0.7)
    axs[1].plot(clean_y, label="Clean Y", linestyle="dashed", alpha=0.7)
    axs[1].set_xlabel("Turns")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_title("Y Plane")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / (CONFIG_NAME + tbt_id))
    print(f"TBT data plot saved as {tbt_id}.")
    # plt.show()


def plot_data_distribution(data, title="Data Distribution"):
    plt.figure(figsize=(8, 5))
    plt.hist(data.flatten().detach().cpu().numpy(), bins=50, density=True, alpha=0.75)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

def plot_noisy_data(noisy, clean, bpm_index):
    """
    Plots the noisy and clean data for a given BPM index.

    Args:
        noisy (torch.Tensor): Noisy data tensor of shape (2*NBPMS, NTURNS).
        clean (torch.Tensor): Clean data tensor.
        bpm_index (int): BPM index (0 to NBPMS-1) for the X plane. The corresponding Y data is at index bpm_index + NBPMS.
    """
    # Separate channels for X and Y
    noisy = noisy[bpm_index, :]
    clean = clean[bpm_index, :]

    # Create subplots for X and Y
    plt.plot(noisy, label="Noisy", alpha=0.8)
    plt.plot(clean, label="Clean", linestyle="dashed", alpha=0.7)
    plt.xlabel("Turns")
    plt.ylabel("Amplitude")
    plt.title("X Plane")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
