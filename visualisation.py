import matplotlib.pyplot as plt
import numpy as np
import torch
import turn_by_turn as tbt
from torchviz import make_dot

from config import CONFIG_NAME, NBPMS, NTURNS, NUM_CHANNELS, PLOT_DIR
from fft_processing import calculate_fft_and_amps

COLOURS = [
    "#0072B2",  # Blue
    "#D55E00",  # Red
    "#009E73",  # Green
    "#56B4E9",  # Sky Blue
    "#E69F00",  # Orange
    "#F0E442",  # Yellow
    "#CC79A7",  # Pink
    "#000000",  # Black
]
plt.rcParams.update({"font.size": 16})

if not PLOT_DIR.exists():
    PLOT_DIR.mkdir(parents=True)


# Function to convert linear ticks to 10^x format
def log_format(y, pos):
    return r"$10^{{{:.0f}}}$".format(y)


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
    plt.savefig(PLOT_DIR / (CONFIG_NAME + filename), bbox_inches="tight", dpi=300)
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
    freqs = freqs[device_index]
    amps = amps[device_index]
    return freqs, amps


def plot_denoised_data(
    sample_dict,
    bpm_index,
    spectrum_id="denoised_spectrum_xy.png",
    tbt_id="tbt_data.png",
    tbt_diff_id="tbt_diff.png",
):
    """
    Plots FFT spectra and turn-by-turn data for both X and Y data at a given BPM index.

    Args:
        denoised (torch.Tensor): Autoencoder cleaned data tensor of shape (2*NBPMS, NTURNS).
        noisy (torch.Tensor): Noisy data tensor.
        nonoise (torch.Tensor): No-noise data tensor.
        bpm_index (int): BPM index (0 to NBPMS-1) for the X plane. The corresponding Y data is at index bpm_index + NBPMS.
    """
    data_dict = {
        "No Noise": {
            "data_x": sample_dict["clean_x"],
            "data_y": sample_dict["clean_y"],
            "color": COLOURS[1],
            "linestyle": "solid",
            "alpha": 1.0,
        },
        "Noisy": {
            "data_x": sample_dict["noisy_x"],
            "data_y": sample_dict["noisy_y"],
            "color": COLOURS[0],
            "linestyle": "dotted",
            "alpha": 0.8,
        },
        "Autoencoder Cleaned": {
            "data_x": sample_dict["recon_x"],
            "data_y": sample_dict["recon_y"],
            "color": COLOURS[6],
            "linestyle": "dashed",
            "alpha": 0.9,
        },
    }

    # Compute FFT for each data type
    fft_data = {}
    for label, props in data_dict.items():
        x_data = props["data_x"]
        y_data = props["data_y"]
        fft_data[label] = {
            "x": get_fft_at_device(x_data, bpm_index),
            "y": get_fft_at_device(y_data, bpm_index),
        }

    # Create subplots for FFT spectra
    fig, axs = plt.subplots(1, 2, figsize=(24, 6))
    for i, plane in enumerate(["X", "Y"]):
        for label, props in data_dict.items():
            freqs, amps = fft_data[label][plane.lower()]
            axs[i].plot(
                freqs,
                amps,
                label=f"{label} {plane}",
                color=props["color"],
                linestyle=props["linestyle"],
                alpha=props["alpha"],
            )
        axs[i].set_xlabel("Frequency")
        axs[i].set_ylabel("Normalized Amplitude (log scale)")
        axs[i].set_title(f"{plane} Plane")
        axs[i].legend()
        axs[i].grid(True, linestyle="--", alpha=0.6)

    plt.savefig(PLOT_DIR / (CONFIG_NAME + spectrum_id), bbox_inches="tight", dpi=300)
    print(f"Denoised spectrum plot saved as {spectrum_id}.")

    # Create subplots for turn-by-turn data
    fig, axs = plt.subplots(1, 2, figsize=(24, 6))
    turn_limit = 50
    for i, plane in enumerate(["X", "Y"]):
        for label, props in data_dict.items():
            data = props[f"data_{plane.lower()}"][bpm_index, :]
            axs[i].plot(
                data * 1e3,  # Convert to mm
                label=f"{label} {plane}",
                color=props["color"],
                linestyle=props["linestyle"],
                alpha=props["alpha"],
                marker="x" if label == "Autoencoder Cleaned" else None,
            )
        axs[i].set_xlabel("Turns")
        axs[i].set_ylabel("Amplitude [mm]")
        axs[i].set_title(f"{plane} Plane")
        axs[i].set_xlim(0, turn_limit)
        axs[i].legend()
        axs[i].grid(True, linestyle="--", alpha=0.6)

    plt.savefig(PLOT_DIR / (CONFIG_NAME + tbt_id), bbox_inches="tight", dpi=300)
    print(f"TBT data plot saved as {tbt_id}.")
    plot_tbt_difference(sample_dict, bpm_index, tbt_diff_id)


def plot_tbt_difference(sample_dict, bpm_index, filename):
    """
    Plots the turn-by-turn (TBT) differences for both X and Y planes.

    Args:
        sample_dict: A dict containing noisy_(x|y), denoised_(x|y), nonoise_(x|y)
        bpm_index (int): BPM index (0 to NBPMS-1) for the X plane. The corresponding Y data is at index bpm_index + NBPMS.
        filename (str): Filename to save the TBT difference plot.
    """
    # Extract turn-by-turn data for X and Y planes
    noisy_x = sample_dict["noisy_x"]
    noisy_y = sample_dict["noisy_y"]
    denoised_x = sample_dict["recon_x"]
    denoised_y = sample_dict["recon_y"]
    nonoise_x = sample_dict["clean_x"]
    nonoise_y = sample_dict["clean_y"]

    # Define data dictionary for TBT differences
    data_dict = {
        "Noisy - No Noise": {
            "data_x": noisy_x - nonoise_x,
            "data_y": noisy_y - nonoise_y,
            "color": COLOURS[0],
            "alpha": 0.8,
        },
        "Autoencoder cleaned - No Noise": {
            "data_x": denoised_x - nonoise_x,
            "data_y": denoised_y - nonoise_y,
            "color": COLOURS[4],
            "alpha": 1,
        },
    }

    # Plot the TBT difference
    fig, axs = plt.subplots(1, 2, figsize=(24, 6))
    for i, plane in enumerate(["X", "Y"]):
        for label, props in data_dict.items():
            data = props["data_x"] if plane == "X" else props["data_y"]
            data = data[bpm_index, :]
            axs[i].plot(
                range(len(data)),
                data * 1e3,  # Convert to mm
                label=label,
                color=props["color"],
                alpha=props["alpha"],
            )
        axs[i].set_xlabel("Turns")
        axs[i].set_ylabel("$\delta$ Amplitude [mm]")
        axs[i].set_title(f"{plane} Plane")
        axs[i].legend()
        axs[i].grid(True, linestyle="--", alpha=0.6)

    # Format axes for scientific notation
    for ax in axs:
        # ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        ax.xaxis.major.formatter._useMathText = True
        ax.yaxis.major.formatter._useMathText = True

    # plt.tight_layout()
    plt.savefig(PLOT_DIR / (CONFIG_NAME + filename), bbox_inches="tight", dpi=300)
    print(f"TBT difference plot saved as {filename}.")


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

    # plt.plot(noisy, label="Noisy", alpha=0.8)
    # plt.plot(clean, label="No Noise", linestyle="dashed", alpha=0.7)
    plt.plot(noisy - clean, label="Noisy - No Noise")
    plt.xlabel("Turns")
    plt.ylabel("Amplitude")
    plt.title("X Plane")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_model_architecture(
    model, input_size=(NUM_CHANNELS, NBPMS, NTURNS), filename="model_architecture"
):
    """
    Generates a .png file showing the model architecture using torchviz.

    Args:
        model: Your PyTorch model.
        input_size (tuple): Shape of a single input sample, excluding batch size.
        filename (str): The desired filename (without extension) for the saved plot.

    Returns:
        None (it saves a PNG file of the model graph to your working directory).
    """
    # Create a dummy input tensor with batch size = 1
    dummy_input = torch.randn(1, *input_size)

    # Forward pass to build the computation graph
    output = model(dummy_input)

    # Use make_dot to create the graph
    dot_graph = make_dot(output, params=dict(model.named_parameters()))

    # Save the graph to a .png file
    dot_graph.format = "png"
    dot_graph.render(filename, cleanup=True)
    print(f"Model architecture diagram saved as {filename}.png")


def load_tbt_tensor(tbt_file_path):
    """
    Given a path to a TBT file, load the TBT data and return a torch.Tensor
    of shape (2, NBPMS, NTURNS) with the data β-scaled.

    Args:
        tbt_file_path (str or Path): Path to the TBT file.

    Returns:
        torch.Tensor: Tensor containing the X and Y data.
    """
    # Read the TBT file
    tbt_data = tbt.read_tbt(tbt_file_path)
    x_data = tbt_data.matrices[0].X.to_numpy()
    y_data = tbt_data.matrices[0].Y.to_numpy()
    return x_data, y_data


def plot_denoised_data_from_files(
    recon_path,
    noisy_path,
    nonoise_path,
    bpm_index,
    spectrum_id="denoised_spectrum_xy_from_tbt.png",
    tbt_id="tbt_data_from_tbt.png",
    tbt_diff_id="tbt_diff_from_tbt.png",
):
    """
    Loads TBT data from the provided file paths and plots FFT spectra and
    turn-by-turn data for both X and Y data at a given BPM index.

    This function behaves like the in-memory version but accepts file paths.

    Args:
        denoised_path (str or Path): Path to the denoised (autoencoder cleaned) TBT file.
        noisy_path (str or Path): Path to the noisy TBT file.
        nonoise_path (str or Path): Path to the no-noise TBT file.
        bpm_index (int): BPM index (0 to NBPMS-1) for the X plane. The corresponding Y data is at bpm_index + NBPMS.
        spectrum_id (str): Filename to save the FFT spectrum plot.
        tbt_id (str): Filename to save the turn-by-turn data plot.
    """

    # Load the TBT data from disk
    denoised_x, denoised_y = load_tbt_tensor(recon_path)
    noisy_x, noisy_y = load_tbt_tensor(noisy_path)
    nonoise_x, nonoise_y = load_tbt_tensor(nonoise_path)

    # Create a dictionary to hold the data
    sample_dict = {
        "recon_x": denoised_x,
        "recon_y": denoised_y,
        "noisy_x": noisy_x,
        "noisy_y": noisy_y,
        "clean_x": nonoise_x,
        "clean_y": nonoise_y,
    }

    # Call the original in-memory plotting function
    plot_denoised_data(sample_dict, bpm_index, spectrum_id, tbt_id, tbt_diff_id)
