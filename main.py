import os
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from lhcng.config import PLOT_DIR
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger

# Import configurations and utilities
from config import (
    ACCUMULATE_BATCHES,
    CONFIG_NAME,
    LEARNING_RATE,
    LOAD_MODEL,
    LOSS_TYPE,
    MODEL_SAVE_PATH,
    NLOGSTEPS,
    NUM_EPOCHS,
    PRECISION,
    RESUME_FROM_CKPT,
    SEED,
    WEIGHT_DECAY,
    print_config,
    save_experiment_config,
)
from dataloader import denormalise_sample_dict, load_data, save_global_norm_params
from pl_module import (
    LitAutoencoder,
    find_newest_file,
    get_model,
)

if __name__ == "__main__":
    print_config()
    if (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    ):  # Check if MPS is built with PyTorch
        print("MPS is available and built.")
    else:
        free, available = torch.cuda.mem_get_info()
        print("Current GPU use:", (available - free) / 1e9, "GB")
        torch.set_float32_matmul_precision("medium")

    # Set up reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    print("Loading data...")
    b4_load = time.time()
    train_loader, val_loader, dataset = load_data()
    print(f"Data loaded. Took {time.time() - b4_load:.2f} seconds.")

    # Visualise data distribution from one batch
    # batch = next(iter(train_loader))
    # plot_data_distribution(batch["noisy"], "Noisy Data Distribution")
    # plot_data_distribution(batch["clean"], "Clean Data Distribution")
    # plot_noisy_data(batch["noisy"][0, 0, :, :], batch["clean"][0, 0, :, :], 111)

    model = get_model()

    if LOAD_MODEL and os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("Loaded pre-trained model.")
    else:
        print("Training new model...")
        num_cpu = min(os.cpu_count() or 1, 32)
        torch.set_num_threads(num_cpu)
        print(f"Using {num_cpu} CPUs for training.")

        lit_model = LitAutoencoder(
            model,
            loss_type=LOSS_TYPE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        if torch.backends.mps.is_available():
            root_dir = Path("/Users/joshuagray/Documents/autocleaner")
        else:
            root_dir = Path("/home/jovyan/")
        log_dir = root_dir / "tensor-logs/"
        logger = TensorBoardLogger(root_dir, name=log_dir.name, version=CONFIG_NAME)
        save_experiment_config(log_dir)

        # torch.cuda.empty_cache()
        b4_train = time.time()
        trainer = pl.Trainer(
            accumulate_grad_batches=ACCUMULATE_BATCHES,
            max_epochs=NUM_EPOCHS,
            log_every_n_steps=NLOGSTEPS,
            default_root_dir=root_dir,  # Set the logging path
            logger=logger,
            precision=PRECISION,
            # callbacks=[noise_annealing_callback],
            # max_time={"hours": 7.5},
            # enable_checkpointing=False,  # Disables automatic checkpoint saving
        )
        if RESUME_FROM_CKPT:
            ckpt_fldr = log_dir / CONFIG_NAME / "checkpoints"
            ckpt_file = find_newest_file(ckpt_fldr)
            ckpt_path = ckpt_fldr / ckpt_file
        else:
            ckpt_path = None
        trainer.fit(lit_model, train_loader, val_loader, ckpt_path=ckpt_path)
        print(f"Training took {time.time() - b4_train:.2f} seconds.")

        # Save the model
        b4_save = time.time()
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved. Took {time.time() - b4_save:.2f} seconds.")

        # Save global normalization parameters
        save_global_norm_params(dataset, filepath="global_norm_params.json")

    # Denoise validation data
    print("Denoising validation data...")
    b4_denoise = time.time()

    batch = next(iter(val_loader))
    noisy_batch_x = batch["noisy_x"]  # shape: (B, 1, NBPMS, NTURNS)
    noisy_batch_y = batch["noisy_y"]  # shape: (B, 1, NBPMS, NTURNS)
    combined_noisy = torch.cat([batch["noisy_x"], batch["noisy_y"]], dim=0)

    with torch.no_grad():
        combined_recon = lit_model.reconstruct(combined_noisy)
        
    recon_x, recon_y = torch.chunk(combined_recon, 2, dim=0)

    assert (
        recon_x.size(0)
        == noisy_batch_x.size(0)
        == noisy_batch_y.size(0)
        == recon_y.size(0)
    )
    sample = {
        "noisy_x": noisy_batch_x[0, 0, ...].numpy(),
        "noisy_y": noisy_batch_y[0, 0, ...].numpy(),
        "recon_x": recon_x[0, 0, ...].numpy(),
        "recon_y": recon_y[0, 0, ...].numpy(),
        "clean_x": batch["clean_x"][0, 0, ...].numpy(),
        "clean_y": batch["clean_y"][0, 0, ...].numpy(),
        "norm_info": batch["norm_info"][0],  # Use the norm_info from the batch
    }

    sample_dict = denormalise_sample_dict(sample, dataset)

    device_index = 111  # Select a BPM to plot the FFT spectrum
    print(f"Denoised Data for Device {device_index}")

    # Save the experiment configuration
    save_experiment_config(PLOT_DIR)

    # Plot the denoised data
    from visualisation import (
        plot_denoised_data,
    )

    plot_denoised_data(sample_dict, device_index)

    plt.show()
