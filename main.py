import os
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from lhcng.config import PLOT_DIR
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger

from config import (
    ACCUMULATE_BATCHES,
    CONFIG_NAME,
    LEARNING_RATE,
    LOAD_MODEL,
    LOSS_TYPE,
    MODEL_SAVE_PATH,
    NLOGSTEPS,
    NUM_EPOCHS,
    RESIDUALS,
    RESUME_FROM_CKPT,
    WEIGHT_DECAY,
    print_config,
    save_experiment_config,
)
from dataloader import build_sample_dict, load_data
from pl_module import LitAutoencoder, find_newest_file, get_model
from visualisation import plot_denoised_data


def main() -> None:
    print_config()
    if torch.cuda.is_available():
        free, available = torch.cuda.mem_get_info()
        print("Current GPU use:", (available - free) / 1e9, "GB")
    else:
        print("CUDA not available. Running on CPU.")

    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading data...")
    b4_load = time.time()
    train_loader, val_loader, dataset = load_data()
    print(f"Data loaded. Took {time.time() - b4_load:.2f} seconds.")

    model = get_model()

    if LOAD_MODEL and os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
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

        root_dir = Path.cwd()
        log_dir = root_dir / "tensor-logs/"
        logger = TensorBoardLogger(root_dir, name=log_dir.name, version=CONFIG_NAME)
        if not RESUME_FROM_CKPT:
            save_experiment_config(log_dir)

        b4_train = time.time()
        trainer = pl.Trainer(
            accumulate_grad_batches=ACCUMULATE_BATCHES,
            max_epochs=NUM_EPOCHS,
            log_every_n_steps=NLOGSTEPS,
            default_root_dir=root_dir,
            logger=logger,
        )
        if RESUME_FROM_CKPT:
            ckpt_fldr = log_dir / CONFIG_NAME / "checkpoints"
            ckpt_path = Path(find_newest_file(ckpt_fldr))
        else:
            ckpt_path = None
        trainer.fit(lit_model, train_loader, val_loader, ckpt_path=ckpt_path)
        print(f"Training took {time.time() - b4_train:.2f} seconds.")

        b4_save = time.time()
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved. Took {time.time() - b4_save:.2f} seconds.")

    print("Denoising validation data...")
    b4_denoise = time.time()

    batch = next(iter(val_loader))
    inference_device = next(model.parameters()).device
    noisy_batch_x = batch["noisy_x"].to(inference_device)
    noisy_batch_y = batch["noisy_y"].to(inference_device)

    model.eval()
    with torch.no_grad():
        if RESIDUALS:
            recon_x = noisy_batch_x - model(noisy_batch_x)
            recon_y = noisy_batch_y - model(noisy_batch_y)
        else:
            recon_x = model(noisy_batch_x)
            recon_y = model(noisy_batch_y)

    assert (
        recon_x.size(0)
        == noisy_batch_x.size(0)
        == noisy_batch_y.size(0)
        == recon_y.size(0)
    )
    sample = {
        "noisy_x": noisy_batch_x[0, 0, ...].cpu().numpy(),
        "noisy_y": noisy_batch_y[0, 0, ...].cpu().numpy(),
        "recon_x": recon_x[0, 0, ...].cpu().numpy(),
        "recon_y": recon_y[0, 0, ...].cpu().numpy(),
        "clean_x": batch["clean_x"][0, 0, ...].cpu().numpy(),
        "clean_y": batch["clean_y"][0, 0, ...].cpu().numpy(),
        "source_idx": batch["source_idx"][0].item(),
    }

    sample_dict = build_sample_dict(sample, dataset)

    print(f"Denoising took {time.time() - b4_denoise:.2f} seconds.")

    device_index = 111
    print(f"Denoised Data for Device {device_index}")

    save_experiment_config(PLOT_DIR)
    plot_denoised_data(sample_dict, device_index)
    plt.show()


if __name__ == "__main__":
    main()
