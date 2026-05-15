import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging


def set_seed(seed: int = 42) -> None:
    """
    Seeds all RNGs for reproducibility.

    cudnn.deterministic disables non-deterministic CUDA kernels;
    cudnn.benchmark is turned off because it picks the fastest algorithm
    per input size, which can vary across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Seed set to: {seed}")


def save_history(history: list[dict], filename: str) -> None:
    """Saves the per-epoch training history to results/<filename>.csv."""
    os.makedirs("results", exist_ok=True)
    path = f"results/{filename}.csv"
    pd.DataFrame(history).to_csv(path, index=False)
    logging.info(f"Training history saved to: {path}")


def plot_training(history: list[dict], model_name: str) -> None:
    """Saves loss and AUC curves to plots/<model_name>_curves.png."""
    os.makedirs("plots", exist_ok=True)
    df = pd.DataFrame(history)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["loss"],     label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss",   marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss — {model_name}")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["val_auc"], label="Val AUC", color="orange", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"AUC — {model_name}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_curves.png")
    plt.close()
