import os
import json
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.utils import save_image, make_grid

from src.setup import get_device, build_dataloaders
from models.variational_autoencoder import BaselineVAE
from models.variational_autoencoder_128 import VAE128


def build_vae_model(model_config):
    model_name = model_config.get("name", "vae64")

    if model_name == "vae64":
        return BaselineVAE(
            img_channels=model_config.get("img_channels", 3),
            feature_maps=model_config.get("feature_maps", 32),
            latent_dim=model_config.get("latent_dim", 128),
        )

    if model_name == "vae128":
        return VAE128(
            img_channels=model_config.get("img_channels", 3),
            feature_maps=model_config.get("feature_maps", 32),
            latent_dim=model_config.get("latent_dim", 256),
        )

    raise ValueError(f"Unknown model name: {model_name}")


def denormalize_vae(imgs):
    return torch.clamp((imgs + 1.0) / 2.0, 0.0, 1.0)


@torch.no_grad()
def extract_latents(model, loader, device, max_batches=20):
    """
    Encodes images and returns the posterior means (mu) as latent representations.

    We use mu rather than a sampled z because mu is the deterministic centre of
    each image's posterior — it gives a stable, noise-free representation that
    is more suitable for visualisation and clustering than a single noisy sample.
    """
    model.eval()

    all_mu, all_labels, all_sources = [], [], []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        imgs = batch["image"].to(device)
        mu, _ = model.encoder(imgs)  # logvar discarded — we only need the mean
        all_mu.append(mu.cpu())

        if "label" in batch:
            all_labels.extend(batch["label"].tolist())
        if "source_type" in batch:
            all_sources.extend(batch["source_type"])

    return torch.cat(all_mu, dim=0).numpy(), all_labels, all_sources


def plot_pca(latents, save_path):
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=8, alpha=0.6)
    plt.title("VAE Latent Space PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


@torch.no_grad()
def save_interpolation(model, loader, device, save_path, num_steps=10):
    """
    Linearly interpolates between two images in latent space and saves the result.

    z = (1 - alpha) * mu_a + alpha * mu_b, alpha in [0, 1].
    A smooth transition indicates a well-structured latent space; abrupt jumps
    suggest the encoder has not learned a continuous manifold.
    """
    model.eval()

    batch = next(iter(loader))
    imgs  = batch["image"].to(device)

    mu_a, _ = model.encoder(imgs[0:1])
    mu_b, _ = model.encoder(imgs[1:2])

    frames = []
    for alpha in torch.linspace(0, 1, num_steps, device=device):
        z = (1 - alpha) * mu_a + alpha * mu_b
        frames.append(model.decoder(z))

    grid = make_grid(denormalize_vae(torch.cat(frames, dim=0)).cpu(), nrow=num_steps)
    save_image(grid, save_path)


@torch.no_grad()
def save_reconstruction_check(model, loader, device, save_path, max_images=8):
    """
    Saves a side-by-side grid of originals (top row) and reconstructions (bottom row).
    Useful for assessing reconstruction quality without any sampling randomness.
    """
    model.eval()

    batch = next(iter(loader))
    imgs  = batch["image"][:max_images].to(device)
    recon, _, _ = model(imgs)

    # Stack originals and reconstructions so they appear as two rows in the grid
    grid = make_grid(
        torch.cat([denormalize_vae(imgs), denormalize_vae(recon)], dim=0).cpu(),
        nrow=max_images,
    )
    save_image(grid, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/latent_analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r") as f:
        config = json.load(f)

    device = get_device()

    loaders = build_dataloaders(
        config["data"],
        config["training"],
        config["preprocessing"],
    )

    model = build_vae_model(config["model"]).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print("Model loaded successfully.")

    latents, labels, sources = extract_latents(
        model=model,
        loader=loaders["val"],
        device=device,
        max_batches=30,
    )

    np.save(os.path.join(args.output_dir, "latents.npy"), latents)

    plot_pca(
        latents,
        save_path=os.path.join(args.output_dir, "latent_pca.png"),
    )

    save_interpolation(
        model=model,
        loader=loaders["val"],
        device=device,
        save_path=os.path.join(args.output_dir, "latent_interpolation.png"),
        num_steps=10,
    )

    save_reconstruction_check(
        model=model,
        loader=loaders["val"],
        device=device,
        save_path=os.path.join(args.output_dir, "reconstruction_check.png"),
    )

    print(f"Saved latent analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()