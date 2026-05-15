import os
import json
import argparse
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.utils import save_image, make_grid

from src.setup import get_device, build_dataloaders

from models.variational_autoencoder import BaselineVAE
from models.variational_autoencoder_128 import VAE128
from models.variational_autoencoder_v2 import BaselineVAEv2
from models.variational_autoencoder_128_v2 import VAE128v2


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


# -------------------------------------------------------------------
# Model factory
# -------------------------------------------------------------------

def build_vae_model(model_config):
    model_name = model_config.get("architecture", "vae64")

    if model_name == "4_layers":
        return BaselineVAE(
            img_channels=model_config.get("img_channels", 3),
            feature_maps=model_config.get("feature_maps", 32),
            latent_dim=model_config.get("latent_dim", 128),
        )

    if model_name == "5_layers":
        return VAE128(
            img_channels=model_config.get("img_channels", 3),
            feature_maps=model_config.get("feature_maps", 32),
            latent_dim=model_config.get("latent_dim", 256),
        )

    if model_name == "4_layers_v2":
        return BaselineVAEv2(
            img_channels=model_config.get("img_channels", 3),
            feature_maps=model_config.get("feature_maps", 32),
            latent_dim=model_config.get("latent_dim", 128),
        )

    if model_name == "5_layers_v2":
        return VAE128v2(
            img_channels=model_config.get("img_channels", 3),
            feature_maps=model_config.get("feature_maps", 32),
            latent_dim=model_config.get("latent_dim", 256),
        )

    raise ValueError(f"Unknown VAE model name: {model_name}")


# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------

def denormalize_vae(imgs):
    return torch.clamp((imgs + 1.0) / 2.0, 0.0, 1.0)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", None)
        best_val_loss = checkpoint.get("best_val_loss", None)
    else:
        model.load_state_dict(checkpoint)
        epoch = None
        best_val_loss = None

    return epoch, best_val_loss


# -------------------------------------------------------------------
# Evaluation outputs
# -------------------------------------------------------------------

@torch.no_grad()
def save_test_reconstructions(model, loader, device, save_path, max_images=8):
    model.eval()

    batch = next(iter(loader))
    imgs = batch["image"][:max_images].to(device)

    recon_imgs, _, _ = model(imgs)

    imgs = denormalize_vae(imgs)
    recon_imgs = denormalize_vae(recon_imgs)

    comparison = torch.cat([imgs, recon_imgs], dim=0)
    grid = make_grid(comparison.cpu(), nrow=max_images)

    save_image(grid, save_path)


@torch.no_grad()
def save_generated_samples(model, device, save_path, num_samples=16):
    model.eval()

    samples = model.generate(num_samples=num_samples, device=device)
    samples = denormalize_vae(samples)

    grid = make_grid(samples.cpu(), nrow=4)
    save_image(grid, save_path)


@torch.no_grad()
def save_latent_interpolation(model, loader, device, save_path, num_steps=10):
    model.eval()

    batch = next(iter(loader))
    imgs = batch["image"].to(device)

    img_a = imgs[0:1]
    img_b = imgs[1:2]

    mu_a, _ = model.encoder(img_a)
    mu_b, _ = model.encoder(img_b)

    interpolated = []

    for alpha in torch.linspace(0, 1, num_steps, device=device):
        z = (1 - alpha) * mu_a + alpha * mu_b
        decoded = model.decoder(z)
        interpolated.append(decoded)

    interpolated = torch.cat(interpolated, dim=0)
    interpolated = denormalize_vae(interpolated)

    grid = make_grid(interpolated.cpu(), nrow=num_steps)
    save_image(grid, save_path)


@torch.no_grad()
def extract_latents(model, loader, device, max_batches=30):
    model.eval()

    latents = []
    labels = []
    sources = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        imgs = batch["image"].to(device)
        mu, _ = model.encoder(imgs)

        latents.append(mu.cpu())

        if "label" in batch:
            labels.extend(batch["label"].tolist())

        if "source_type" in batch:
            sources.extend(batch["source_type"])

    latents = torch.cat(latents, dim=0).numpy()

    return latents, labels, sources


def save_latent_pca(latents, save_path):
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=8, alpha=0.6)
    plt.title("VAE Latent Space PCA - Test Set")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_summary(config, checkpoint_path, output_dir, epoch, best_val_loss):
    summary_path = os.path.join(output_dir, "evaluation_summary.txt")

    model_config = config["model"]
    preproc_config = config["preprocessing"]

    with open(summary_path, "w") as f:
        f.write("VAE Evaluation Summary\n")
        f.write("======================\n\n")

        f.write(f"Experiment: {config['experiment_name']}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Checkpoint epoch: {epoch}\n")
        f.write(f"Best validation loss: {best_val_loss}\n\n")

        f.write("Model\n")
        f.write("-----\n")
        f.write(f"Model name: {model_config.get('name')}\n")
        f.write(f"Latent dim: {model_config.get('latent_dim')}\n")
        f.write(f"Feature maps: {model_config.get('feature_maps')}\n")
        f.write(f"Image channels: {model_config.get('img_channels')}\n\n")

        f.write("Preprocessing\n")
        f.write("-------------\n")
        f.write(f"Image size: {preproc_config.get('img_size')}\n")
        f.write(f"Type: {preproc_config.get('type')}\n\n")

        f.write("Generated files\n")
        f.write("---------------\n")
        f.write("test_reconstructions.png\n")
        f.write("generated_samples.png\n")
        f.write("latent_interpolation.png\n")
        f.write("latent_pca_test.png\n")
        f.write("latents_test.npy\n")


# -------------------------------------------------------------------
# Main evaluation
# -------------------------------------------------------------------

def evaluate(config, checkpoint_path, output_root):
    device = get_device()

    experiment_name = config["experiment_name"]
    output_dir = os.path.join(output_root, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Evaluating experiment: {experiment_name}")
    logging.info(f"Saving outputs to: {output_dir}")

    loaders = build_dataloaders(
        config["data"],
        config["training"],
        config["preprocessing"],
    )

    test_loader = loaders["test"]

    model = build_vae_model(config["model"]).to(device)
    epoch, best_val_loss = load_checkpoint(model, checkpoint_path, device)

    logging.info("Checkpoint loaded successfully.")
    logging.info(f"Test batches: {len(test_loader)}")

    save_test_reconstructions(
        model=model,
        loader=test_loader,
        device=device,
        save_path=os.path.join(output_dir, "test_reconstructions.png"),
        max_images=config["training"].get("num_reconstruction_images", 8),
    )

    save_generated_samples(
        model=model,
        device=device,
        save_path=os.path.join(output_dir, "generated_samples.png"),
        num_samples=config["training"].get("num_generated_samples", 16),
    )

    save_latent_interpolation(
        model=model,
        loader=test_loader,
        device=device,
        save_path=os.path.join(output_dir, "latent_interpolation.png"),
        num_steps=10,
    )

    latents, labels, sources = extract_latents(
        model=model,
        loader=test_loader,
        device=device,
        max_batches=30,
    )

    np.save(os.path.join(output_dir, "latents_test.npy"), latents)

    save_latent_pca(
        latents=latents,
        save_path=os.path.join(output_dir, "latent_pca_test.png"),
    )

    save_summary(
        config=config,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        epoch=epoch,
        best_val_loss=best_val_loss,
    )

    logging.info("Evaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="vae_evaluation")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    evaluate(
        config=config,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
    )