import os
import json
import argparse
import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from src.setup import get_device, build_dataloaders
from src.generator import BaselineVAE
from src.utils import set_seed, save_history


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


# -------------------------------------------------------------------
# Loss
# -------------------------------------------------------------------

def vae_loss(recon_imgs, imgs, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_imgs, imgs, reduction="mean")

    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    return recon_loss + beta * kl_loss

def denormalize_imagenet(imgs):
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
    return torch.clamp(imgs * std + mean, 0, 1)


def save_generated_samples(model, device, save_path, num_samples=16):
    model.eval()

    with torch.no_grad():
        samples = model.generate(num_samples=num_samples, device=device)

    samples = denormalize_imagenet(samples)
    grid = make_grid(samples.cpu(), nrow=4)

    save_image(grid, save_path)


def save_reconstructions(model, loader, device, save_path, max_images=8):
    model.eval()

    batch = next(iter(loader))
    imgs = batch["image"][:max_images].to(device)

    with torch.no_grad():
        recon_imgs, _, _ = model(imgs)

    imgs = denormalize_imagenet(imgs)
    recon_imgs = denormalize_imagenet(recon_imgs)

    comparison = torch.cat([imgs, recon_imgs], dim=0)
    grid = make_grid(comparison.cpu(), nrow=max_images)

    save_image(grid, save_path)

# -------------------------------------------------------------------
# Train / Val
# -------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch_idx, beta):
    model.train()
    total_loss, total = 0.0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1} [Train VAE]", leave=False)

    for batch in pbar:
        imgs = batch["image"].to(device)

        optimizer.zero_grad()

        recon_imgs, mu, logvar = model(imgs)
        loss = vae_loss(recon_imgs, imgs, mu, logvar, beta)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total


def validate_one_epoch(model, loader, device, beta):
    model.eval()
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)

            recon_imgs, mu, logvar = model(imgs)
            loss = vae_loss(recon_imgs, imgs, mu, logvar, beta)

            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

    return total_loss / total


# -------------------------------------------------------------------
# Experiment
# -------------------------------------------------------------------

def run_experiment(config):
    set_seed(config.get("seed", 42))
    device = get_device()

    c_data = config["data"]
    c_model = config["model"]
    c_train = config["training"]
    c_preproc = config["preprocessing"]

    logging.info(f"=== VAE Experiment: {config['experiment_name']} ===")

    # SAME loader as classifier (important)
    loaders = build_dataloaders(c_data, c_train, c_preproc)
    logging.info(f"Train batches: {len(loaders['train'])}")
    logging.info(f"Val batches: {len(loaders['val'])}")
    logging.info(f"Test batches: {len(loaders['test'])}")

    model = BaselineVAE(
        latent_dim=c_model.get("latent_dim", 128),
        img_channels=c_model.get("img_channels", 3),
        feature_maps=c_model.get("feature_maps", 32)
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=c_train.get("lr", 2e-4)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=c_train.get("num_epochs", 50)
    )

    beta = c_train.get("beta", 1.0)

    best_loss = float("inf")
    history = []

    save_dir = c_train.get("save_dir", f"outputs/{config['experiment_name']}")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(c_train["num_epochs"]):

        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, device, epoch, beta
        )

        val_loss = validate_one_epoch(
            model, loaders["val"], device, beta
        )

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "loss": train_loss,
            "val_loss": val_loss
        })

        logging.info(f"Epoch {epoch+1}: Loss={train_loss:.4f} | Val={val_loss:.4f}")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{config['experiment_name']}.pt")
            logging.info("  >> Checkpoint saved")
        
        if (epoch + 1) % c_train.get("sample_every", 5) == 0:

            save_generated_samples(
                model=model,
                device=device,
                save_path=os.path.join(save_dir, f"generated_epoch_{epoch+1}.png"),
                num_samples=c_train.get("num_generated_samples", 16)
            )

            recon_loader = loaders["val"] if len(loaders["val"]) > 0 else loaders["train"]

            save_reconstructions(
                model=model,
                loader=recon_loader,
                device=device,
                save_path=os.path.join(save_dir, f"recon_epoch_{epoch+1}.png")
            )

    save_history(history, config["experiment_name"])

    return model


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    run_experiment(config)