import os
import json
import argparse
import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from src.setup import get_device, build_dataloaders
from models.variational_autoencoder import BaselineVAE
from models.variational_autoencoder_128 import VAE128
from src.utils import set_seed, save_history


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


# -------------------------------------------------------------------
# Loss
# -------------------------------------------------------------------

# def vae_loss(recon_imgs, imgs, mu, logvar, beta=1.0):
#     reconstruction_loss = F.mse_loss(recon_imgs, imgs, reduction="mean")

#     kl_loss = -0.5 * torch.mean(
#         torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
#     )

#     total_loss = reconstruction_loss + beta * kl_loss

#     return total_loss, reconstruction_loss, kl_loss

def build_vae_model(model_config):
    model_name = model_config.get("architecture", "4_layers")

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

    raise ValueError(f"Unknown VAE model name: {model_name}")

def vae_loss(recon_imgs, imgs, mu, logvar, beta=1.0):
    reconstruction_loss = F.mse_loss(recon_imgs, imgs, reduction="sum")
    reconstruction_loss = reconstruction_loss / imgs.size(0)

    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    total_loss = reconstruction_loss + beta * kl_loss

    return total_loss, reconstruction_loss, kl_loss


# -------------------------------------------------------------------
# Image utilities
# -------------------------------------------------------------------

def denormalize_vae(imgs):
    """
    Convert images from [-1, 1] back to [0, 1].
    """
    return torch.clamp((imgs + 1.0) / 2.0, 0.0, 1.0)


def save_generated_samples(model, device, save_path, num_samples=16):
    model.eval()

    with torch.no_grad():
        samples = model.generate(num_samples=num_samples, device=device)

    samples = denormalize_vae(samples)
    grid = make_grid(samples.cpu(), nrow=4)

    save_image(grid, save_path)


def save_reconstructions(model, loader, device, save_path, max_images=8):
    model.eval()

    batch = next(iter(loader))
    imgs = batch["image"][:max_images].to(device)

    with torch.no_grad():
        recon_imgs, _, _ = model(imgs)

    imgs = denormalize_vae(imgs)
    recon_imgs = denormalize_vae(recon_imgs)

    comparison = torch.cat([imgs, recon_imgs], dim=0)
    grid = make_grid(comparison.cpu(), nrow=max_images)

    save_image(grid, save_path)


# -------------------------------------------------------------------
# Train / Validation
# -------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch_idx, beta):
    model.train()

    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx + 1} [Train VAE]", leave=False)

    for batch in pbar:
        imgs = batch["image"].to(device)

        optimizer.zero_grad()

        recon_imgs, mu, logvar = model(imgs)

        loss, reconstruction_loss, kl_loss = vae_loss(
            recon_imgs, imgs, mu, logvar, beta
        )

        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)

        total_loss += loss.item() * batch_size
        total_reconstruction_loss += reconstruction_loss.item() * batch_size
        total_kl_loss += kl_loss.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "recon": f"{reconstruction_loss.item():.4f}",
            "kl": f"{kl_loss.item():.4f}",
        })

    return {
        "loss": total_loss / total_samples,
        "reconstruction_loss": total_reconstruction_loss / total_samples,
        "kl_loss": total_kl_loss / total_samples,
    }


def validate_one_epoch(model, loader, device, beta):
    model.eval()

    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)

            recon_imgs, mu, logvar = model(imgs)

            loss, reconstruction_loss, kl_loss = vae_loss(
                recon_imgs, imgs, mu, logvar, beta
            )

            batch_size = imgs.size(0)

            total_loss += loss.item() * batch_size
            total_reconstruction_loss += reconstruction_loss.item() * batch_size
            total_kl_loss += kl_loss.item() * batch_size
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "reconstruction_loss": total_reconstruction_loss / total_samples,
        "kl_loss": total_kl_loss / total_samples,
    }


# -------------------------------------------------------------------
# Experiment utilities
# -------------------------------------------------------------------

def validate_train_sources(train_sources):
    allowed_sources = {"inpainting", "insight", "text2img"}

    if not isinstance(train_sources, list):
        raise ValueError(
            "data.train_source must be a list. "
            "Example: ['text2img'] or ['inpainting', 'insight', 'text2img']"
        )

    if len(train_sources) == 0:
        raise ValueError("data.train_source cannot be an empty list.")

    invalid_sources = set(train_sources) - allowed_sources

    if invalid_sources:
        raise ValueError(
            f"Invalid train_source values: {sorted(invalid_sources)}. "
            f"Allowed synthetic sources are: {sorted(allowed_sources)}"
        )


def build_scheduler(optimizer, training_config):
    scheduler_config = training_config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "cosine")

    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get(
                "T_max",
                training_config.get("num_epochs", 50),
            ),
        )

    if scheduler_type == "none":
        return None

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


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

    validate_train_sources(c_data.get("train_source"))

    logging.info(f"=== VAE Experiment: {config['experiment_name']} ===")
    logging.info(f"Training sources: {c_data['train_source']}")

    loaders = build_dataloaders(c_data, c_train, c_preproc)

    logging.info(f"Train batches: {len(loaders['train'])}")
    logging.info(f"Validation batches: {len(loaders['val'])}")
    logging.info(f"Test batches: {len(loaders['test'])}")

    model = build_vae_model(c_model).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=c_train.get("lr", 1e-4),
        weight_decay=c_train.get("weight_decay", 0.0),
    )

    scheduler = build_scheduler(optimizer, c_train)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    save_dir = c_train.get("save_dir", f"outputs/{config['experiment_name']}")
    checkpoint_dir = c_train.get("checkpoint_dir", "checkpoints")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    early_stopping_config = c_train.get("early_stopping", {})
    early_stopping_enabled = early_stopping_config.get("enabled", False)
    patience = early_stopping_config.get("patience", 10)

    kl_config = c_train.get("kl_annealing", {})
    kl_annealing_enabled = kl_config.get("enabled", False)

    for epoch in range(c_train["num_epochs"]):

        if kl_annealing_enabled:
            warmup_epochs = kl_config.get("warmup_epochs", 30)
            start_beta = kl_config.get("start_beta", 0.0)
            end_beta = kl_config.get("end_beta", 1.0)

            progress = min((epoch + 1) / warmup_epochs, 1.0)
            beta = start_beta + (end_beta - start_beta) * progress
        else:
            beta = c_train.get("beta", 1.0)

        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            beta=beta,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=loaders["val"],
            device=device,
            beta=beta,
        )

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_reconstruction_loss": train_metrics["reconstruction_loss"],
            "train_kl_loss": train_metrics["kl_loss"],
            "val_loss": val_metrics["loss"],
            "val_reconstruction_loss": val_metrics["reconstruction_loss"],
            "val_kl_loss": val_metrics["kl_loss"],
            "lr": current_lr,
            "beta": beta,
        })

        logging.info(
            f"Epoch {epoch + 1}: "
            f"Beta={beta:.4f} | "
            f"Train Loss={train_metrics['loss']:.4f} | "
            f"Train Recon={train_metrics['reconstruction_loss']:.4f} | "
            f"Train KL={train_metrics['kl_loss']:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f} | "
            f"Val Recon={val_metrics['reconstruction_loss']:.4f} | "
            f"Val KL={val_metrics['kl_loss']:.4f} | "
            f"LR={current_lr:.6f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{config['experiment_name']}.pt",
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "decoder_state_dict": model.decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "best_val_loss": best_val_loss,
                    "config": config,
                },
                checkpoint_path,
            )

            logging.info(f"Checkpoint saved to {checkpoint_path}")

        else:
            epochs_without_improvement += 1

        if (epoch + 1) % c_train.get("sample_every", 5) == 0:
            save_generated_samples(
                model=model,
                device=device,
                save_path=os.path.join(
                    save_dir,
                    f"generated_epoch_{epoch + 1}.png",
                ),
                num_samples=c_train.get("num_generated_samples", 16),
            )

            reconstruction_loader = (
                loaders["val"] if len(loaders["val"]) > 0 else loaders["train"]
            )

            save_reconstructions(
                model=model,
                loader=reconstruction_loader,
                device=device,
                save_path=os.path.join(
                    save_dir,
                    f"recon_epoch_{epoch + 1}.png",
                ),
                max_images=c_train.get("num_reconstruction_images", 8),
            )

        if early_stopping_enabled and epochs_without_improvement >= patience:
            logging.info(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

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