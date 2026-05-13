import os
import json
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.models import vgg16, VGG16_Weights

from src.setup import get_device, build_dataloaders
from src.utils import set_seed, save_history

from models.variational_autoencoder import BaselineVAE
from models.variational_autoencoder_128 import VAE128


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


# -------------------------------------------------------------------
# Model factory
# -------------------------------------------------------------------

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

    raise ValueError(f"Unknown VAE model name: {model_name}")


# -------------------------------------------------------------------
# Perceptual loss
# -------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG16 features.

    Assumes input images are normalized in [-1, 1].
    Internally converts them to ImageNet normalization for VGG.
    """

    def __init__(self, device, layer_cutoff=16):
        super().__init__()

        weights = VGG16_Weights.IMAGENET1K_V1
        vgg = vgg16(weights=weights).features[:layer_cutoff].to(device).eval()

        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def preprocess_for_vgg(self, x):
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)
        x = (x - self.mean) / self.std
        return x

    def forward(self, recon_imgs, imgs):
        recon_imgs = self.preprocess_for_vgg(recon_imgs)
        imgs = self.preprocess_for_vgg(imgs)

        recon_features = self.vgg(recon_imgs)
        img_features = self.vgg(imgs)

        return F.l1_loss(recon_features, img_features)


def vae_perceptual_loss(
    recon_imgs,
    imgs,
    mu,
    logvar,
    perceptual_loss_fn,
    beta=1.0,
    pixel_weight=0.1,
):
    perceptual_loss = perceptual_loss_fn(recon_imgs, imgs)
    pixel_loss = F.l1_loss(recon_imgs, imgs)

    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    total_loss = perceptual_loss + pixel_weight * pixel_loss + beta * kl_loss

    return total_loss, perceptual_loss, pixel_loss, kl_loss


# -------------------------------------------------------------------
# Image utilities
# -------------------------------------------------------------------

def denormalize_vae(imgs):
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

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch_idx,
    beta,
    perceptual_loss_fn,
    pixel_weight,
):
    model.train()

    total_loss = 0.0
    total_perceptual_loss = 0.0
    total_pixel_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx + 1} [Train VAE Perceptual]", leave=False)

    for batch in pbar:
        imgs = batch["image"].to(device)

        optimizer.zero_grad()

        recon_imgs, mu, logvar = model(imgs)

        loss, perceptual_loss, pixel_loss, kl_loss = vae_perceptual_loss(
            recon_imgs=recon_imgs,
            imgs=imgs,
            mu=mu,
            logvar=logvar,
            perceptual_loss_fn=perceptual_loss_fn,
            beta=beta,
            pixel_weight=pixel_weight,
        )

        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)

        total_loss += loss.item() * batch_size
        total_perceptual_loss += perceptual_loss.item() * batch_size
        total_pixel_loss += pixel_loss.item() * batch_size
        total_kl_loss += kl_loss.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "perc": f"{perceptual_loss.item():.4f}",
            "pixel": f"{pixel_loss.item():.4f}",
            "kl": f"{kl_loss.item():.4f}",
        })

    return {
        "loss": total_loss / total_samples,
        "perceptual_loss": total_perceptual_loss / total_samples,
        "pixel_loss": total_pixel_loss / total_samples,
        "kl_loss": total_kl_loss / total_samples,
    }


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
    beta,
    perceptual_loss_fn,
    pixel_weight,
):
    model.eval()

    total_loss = 0.0
    total_perceptual_loss = 0.0
    total_pixel_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0

    for batch in loader:
        imgs = batch["image"].to(device)

        recon_imgs, mu, logvar = model(imgs)

        loss, perceptual_loss, pixel_loss, kl_loss = vae_perceptual_loss(
            recon_imgs=recon_imgs,
            imgs=imgs,
            mu=mu,
            logvar=logvar,
            perceptual_loss_fn=perceptual_loss_fn,
            beta=beta,
            pixel_weight=pixel_weight,
        )

        batch_size = imgs.size(0)

        total_loss += loss.item() * batch_size
        total_perceptual_loss += perceptual_loss.item() * batch_size
        total_pixel_loss += pixel_loss.item() * batch_size
        total_kl_loss += kl_loss.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "perceptual_loss": total_perceptual_loss / total_samples,
        "pixel_loss": total_pixel_loss / total_samples,
        "kl_loss": total_kl_loss / total_samples,
    }


# -------------------------------------------------------------------
# Experiment utilities
# -------------------------------------------------------------------

def validate_train_sources(train_sources):
    allowed_sources = {"inpainting", "insight", "text2img"}

    if not isinstance(train_sources, list):
        raise ValueError("data.train_source must be a list.")

    if len(train_sources) == 0:
        raise ValueError("data.train_source cannot be empty.")

    invalid_sources = set(train_sources) - allowed_sources

    if invalid_sources:
        raise ValueError(
            f"Invalid train_source values: {sorted(invalid_sources)}. "
            f"Allowed sources: {sorted(allowed_sources)}"
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


def compute_beta(epoch, training_config):
    kl_config = training_config.get("kl_annealing", {})

    if not kl_config.get("enabled", False):
        return training_config.get("beta", 1.0)

    warmup_epochs = kl_config.get("warmup_epochs", 30)
    start_beta = kl_config.get("start_beta", 0.0)
    end_beta = kl_config.get("end_beta", 1.0)

    progress = min((epoch + 1) / warmup_epochs, 1.0)
    beta = start_beta + (end_beta - start_beta) * progress

    return beta


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

    logging.info(f"=== VAE Perceptual Loss Experiment: {config['experiment_name']} ===")
    logging.info(f"Training sources: {c_data['train_source']}")
    logging.info(f"Model: {c_model.get('name', 'vae64')}")
    logging.info(f"Image size: {c_preproc.get('img_size')}")

    loaders = build_dataloaders(c_data, c_train, c_preproc)

    logging.info(f"Train batches: {len(loaders['train'])}")
    logging.info(f"Validation batches: {len(loaders['val'])}")
    logging.info(f"Test batches: {len(loaders['test'])}")

    model = build_vae_model(c_model).to(device)

    perceptual_loss_fn = VGGPerceptualLoss(
        device=device,
        layer_cutoff=c_train.get("perceptual_layer_cutoff", 16),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=c_train.get("lr", 1e-4),
        weight_decay=c_train.get("weight_decay", 0.0),
    )

    scheduler = build_scheduler(optimizer, c_train)

    pixel_weight = c_train.get("pixel_weight", 0.1)

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

    for epoch in range(c_train["num_epochs"]):
        beta = compute_beta(epoch, c_train)

        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            beta=beta,
            perceptual_loss_fn=perceptual_loss_fn,
            pixel_weight=pixel_weight,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=loaders["val"],
            device=device,
            beta=beta,
            perceptual_loss_fn=perceptual_loss_fn,
            pixel_weight=pixel_weight,
        )

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_perceptual_loss": train_metrics["perceptual_loss"],
            "train_pixel_loss": train_metrics["pixel_loss"],
            "train_kl_loss": train_metrics["kl_loss"],
            "val_loss": val_metrics["loss"],
            "val_perceptual_loss": val_metrics["perceptual_loss"],
            "val_pixel_loss": val_metrics["pixel_loss"],
            "val_kl_loss": val_metrics["kl_loss"],
            "lr": current_lr,
            "beta": beta,
            "pixel_weight": pixel_weight,
        })

        logging.info(
            f"Epoch {epoch + 1}: "
            f"Beta={beta:.4f} | "
            f"Train Loss={train_metrics['loss']:.4f} | "
            f"Train Perc={train_metrics['perceptual_loss']:.4f} | "
            f"Train Pixel={train_metrics['pixel_loss']:.4f} | "
            f"Train KL={train_metrics['kl_loss']:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f} | "
            f"Val Perc={val_metrics['perceptual_loss']:.4f} | "
            f"Val Pixel={val_metrics['pixel_loss']:.4f} | "
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
                save_path=os.path.join(save_dir, f"generated_epoch_{epoch + 1}.png"),
                num_samples=c_train.get("num_generated_samples", 16),
            )

            reconstruction_loader = (
                loaders["val"] if len(loaders["val"]) > 0 else loaders["train"]
            )

            save_reconstructions(
                model=model,
                loader=reconstruction_loader,
                device=device,
                save_path=os.path.join(save_dir, f"recon_epoch_{epoch + 1}.png"),
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