#!/usr/bin/env python3
"""
Computes FID for all trained VAE models.

Usage (from the project root):
    python evaluation/compute_metrics_vae.py                           # all experiments
    python evaluation/compute_metrics_vae.py --experiments vae_perceptual_loss_v9_full_data

Results saved to results/vae_metrics.csv
Temporary images written to metrics_tmp/ (shared with compute_metrics_gan.py)
"""

import os
import sys
import argparse
import importlib
import subprocess


def _install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

try:
    import torch_fidelity
except ImportError:
    print("Installing torch-fidelity...")
    _install("torch-fidelity")
    import torch_fidelity

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
METADATA_PATH  = "face_crop_final/full_face_crop_metadata.csv"
REAL_BASE_DIR  = "metrics_tmp/real"
GEN_DIR        = "metrics_tmp/generated"
RESULTS_PATH   = "results/vae_metrics.csv"
N_GENERATED    = 2048
BATCH_SIZE     = 64

# Maps config.model.name or config.model.architecture → (module path, class name).
# Older checkpoints use the 'architecture' key; newer ones use 'name'.
MODEL_REGISTRY = {
    # newer checkpoints (name key)
    "vae64"     : ("models.variational_autoencoder",        "BaselineVAE"),
    "vae64_v2"  : ("models.variational_autoencoder_v2",     "BaselineVAEv2"),
    "vae128"    : ("models.variational_autoencoder_128",    "VAE128"),
    "vae128_v2" : ("models.variational_autoencoder_128_v2", "VAE128v2"),
    # older checkpoints (architecture key)
    "4_layers"  : ("models.variational_autoencoder",        "BaselineVAE"),
    "5_layers"  : ("models.variational_autoencoder_128",    "VAE128"),
}

EXPERIMENTS = [
    # older experiments (train_vae.py / train_vae_perceptual_loss_simple.py)
    "vae_baseline",
    "vae_kl_annealing",
    "vae_latent256_beta2",
    "5_layer_vae_big_imgsize",
    "06_baseline_vae_perceptual_loss",
    # perceptual loss experiments (train_vae_perceptual_loss.py)
    "vae_perceptual_loss",
    "vae_perceptual_loss_v2_warmup_early",
    "vae_perceptual_loss_v3_pixel_weight",
    "vae_perceptual_loss_v4_pixel_weight_7",
    "vae_perceptual_loss_v5_upsampling",
    "vae_perceptual_loss_v6_relu1_1",
    "vae_perceptual_loss_v7_end_beta",
    "vae_perceptual_loss_v9_full_data",
    "vae_perceptual_loss_v10_data_augmentation",
    "vae_perceptual_loss_v11_latent_dim",
    "vae_perceptual_loss_v12_64_full_data",
]


# ---------------------------------------------------------------------------
# Real images (reuses directories created by compute_metrics_gan.py if present)
# ---------------------------------------------------------------------------
def prepare_real_images(img_size: int) -> str:
    """
    Saves wiki test-set images resized to img_size into a resolution-specific directory.
    Returns the path to the directory.
    """
    real_dir = os.path.join(REAL_BASE_DIR, f"real_{img_size}")

    if os.path.isdir(real_dir) and len(os.listdir(real_dir)) >= 500:
        print(f"  Real images at {img_size}px already exist ({len(os.listdir(real_dir))}) — skipping.")
        return real_dir

    print(f"  Preparing real images {img_size}×{img_size} (wiki test set)...")
    df = pd.read_csv(METADATA_PATH)
    df = df[(df["split"] == "test") & (df["source_type"] == "wiki")].reset_index(drop=True)

    transform = transforms.Resize((img_size, img_size))
    os.makedirs(real_dir, exist_ok=True)

    for i, row in df.iterrows():
        img = Image.open(row["filepath"]).convert("RGB")
        img = transform(img)
        img.save(os.path.join(real_dir, f"real_{i:05d}.png"))

    print(f"  {len(df)} real images saved to {real_dir}")
    return real_dir


# ---------------------------------------------------------------------------
# Load VAE model
# ---------------------------------------------------------------------------
def load_vae(experiment_name):
    """
    Loads the correct VAE class based on config.model.name from the checkpoint.
    Uses MODEL_REGISTRY to map the name string to the Python class dynamically.
    Returns (model, latent_dim, img_size).
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{experiment_name}.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    cfg_model   = ckpt["config"]["model"]
    cfg_preproc = ckpt["config"].get("preprocessing", {})

    # Older checkpoints store the architecture under 'architecture'; newer ones use 'name'
    model_name   = cfg_model.get("name") or cfg_model.get("architecture", "vae64")
    latent_dim   = cfg_model.get("latent_dim", 256)
    feature_maps = cfg_model.get("feature_maps", 64)
    img_channels = cfg_model.get("img_channels", 3)
    img_size     = cfg_preproc.get("img_size", 64)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: '{model_name}'. Add it to MODEL_REGISTRY.")

    module_path, class_name = MODEL_REGISTRY[model_name]
    ModelClass = getattr(importlib.import_module(module_path), class_name)

    model = ModelClass(
        img_channels=img_channels,
        feature_maps=feature_maps,
        latent_dim=latent_dim,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, latent_dim, img_size


# ---------------------------------------------------------------------------
# Generate images
# ---------------------------------------------------------------------------
def generate_images(model, latent_dim, out_dir):
    """
    Generates N_GENERATED images by sampling z ~ N(0, I) and decoding via model.generate().
    Clears out_dir before writing so stale images from previous runs don't pollute results.
    """
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

    n_done = 0
    with torch.no_grad():
        while n_done < N_GENERATED:
            batch = min(BATCH_SIZE, N_GENERATED - n_done)
            imgs = model.generate(batch, DEVICE)        # [-1, 1]
            imgs = ((imgs + 1) / 2).clamp(0, 1)         # [0, 1]
            imgs = (imgs * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            for arr in imgs:
                Image.fromarray(arr, "RGB").save(
                    os.path.join(out_dir, f"gen_{n_done:05d}.png")
                )
                n_done += 1

    return n_done


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_fid(real_dir, gen_dir) -> float:
    """Computes FID between real and generated image directories using torch-fidelity."""
    metrics = torch_fidelity.calculate_metrics(
        input1  = real_dir,
        input2  = gen_dir,
        cuda    = torch.cuda.is_available(),
        fid     = True,
        isc     = False,
        verbose = False,
    )
    return round(metrics["frechet_inception_distance"], 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Experiments to evaluate (default: all).")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Reuse previously generated images in metrics_tmp/.")
    args = parser.parse_args()

    experiments = args.experiments or EXPERIMENTS

    print(f"Device: {DEVICE}\n")

    if os.path.exists(RESULTS_PATH):
        df_existing = pd.read_csv(RESULTS_PATH)
        done    = set(df_existing["experiment"].tolist())
        results = df_existing.to_dict("records")
        print(f"Already computed: {done}\n")
    else:
        done    = set()
        results = []

    for exp in experiments:
        if exp in done:
            print(f"[SKIP] {exp} — already computed")
            continue

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{exp}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {exp} — checkpoint not found")
            continue

        print(f"\n{'='*55}")
        print(f"Experiment: {exp}")

        gen_dir = os.path.join(GEN_DIR, exp)

        if not args.skip_generation:
            print(f"  Generating {N_GENERATED} images...")
            model, latent_dim, img_size = load_vae(exp)
            generate_images(model, latent_dim, gen_dir)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Read img_size from checkpoint without loading model weights
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            img_size = ckpt["config"].get("preprocessing", {}).get("img_size", 64)
            del ckpt

        real_dir = prepare_real_images(img_size)

        print("  Computing FID...")
        fid = compute_fid(real_dir, gen_dir)
        print(f"  FID={fid:.2f} | img_size={img_size}")

        results.append({"experiment": exp, "img_size": img_size, "fid": fid})

        # Save incrementally so partial results are not lost if the script crashes
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)

    print(f"\n{'='*55}")
    print(f"Results saved to {RESULTS_PATH}\n")
    df_final = pd.DataFrame(results).sort_values("fid")
    print(df_final.to_string(index=False))


if __name__ == "__main__":
    main()
