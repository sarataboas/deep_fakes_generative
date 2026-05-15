#!/usr/bin/env python3
"""
Computa FID para todos os modelos GAN treinados.

Uso (a partir da raiz do projeto):
    python compute_fid_gan.py

Resultados guardados em results/gan_fid_scores.csv
Imagens temporárias em fid_tmp/ (pode apagar depois)
"""

import os
import sys
import subprocess

# ---------------------------------------------------------------------------
# Instala dependências se não estiverem disponíveis
# ---------------------------------------------------------------------------
def _install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

try:
    from cleanfid import fid as cleanfid
except ImportError:
    print("A instalar clean-fid...")
    _install("clean-fid")
    from cleanfid import fid as cleanfid

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gan import Generator


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
METADATA_PATH  = "face_crop_final/full_face_crop_metadata.csv"
REAL_DIR       = "fid_tmp/real"
GEN_DIR        = "fid_tmp/generated"
RESULTS_PATH   = "results/gan_fid_scores.csv"
N_GENERATED    = 2048   # imagens geradas por modelo
BATCH_SIZE     = 128

EXPERIMENTS = [
    "gan_baseline",
    "gan_v2_balanced",
    "gan_v3_nsteps",
    "gan_v4_weaker_d",
    "gan_v5_noise",
    "gan_v6_labelflip",
    "gan_v7_dropout",
    "gan_v8_spectralnorm",
    "gan_v9_lr",
    "gan_v10_dropout",
    "gan_v11_gradclip",
    "gan_wgan_gp",
    "gan_wgan_gp_v2_latent",
]


# ---------------------------------------------------------------------------
# Imagens reais
# ---------------------------------------------------------------------------
def prepare_real_images():
    """Guarda todas as imagens reais do test set (wiki) redimensionadas a 64×64 em REAL_DIR."""
    if os.path.isdir(REAL_DIR) and len(os.listdir(REAL_DIR)) > 0:
        print(f"Imagens reais já existem em {REAL_DIR} ({len(os.listdir(REAL_DIR))} imgs) — a saltar.")
        return

    print("A preparar imagens reais (test set)...")
    df = pd.read_csv(METADATA_PATH)
    df = df[(df["split"] == "test") & (df["source_type"] == "wiki")].reset_index(drop=True)

    transform = transforms.Resize((64, 64))
    os.makedirs(REAL_DIR, exist_ok=True)

    for i, row in df.iterrows():
        img = Image.open(row["filepath"]).convert("RGB")
        img = transform(img)
        img.save(os.path.join(REAL_DIR, f"real_{i:05d}.png"))

    print(f"  {len(df)} imagens reais guardadas em {REAL_DIR}")


# ---------------------------------------------------------------------------
# Carregar Generator
# ---------------------------------------------------------------------------
def load_generator(experiment_name):
    """Carrega o Generator a partir do checkpoint, usando o config guardado."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{experiment_name}.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    cfg = ckpt["config"]["model"]
    feature_maps = cfg.get("feature_maps_g") or cfg.get("feature_maps", 64)
    generator = Generator(
        latent_dim   = cfg["latent_dim"],
        feature_maps = feature_maps,
        img_channels = cfg.get("img_channels", 3),
        spectral_norm= cfg.get("spectral_norm", False),
        dropout      = cfg.get("dropout_g", 0.0),
    ).to(DEVICE)

    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()

    return generator, cfg["latent_dim"]


# ---------------------------------------------------------------------------
# Gerar imagens
# ---------------------------------------------------------------------------
def generate_images(generator, latent_dim, out_dir):
    """Gera N_GENERATED imagens individuais (PNG) em out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    n_done = 0
    with torch.no_grad():
        while n_done < N_GENERATED:
            batch = min(BATCH_SIZE, N_GENERATED - n_done)
            z    = torch.randn(batch, latent_dim, device=DEVICE)
            imgs = generator(z)                        # [-1, 1]
            imgs = ((imgs + 1) / 2).clamp(0, 1)        # [0, 1]
            imgs = (imgs * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

            for arr in imgs:
                Image.fromarray(arr, "RGB").save(
                    os.path.join(out_dir, f"gen_{n_done:05d}.png")
                )
                n_done += 1

    return n_done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}\n")

    prepare_real_images()

    results = []

    for exp in EXPERIMENTS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{exp}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] checkpoint não encontrado: {ckpt_path}")
            continue

        print(f"\n{'='*50}")
        print(f"Experimento: {exp}")

        gen_dir = os.path.join(GEN_DIR, exp)

        # Gerar imagens
        print(f"  A gerar {N_GENERATED} imagens...")
        generator, latent_dim = load_generator(exp)
        generate_images(generator, latent_dim, gen_dir)
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Calcular FID
        print(f"  A calcular FID...")
        score = cleanfid.compute_fid(REAL_DIR, gen_dir, mode="clean", num_workers=2)
        print(f"  FID = {score:.2f}")

        results.append({"experiment": exp, "fid": round(score, 2)})

    # Guardar resultados
    os.makedirs("results", exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_PATH, index=False)

    print(f"\n{'='*50}")
    print(f"Resultados guardados em {RESULTS_PATH}\n")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
