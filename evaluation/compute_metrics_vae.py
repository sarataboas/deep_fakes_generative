#!/usr/bin/env python3
"""
Computa FID para todos os modelos VAE treinados.

Uso (a partir da raiz do projecto):
    python compute_metrics_vae.py                          # todos
    python compute_metrics_vae.py --experiments vae_perceptual_loss_v9_full_data

Resultados guardados em results/vae_metrics.csv
Imagens temporárias em metrics_tmp/ (partilhado com compute_metrics_gan.py)
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
    print("A instalar torch-fidelity...")
    _install("torch-fidelity")
    import torch_fidelity

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
METADATA_PATH  = "face_crop_final/full_face_crop_metadata.csv"
REAL_BASE_DIR  = "metrics_tmp/real"
GEN_DIR        = "metrics_tmp/generated"
RESULTS_PATH   = "results/vae_metrics.csv"
N_GENERATED    = 2048
BATCH_SIZE     = 64

# Mapeamento model.name → (módulo, classe)
MODEL_REGISTRY = {
    "vae64"     : ("models.variational_autoencoder",       "BaselineVAE"),
    "vae64_v2"  : ("models.variational_autoencoder_v2",    "BaselineVAEv2"),
    "vae128"    : ("models.variational_autoencoder_128",   "VAE128"),
    "vae128_v2" : ("models.variational_autoencoder_128_v2","VAE128v2"),
}

EXPERIMENTS = [
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
# Imagens reais (reutiliza directórios do script GAN se já existirem)
# ---------------------------------------------------------------------------
def prepare_real_images(img_size: int) -> str:
    real_dir = os.path.join(REAL_BASE_DIR, f"real_{img_size}")

    if os.path.isdir(real_dir) and len(os.listdir(real_dir)) >= 500:
        print(f"  Imagens reais {img_size}px já existem ({len(os.listdir(real_dir))}) — a saltar.")
        return real_dir

    print(f"  A preparar imagens reais {img_size}×{img_size} (test set wiki)...")
    df = pd.read_csv(METADATA_PATH)
    df = df[(df["split"] == "test") & (df["source_type"] == "wiki")].reset_index(drop=True)

    transform = transforms.Resize((img_size, img_size))
    os.makedirs(real_dir, exist_ok=True)

    for i, row in df.iterrows():
        img = Image.open(row["filepath"]).convert("RGB")
        img = transform(img)
        img.save(os.path.join(real_dir, f"real_{i:05d}.png"))

    print(f"  {len(df)} imagens reais guardadas em {real_dir}")
    return real_dir


# ---------------------------------------------------------------------------
# Carregar modelo VAE
# ---------------------------------------------------------------------------
def load_vae(experiment_name):
    """
    Carrega o modelo VAE correcto com base em config.model.name.
    Devolve (model, latent_dim, img_size).
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{experiment_name}.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    cfg_model  = ckpt["config"]["model"]
    cfg_preproc = ckpt["config"].get("preprocessing", {})

    model_name = cfg_model.get("name", "vae64")
    latent_dim = cfg_model.get("latent_dim", 256)
    feature_maps = cfg_model.get("feature_maps", 64)
    img_channels = cfg_model.get("img_channels", 3)
    img_size   = cfg_preproc.get("img_size", 64)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo desconhecido: '{model_name}'. Adiciona ao MODEL_REGISTRY.")

    module_path, class_name = MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)

    model = ModelClass(
        img_channels=img_channels,
        feature_maps=feature_maps,
        latent_dim=latent_dim,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, latent_dim, img_size


# ---------------------------------------------------------------------------
# Gerar imagens
# ---------------------------------------------------------------------------
def generate_images(model, latent_dim, out_dir):
    """Gera N_GENERATED imagens usando model.generate() — z ~ N(0, I)."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

    n_done = 0
    with torch.no_grad():
        while n_done < N_GENERATED:
            batch = min(BATCH_SIZE, N_GENERATED - n_done)
            imgs = model.generate(batch, DEVICE)       # [-1, 1]
            imgs = ((imgs + 1) / 2).clamp(0, 1)        # [0, 1]
            imgs = (imgs * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            for arr in imgs:
                Image.fromarray(arr, "RGB").save(
                    os.path.join(out_dir, f"gen_{n_done:05d}.png")
                )
                n_done += 1

    return n_done


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def compute_fid(real_dir, gen_dir):
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
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    experiments = args.experiments or EXPERIMENTS

    print(f"Device: {DEVICE}\n")

    if os.path.exists(RESULTS_PATH):
        df_existing = pd.read_csv(RESULTS_PATH)
        done    = set(df_existing["experiment"].tolist())
        results = df_existing.to_dict("records")
        print(f"Resultados já existentes: {done}\n")
    else:
        done    = set()
        results = []

    for exp in experiments:
        if exp in done:
            print(f"[SKIP] {exp} — já computado")
            continue

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{exp}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {exp} — checkpoint não encontrado")
            continue

        print(f"\n{'='*55}")
        print(f"Experimento: {exp}")

        gen_dir = os.path.join(GEN_DIR, exp)

        if not args.skip_generation:
            print(f"  A gerar {N_GENERATED} imagens...")
            model, latent_dim, img_size = load_vae(exp)
            generate_images(model, latent_dim, gen_dir)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            img_size = ckpt["config"].get("preprocessing", {}).get("img_size", 64)
            del ckpt

        real_dir = prepare_real_images(img_size)

        print("  A calcular FID...")
        fid = compute_fid(real_dir, gen_dir)
        print(f"  FID={fid:.2f} | img_size={img_size}")

        results.append({"experiment": exp, "img_size": img_size, "fid": fid})

        os.makedirs("results", exist_ok=True)
        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)

    print(f"\n{'='*55}")
    print(f"Resultados guardados em {RESULTS_PATH}\n")
    df_final = pd.DataFrame(results).sort_values("fid")
    print(df_final.to_string(index=False))


if __name__ == "__main__":
    main()
