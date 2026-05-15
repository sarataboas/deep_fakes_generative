#!/usr/bin/env python3
"""
Computa FID e Inception Score para todos os modelos GAN.

Uso (a partir da raiz do projecto):
    python compute_metrics_gan.py                        # todos os experimentos
    python compute_metrics_gan.py --experiments gan_wgan_gp_v4_128  # um só

Resultados guardados em results/gan_metrics.csv
Imagens temporárias em metrics_tmp/ (pode apagar depois)
"""

import os
import sys
import argparse
import subprocess

# ---------------------------------------------------------------------------
# Dependências
# ---------------------------------------------------------------------------
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gan import Generator


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
METADATA_PATH  = "face_crop_final/full_face_crop_metadata.csv"
REAL_BASE_DIR  = "metrics_tmp/real"   # directório por resolução: real_64/, real_128/, ...
GEN_DIR        = "metrics_tmp/generated"
RESULTS_PATH   = "results/gan_metrics.csv"
N_GENERATED    = 2048
BATCH_SIZE     = 64

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
    "gan_wgan_gp_v3_full_data",
    "gan_wgan_gp_v4_128",
]


# ---------------------------------------------------------------------------
# Imagens reais
# ---------------------------------------------------------------------------
def prepare_real_images(img_size: int) -> str:
    """
    Guarda imagens reais do test set (wiki) redimensionadas a img_size.
    Cada resolução tem o seu próprio directório (real_64/, real_128/, ...)
    para garantir que o FID compara sempre imagens à mesma resolução.
    Devolve o caminho do directório.
    """
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
# Carregar Generator
# ---------------------------------------------------------------------------
def load_generator(experiment_name):
    """Carrega Generator do checkpoint, lendo img_size do config guardado."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{experiment_name}.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    cfg_model  = ckpt["config"]["model"]
    cfg_preproc = ckpt["config"].get("preprocessing", {})

    feature_maps = cfg_model.get("feature_maps_g") or cfg_model.get("feature_maps", 64)
    img_size     = cfg_preproc.get("img_size", 64)

    generator = Generator(
        latent_dim   = cfg_model["latent_dim"],
        feature_maps = feature_maps,
        img_channels = cfg_model.get("img_channels", 3),
        img_size     = img_size,
        spectral_norm= cfg_model.get("spectral_norm", False),
        dropout      = cfg_model.get("dropout_g", 0.0),
    ).to(DEVICE)

    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()

    return generator, cfg_model["latent_dim"], img_size


# ---------------------------------------------------------------------------
# Gerar imagens
# ---------------------------------------------------------------------------
def generate_images(generator, latent_dim, out_dir):
    """Gera N_GENERATED imagens em out_dir, apagando as anteriores."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

    n_done = 0
    with torch.no_grad():
        while n_done < N_GENERATED:
            batch = min(BATCH_SIZE, N_GENERATED - n_done)
            z    = torch.randn(batch, latent_dim, device=DEVICE)
            imgs = generator(z)                          # [-1, 1]
            imgs = ((imgs + 1) / 2).clamp(0, 1)          # [0, 1]
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
def compute_metrics(real_dir, gen_dir):
    """
    Calcula FID e Inception Score usando torch-fidelity.

    FID  — mede qualidade + diversidade em conjunto (mais baixo = melhor).
    IS   — mede qualidade e diversidade separadamente em modo self-contained
           (mais alto = melhor; IS médio de imagens reais de faces ~2–3).
    """
    metrics = torch_fidelity.calculate_metrics(
        input1  = real_dir,
        input2  = gen_dir,
        cuda    = torch.cuda.is_available(),
        fid     = True,
        isc     = True,   # Inception Score
        verbose = False,
    )
    return {
        "fid"    : round(metrics["frechet_inception_distance"], 2),
        "is_mean": round(metrics["inception_score_mean"], 3),
        "is_std" : round(metrics["inception_score_std"],  3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments", nargs="*", default=None,
        help="Experimentos a computar (default: todos).",
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Reutiliza imagens já geradas em metrics_tmp/ (útil para re-computar métricas).",
    )
    args = parser.parse_args()

    experiments = args.experiments or EXPERIMENTS

    print(f"Device: {DEVICE}\n")

    # Carrega resultados já existentes para não repetir experimentos
    if os.path.exists(RESULTS_PATH):
        df_existing = pd.read_csv(RESULTS_PATH)
        done = set(df_existing["experiment"].tolist())
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
            generator, latent_dim, img_size = load_generator(exp)
            generate_images(generator, latent_dim, gen_dir)
            del generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Lê img_size directamente do checkpoint — sem instanciar o modelo
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            img_size = ckpt["config"].get("preprocessing", {}).get("img_size", 64)
            del ckpt

        # Imagens reais à mesma resolução que as geradas — comparação justa
        real_dir = prepare_real_images(img_size)

        print("  A calcular FID + IS...")
        m = compute_metrics(real_dir, gen_dir)
        print(f"  FID={m['fid']:.2f} | IS={m['is_mean']:.3f} ± {m['is_std']:.3f} | img_size={img_size}")

        results.append({"experiment": exp, "img_size": img_size, **m})

        # Guarda resultados incrementalmente
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)

    print(f"\n{'='*55}")
    print(f"Resultados guardados em {RESULTS_PATH}\n")
    df_final = pd.DataFrame(results).sort_values("fid")
    print(df_final.to_string(index=False))


if __name__ == "__main__":
    main()
