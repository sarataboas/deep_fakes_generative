import os
import json
import argparse
import logging

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from src.setup import get_device, build_dataloaders
from src.utils import set_seed, save_history
from models.gan import Generator, Discriminator


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


# -------------------------------------------------------------------
# Utilitários de imagem
# -------------------------------------------------------------------

def denormalize(imgs):
    """Converte imagens de [-1, 1] para [0, 1] para visualização."""
    return torch.clamp((imgs + 1.0) / 2.0, 0.0, 1.0)


def plot_training(experiment_name):
    """Lê o CSV de histórico e guarda o gráfico de losses em plots/."""
    csv_path = os.path.join("results", f"{experiment_name}.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    os.makedirs("plots", exist_ok=True)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"{experiment_name} — Treino", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df["epoch"], df["loss_D"], label="Loss D", color="#e74c3c", linewidth=1.5)
    ax1.plot(df["epoch"], df["loss_G"], label="Loss G", color="#3498db", linewidth=1.5)
    ax1.axhline(y=0.693, color="gray", linestyle="--", linewidth=1,
                alpha=0.6, label="Equilíbrio teórico (ln2)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss D vs Loss G")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df["epoch"], df["Dx"], color="#2ecc71", linewidth=1.5)
    ax2.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Equilíbrio (0.5)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("D(x)")
    ax2.set_title("D(x) — confiança em imagens reais")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df["epoch"], df["DGz"], color="#e67e22", linewidth=1.5)
    ax3.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Equilíbrio (0.5)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("D(G(z))")
    ax3.set_title("D(G(z)) — confiança em imagens falsas")
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(alpha=0.3)

    out_path = os.path.join("plots", f"{experiment_name}_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Gráfico guardado em {out_path}")


def add_instance_noise(imgs, noise_std):
    """
    Adiciona ruído gaussiano às imagens antes de entrarem no Discriminator.

    Impede o D de memorizar padrões de pixel exactos, forçando-o a
    aprender features mais gerais. O std decai ao longo do treino —
    quando o G melhora, o ruído diminui e o D pode ser mais exigente.

    Clamp garante que as imagens se mantêm no range [-1, 1].
    """
    if noise_std <= 0:
        return imgs
    return torch.clamp(imgs + torch.randn_like(imgs) * noise_std, -1.0, 1.0)


def save_generated_samples(generator, fixed_noise, device, save_path):
    """
    Gera imagens a partir do fixed_noise (sempre o mesmo z ao longo do treino)
    para poder comparar visualmente a evolução entre epochs.
    """
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_noise)
    samples = denormalize(samples.cpu())
    grid = make_grid(samples, nrow=4)
    save_image(grid, save_path)
    generator.train()


# -------------------------------------------------------------------
# Loop de treino
# -------------------------------------------------------------------

def train_one_epoch(
    generator,
    discriminator,
    loader,
    optimizer_G,
    optimizer_D,
    criterion,
    device,
    latent_dim,
    label_smoothing,
    n_steps_G,
    noise_std,
    flip_prob,
    grad_clip,
    epoch_idx,
):
    """
    Loop de treino GAN para uma epoch.

    Em cada batch fazemos passos separados para D e G:

    1. Passo do Discriminator (1 vez):
       - Passa imagens reais → quer output próximo de 1
       - Passa imagens falsas (geradas) → quer output próximo de 0
       - Actualiza só os pesos do D

    2. Passo do Generator (n_steps_G vezes):
       - Passa imagens falsas pelo D (sem actualizar D)
       - Quer que o D classifique as falsas como reais (labels = 1)
       - Actualiza só os pesos do G
       - Repetir n_steps_G vezes dá ao G mais oportunidades de
         aprender por cada passo do D — útil quando D domina.

    Métricas D(x) e D(G(z)) medem o equilíbrio do treino:
       - D(x)    → deve convergir para ~0.5 (D incerto sobre reais)
       - D(G(z)) → deve convergir para ~0.5 (D incerto sobre falsas)
    """
    generator.train()
    discriminator.train()

    total_loss_D = 0.0
    total_loss_G = 0.0
    total_Dx = 0.0
    total_DGz = 0.0
    total_samples = 0

    real_label_val = 1.0 - label_smoothing
    fake_label_val = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx + 1} [GAN]", leave=False)

    for batch in pbar:
        real_imgs = batch["image"].to(device)
        batch_size = real_imgs.size(0)

        real_labels = torch.full((batch_size, 1), real_label_val, device=device)
        fake_labels = torch.full((batch_size, 1), fake_label_val, device=device)

        # ---- Passo 1: Treinar o Discriminator ----
        optimizer_D.zero_grad()

        # One-sided label flipping: com probabilidade flip_prob, um label
        # real é trocado para 0. O D nunca sabe quando vai ser enganado,
        # o que o impede de ficar demasiado confiante nas imagens reais.
        if flip_prob > 0:
            flip_mask = torch.rand(batch_size, 1, device=device) < flip_prob
            real_labels_d = torch.where(flip_mask, fake_labels, real_labels)
        else:
            real_labels_d = real_labels

        # Ruído adicionado às imagens antes do D — o G nunca vê este ruído
        pred_real = discriminator(add_instance_noise(real_imgs, noise_std))
        loss_D_real = criterion(pred_real, real_labels_d)

        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(z).detach()
        pred_fake = discriminator(add_instance_noise(fake_imgs, noise_std))
        loss_D_fake = criterion(pred_fake, fake_labels)

        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
        optimizer_D.step()

        # ---- Passo 2: Treinar o Generator (n_steps_G vezes) ----
        for _ in range(n_steps_G):
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            pred_fake_for_G = discriminator(fake_imgs)
            loss_G = criterion(pred_fake_for_G, real_labels)

            loss_G.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            optimizer_G.step()

        # Métricas — usamos o último passo do G para D(G(z))
        Dx = torch.sigmoid(pred_real).mean().item()
        DGz = torch.sigmoid(pred_fake_for_G).mean().item()

        total_loss_D += loss_D.item() * batch_size
        total_loss_G += loss_G.item() * batch_size
        total_Dx += Dx * batch_size
        total_DGz += DGz * batch_size
        total_samples += batch_size

        pbar.set_postfix({
            "D": f"{loss_D.item():.3f}",
            "G": f"{loss_G.item():.3f}",
            "D(x)": f"{Dx:.2f}",
            "D(G(z))": f"{DGz:.2f}",
        })

    return {
        "loss_D": total_loss_D / total_samples,
        "loss_G": total_loss_G / total_samples,
        "Dx": total_Dx / total_samples,
        "DGz": total_DGz / total_samples,
    }


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

    logging.info(f"=== GAN Experiment: {config['experiment_name']} ===")
    logging.info(f"Device: {device}")

    loaders = build_dataloaders(c_data, c_train, c_preproc)
    logging.info(f"Train batches: {len(loaders['train'])}")

    latent_dim = c_model.get("latent_dim", 128)
    img_channels = c_model.get("img_channels", 3)
    # feature_maps_g e feature_maps_d permitem dar capacidades diferentes
    # a G e D. Se não especificados, usam o valor de feature_maps.
    feature_maps_g = c_model.get("feature_maps_g", c_model.get("feature_maps", 32))
    feature_maps_d = c_model.get("feature_maps_d", c_model.get("feature_maps", 32))

    use_sn = c_model.get("spectral_norm", False)

    generator = Generator(
        latent_dim=latent_dim,
        feature_maps=feature_maps_g,
        img_channels=img_channels,
        spectral_norm=use_sn,
        dropout=c_model.get("dropout_g", 0.0),
    ).to(device)

    discriminator = Discriminator(
        feature_maps=feature_maps_d,
        img_channels=img_channels,
        spectral_norm=use_sn,
        dropout=c_model.get("dropout_d", 0.0),
    ).to(device)

    logging.info(f"Generator params:     {sum(p.numel() for p in generator.parameters()):,}")
    logging.info(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Adam com beta1=0.5 — recomendado no paper DCGAN original.
    # beta1 mais baixo (0.5 vs 0.9 padrão) torna o optimizador menos
    # dependente do momentum, o que estabiliza o treino adversarial.
    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=c_train.get("lr_g", 2e-4),
        betas=(c_train.get("beta1", 0.5), c_train.get("beta2", 0.999)),
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=c_train.get("lr_d", 2e-4),
        betas=(c_train.get("beta1", 0.5), c_train.get("beta2", 0.999)),
    )

    # BCEWithLogitsLoss = sigmoid + BCE numa operação numericamente estável
    criterion = nn.BCEWithLogitsLoss()

    label_smoothing = c_train.get("label_smoothing", 0.1)
    n_steps_G = c_train.get("n_steps_G", 1)
    noise_std_start = c_train.get("noise_std_start", 0.0)
    noise_std_end = c_train.get("noise_std_end", 0.0)
    flip_prob = c_train.get("flip_prob", 0.0)
    grad_clip = c_train.get("grad_clip", 0.0)

    save_dir = c_train.get("save_dir", f"outputs/{config['experiment_name']}")
    checkpoint_dir = c_train.get("checkpoint_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Fixed noise: z fixo guardado uma vez, usado em todas as epochs
    # para gerar sempre as mesmas amostras e ver a evolução visual
    fixed_noise = torch.randn(16, latent_dim, device=device)

    history = []
    best_loss_G = float("inf")

    num_epochs = c_train["num_epochs"]

    for epoch in range(num_epochs):
        # Decay linear: começa em noise_std_start, termina em noise_std_end
        progress = epoch / max(num_epochs - 1, 1)
        noise_std = noise_std_start + (noise_std_end - noise_std_start) * progress

        metrics = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            loader=loaders["train"],
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
            criterion=criterion,
            device=device,
            latent_dim=latent_dim,
            label_smoothing=label_smoothing,
            n_steps_G=n_steps_G,
            noise_std=noise_std,
            flip_prob=flip_prob,
            grad_clip=grad_clip,
            epoch_idx=epoch,
        )

        history.append({
            "epoch": epoch + 1,
            "loss_D": metrics["loss_D"],
            "loss_G": metrics["loss_G"],
            "Dx": metrics["Dx"],
            "DGz": metrics["DGz"],
            "noise_std": noise_std,
        })

        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Loss D={metrics['loss_D']:.4f} | "
            f"Loss G={metrics['loss_G']:.4f} | "
            f"D(x)={metrics['Dx']:.3f} | "
            f"D(G(z))={metrics['DGz']:.3f} | "
            f"noise={noise_std:.3f}"
        )

        # Guardar amostras visuais a cada N epochs
        if (epoch + 1) % c_train.get("sample_every", 5) == 0:
            save_generated_samples(
                generator=generator,
                fixed_noise=fixed_noise,
                device=device,
                save_path=os.path.join(save_dir, f"generated_epoch_{epoch + 1}.png"),
            )

        # Checkpoint: guarda sempre que o Generator melhora
        if metrics["loss_G"] < best_loss_G:
            best_loss_G = metrics["loss_G"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "config": config,
                },
                os.path.join(checkpoint_dir, f"{config['experiment_name']}.pt"),
            )

    save_history(history, config["experiment_name"])
    plot_training(config["experiment_name"])
    return generator, discriminator


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
