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
# O que muda face ao DCGAN
# -------------------------------------------------------------------
#
# DCGAN usa Binary Cross-Entropy: o Discriminator aprende a classificar
# real/fake como probabilidade (0 ou 1). Quando o D fica muito bom,
# o gradiente que chega ao G através da sigmoid fica próximo de zero
# — o G não aprende porque o sinal é demasiado fraco.
#
# WGAN-GP usa Wasserstein distance: o Critic (nome correcto em WGAN)
# aprende a atribuir scores contínuos em vez de probabilidades.
# Nunca há sigmoid. O gradiente é sempre informativo, mesmo quando
# o Critic é muito bom.
#
# O "GP" (Gradient Penalty) substitui o weight clipping original do
# WGAN e força o Critic a ser 1-Lipschitz de forma mais estável.
#
# Diferenças práticas:
#   - Sem BCEWithLogitsLoss — loss é simplesmente a média dos scores
#   - Critic treina n_critic vezes por cada passo do Generator
#   - Sem label smoothing, sem label flipping
#   - Sem BatchNorm no Critic (BN é incompatível com gradient penalty)
#   - Métricas diferentes: Wasserstein distance em vez de D(x)/D(G(z))


# -------------------------------------------------------------------
# Gradient Penalty
# -------------------------------------------------------------------

def compute_gradient_penalty(critic, real_imgs, fake_imgs, device):
    """
    Calcula o gradient penalty do WGAN-GP.

    Interpola linearmente entre imagens reais e falsas e força o
    gradiente do Critic nessa interpolação a ter norma ≈ 1.

    Sem isto, o Critic poderia aprender funções muito "afiadas" que
    violam a condição de Lipschitz e tornam o treino instável.

    Penalidade = E[(||∇D(x̂)||₂ - 1)²]  onde  x̂ = α·real + (1-α)·fake
    """
    batch_size = real_imgs.size(0)

    # Alpha aleatório por imagem — amostra uniformemente no segmento real↔fake
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs.detach())
    interpolated = interpolated.requires_grad_(True)

    critic_score = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=critic_score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_score),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Flatten: (batch, C, H, W) → (batch, C*H*W) e calcula a norma L2
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


# -------------------------------------------------------------------
# Utilitários de imagem
# -------------------------------------------------------------------

def denormalize(imgs):
    return torch.clamp((imgs + 1.0) / 2.0, 0.0, 1.0)


def save_generated_samples(generator, fixed_noise, device, save_path):
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_noise)
    samples = denormalize(samples.cpu())
    grid = make_grid(samples, nrow=4)
    save_image(grid, save_path)
    generator.train()


# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------

def plot_training(experiment_name):
    csv_path = os.path.join("results", f"{experiment_name}.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    os.makedirs("plots", exist_ok=True)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"{experiment_name} — Treino (WGAN-GP)", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # Wasserstein distance estimate + loss G
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df["epoch"], df["loss_C"], label="Loss Critic", color="#e74c3c", linewidth=1.5)
    ax1.plot(df["epoch"], df["loss_G"], label="Loss Generator", color="#3498db", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Critic vs Loss Generator")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Wasserstein distance (sem GP)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df["epoch"], df["wasserstein_d"], color="#2ecc71", linewidth=1.5)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("W distance")
    ax2.set_title("Wasserstein Distance (E[C(real)] - E[C(fake)])")
    ax2.grid(alpha=0.3)

    # Gradient penalty
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df["epoch"], df["gradient_penalty"], color="#e67e22", linewidth=1.5)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("GP")
    ax3.set_title("Gradient Penalty")
    ax3.grid(alpha=0.3)

    out_path = os.path.join("plots", f"{experiment_name}_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Gráfico guardado em {out_path}")


# -------------------------------------------------------------------
# Loop de treino
# -------------------------------------------------------------------

def train_one_epoch(
    generator,
    critic,
    loader,
    optimizer_G,
    optimizer_C,
    device,
    latent_dim,
    n_critic,
    lambda_gp,
    grad_clip,
    epoch_idx,
):
    """
    Loop de treino WGAN-GP para uma epoch.

    Diferença fundamental face ao DCGAN:
    - O Critic treina n_critic vezes por cada passo do Generator.
      Em DCGAN era 1:1 — aqui é tipicamente 5:1.
      Porquê: o Critic precisa de estar bem treinado para fornecer
      um sinal de gradiente útil ao Generator.

    - A loss do Critic não é BCE mas sim:
        L_C = E[C(fake)] - E[C(real)] + lambda_gp * GP
      O Critic quer maximizar E[C(real)] - E[C(fake)], i.e., scores
      altos para reais e baixos para falsas.

    - A loss do Generator é simplesmente:
        L_G = -E[C(G(z))]
      O Generator quer que o Critic dê scores altos às suas imagens.

    Métricas:
    - wasserstein_d = E[C(real)] - E[C(fake)]: estimativa da distância
      de Wasserstein. Deve ser positiva e diminuir à medida que G melhora
      (as distribuições ficam mais próximas).
    - gradient_penalty: deve ficar próximo de 0 quando o Critic é
      bem comportado (norma dos gradientes ≈ 1).
    """
    generator.train()
    critic.train()

    total_loss_C = 0.0
    total_loss_G = 0.0
    total_wasserstein = 0.0
    total_gp = 0.0
    total_samples = 0
    g_steps = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx + 1} [WGAN-GP]", leave=False)

    for batch in pbar:
        real_imgs = batch["image"].to(device)
        batch_size = real_imgs.size(0)

        # ---- Passo 1: Treinar o Critic (n_critic vezes) ----
        for _ in range(n_critic):
            optimizer_C.zero_grad()

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z).detach()

            score_real = critic(real_imgs).mean()
            score_fake = critic(fake_imgs).mean()

            gp = compute_gradient_penalty(critic, real_imgs, fake_imgs, device)

            # Critic quer maximizar score_real - score_fake
            # Minimizamos o negativo: -(score_real - score_fake) = score_fake - score_real
            loss_C = score_fake - score_real + lambda_gp * gp

            loss_C.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
            optimizer_C.step()

        wasserstein_d = (score_real - score_fake).item()

        # ---- Passo 2: Treinar o Generator (1 vez) ----
        optimizer_G.zero_grad()

        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(z)
        loss_G = -critic(fake_imgs).mean()

        loss_G.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
        optimizer_G.step()

        total_loss_C += loss_C.item() * batch_size
        total_loss_G += loss_G.item() * batch_size
        total_wasserstein += wasserstein_d * batch_size
        total_gp += gp.item() * batch_size
        total_samples += batch_size
        g_steps += 1

        pbar.set_postfix({
            "C": f"{loss_C.item():.3f}",
            "G": f"{loss_G.item():.3f}",
            "W": f"{wasserstein_d:.3f}",
            "GP": f"{gp.item():.3f}",
        })

    return {
        "loss_C": total_loss_C / total_samples,
        "loss_G": total_loss_G / total_samples,
        "wasserstein_d": total_wasserstein / total_samples,
        "gradient_penalty": total_gp / total_samples,
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

    logging.info(f"=== WGAN-GP Experiment: {config['experiment_name']} ===")
    logging.info(f"Device: {device}")

    loaders = build_dataloaders(c_data, c_train, c_preproc)
    logging.info(f"Train batches: {len(loaders['train'])}")

    latent_dim = c_model.get("latent_dim", 128)
    feature_maps = c_model.get("feature_maps", 64)
    img_channels = c_model.get("img_channels", 3)
    use_sn = c_model.get("spectral_norm", False)

    generator = Generator(
        latent_dim=latent_dim,
        feature_maps=feature_maps,
        img_channels=img_channels,
        spectral_norm=use_sn,
    ).to(device)

    # Critic sem BatchNorm — incompatível com gradient penalty
    critic = Discriminator(
        feature_maps=feature_maps,
        img_channels=img_channels,
        spectral_norm=use_sn,
        use_batch_norm=False,
    ).to(device)

    logging.info(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    logging.info(f"Critic params:    {sum(p.numel() for p in critic.parameters()):,}")

    optimizer_G = torch.optim.Adam(
        generator.parameters(),
        lr=c_train.get("lr_g", 1e-4),
        betas=(c_train.get("beta1", 0.0), c_train.get("beta2", 0.9)),
    )
    optimizer_C = torch.optim.Adam(
        critic.parameters(),
        lr=c_train.get("lr_c", 1e-4),
        betas=(c_train.get("beta1", 0.0), c_train.get("beta2", 0.9)),
    )

    n_critic = c_train.get("n_critic", 5)
    lambda_gp = c_train.get("lambda_gp", 10.0)
    grad_clip = c_train.get("grad_clip", 0.0)

    save_dir = c_train.get("save_dir", f"outputs/{config['experiment_name']}")
    checkpoint_dir = c_train.get("checkpoint_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    fixed_noise = torch.randn(16, latent_dim, device=device)

    history = []
    best_wasserstein = float("inf")

    num_epochs = c_train["num_epochs"]

    for epoch in range(num_epochs):
        metrics = train_one_epoch(
            generator=generator,
            critic=critic,
            loader=loaders["train"],
            optimizer_G=optimizer_G,
            optimizer_C=optimizer_C,
            device=device,
            latent_dim=latent_dim,
            n_critic=n_critic,
            lambda_gp=lambda_gp,
            grad_clip=grad_clip,
            epoch_idx=epoch,
        )

        history.append({
            "epoch": epoch + 1,
            "loss_C": metrics["loss_C"],
            "loss_G": metrics["loss_G"],
            "wasserstein_d": metrics["wasserstein_d"],
            "gradient_penalty": metrics["gradient_penalty"],
        })

        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Loss C={metrics['loss_C']:.4f} | "
            f"Loss G={metrics['loss_G']:.4f} | "
            f"W={metrics['wasserstein_d']:.4f} | "
            f"GP={metrics['gradient_penalty']:.4f}"
        )

        if (epoch + 1) % c_train.get("sample_every", 5) == 0:
            save_generated_samples(
                generator=generator,
                fixed_noise=fixed_noise,
                device=device,
                save_path=os.path.join(save_dir, f"generated_epoch_{epoch + 1}.png"),
            )

        # Guarda checkpoint quando a distância de Wasserstein diminui
        # (G e Critic mais próximos do equilíbrio)
        if metrics["wasserstein_d"] < best_wasserstein:
            best_wasserstein = metrics["wasserstein_d"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "generator_state_dict": generator.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_C_state_dict": optimizer_C.state_dict(),
                    "config": config,
                },
                os.path.join(checkpoint_dir, f"{config['experiment_name']}.pt"),
            )

    save_history(history, config["experiment_name"])
    plot_training(config["experiment_name"])
    return generator, critic


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
