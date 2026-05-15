import math
import torch
import torch.nn as nn


def _apply_sn(layer, use_sn):
    return nn.utils.spectral_norm(layer) if use_sn else layer


class Generator(nn.Module):
    """
    Generator flexível para qualquer resolução de saída (64, 128, 256, ...).

    Parte de uma representação 4×4 e aplica n_ups blocos de Upsample+Conv
    até atingir img_size. O número de canais é dividido por 2 em cada bloco.

    Exemplos de canais com feature_maps=64:
      img_size=64  (n_ups=4): 512→256→128→64→3
      img_size=128 (n_ups=5): 512→256→128→64→32→3

    Input:  z — (batch_size, latent_dim)
    Output: imagem — (batch_size, img_channels, img_size, img_size) em [-1, 1]
    """

    def __init__(
        self,
        latent_dim: int = 128,
        feature_maps: int = 64,
        img_channels: int = 3,
        img_size: int = 64,
        spectral_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.feature_maps = feature_maps

        # Número de upsamples: 4×4 → img_size  (64→4, 128→5, 256→6)
        n_ups = int(math.log2(img_size // 4))
        fm = feature_maps

        self.fc = _apply_sn(
            nn.Linear(latent_dim, fm * 8 * 4 * 4),
            spectral_norm,
        )

        def upsample_block(in_ch, out_ch, last=False):
            layers = [nn.Upsample(scale_factor=2, mode="nearest")]
            layers.append(_apply_sn(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                spectral_norm,
            ))
            if not last:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout2d(dropout))
            else:
                layers.append(nn.Tanh())
            return layers

        # Canais: fm*8 → fm*4 → fm*2 → fm → fm//2 → ... → img_channels
        layers = []
        ch = fm * 8
        for i in range(n_ups - 1):
            out_ch = (fm * 8) >> (i + 1)  # divide por 2 a cada bloco
            layers += upsample_block(ch, out_ch)
            ch = out_ch
        layers += upsample_block(ch, img_channels, last=True)

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.size(0), self.feature_maps * 8, 4, 4)
        return self.net(h)


class Discriminator(nn.Module):
    """
    Discriminator/Critic flexível para qualquer resolução de entrada.

    Aplica n_downs blocos de Conv2d stride=2 até chegar a 4×4,
    depois um FC → 1. O número de canais duplica em cada bloco
    (até ao máximo de feature_maps*8).

    Exemplos de canais com feature_maps=64:
      img_size=64  (n_downs=4): 3→64→128→256→512→FC
      img_size=128 (n_downs=5): 3→64→128→256→512→512→FC

    Sem sigmoid na saída — loss calculada externamente (WGAN ou BCEWithLogits).
    """

    def __init__(
        self,
        feature_maps: int = 64,
        img_channels: int = 3,
        img_size: int = 64,
        spectral_norm: bool = False,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        fm = feature_maps
        # Número de downsamples stride=2 para chegar de img_size a 4×4
        n_downs = int(math.log2(img_size // 4))

        def conv_block(in_ch, out_ch, bn=True):
            layers = [_apply_sn(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                spectral_norm,
            )]
            # BN desactivado para WGAN-GP: gradient penalty calculado por amostra
            if bn and use_batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return layers

        # Primeiro bloco sem BN (padrão DCGAN)
        layers = []
        layers += conv_block(img_channels, fm, bn=False)
        ch = fm
        for i in range(1, n_downs):
            out_ch = min(fm * (2 ** i), fm * 8)  # cresce até fm*8, depois mantém
            layers += conv_block(ch, out_ch)
            ch = out_ch

        self.net = nn.Sequential(*layers)
        self.fc = _apply_sn(
            nn.Linear(fm * 8 * 4 * 4, 1),
            spectral_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)
