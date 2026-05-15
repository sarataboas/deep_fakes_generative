import torch
import torch.nn as nn


def _apply_sn(layer, use_sn):
    """Aplica spectral normalization a uma camada se use_sn=True."""
    return nn.utils.spectral_norm(layer) if use_sn else layer


class Generator(nn.Module):
    """
    DCGAN Generator para imagens 64x64.

    Usa Upsample + Conv2d para evitar checkerboard artifacts.
    Suporta Spectral Normalization opcional nas camadas conv e FC.

    A spectral norm normaliza os pesos de cada camada pelo seu maior
    valor singular, controlando a constante de Lipschitz do modelo.
    No Generator, estabiliza o treino adversarial e melhora a qualidade
    do gradiente recebido do Discriminator.

    Input:  z — (batch_size, latent_dim)  amostrado de N(0, I)
    Output: imagem — (batch_size, 3, 64, 64) em [-1, 1]
    """

    def __init__(
        self,
        latent_dim: int = 128,
        feature_maps: int = 32,
        img_channels: int = 3,
        spectral_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        fm = feature_maps

        self.fc = _apply_sn(
            nn.Linear(latent_dim, fm * 8 * 4 * 4),
            spectral_norm,
        )

        # Construção por blocos para aplicar sn a cada Conv2d individualmente
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

        layers = []
        layers += upsample_block(fm * 8, fm * 4)   # 4x4   → 8x8
        layers += upsample_block(fm * 4, fm * 2)   # 8x8   → 16x16
        layers += upsample_block(fm * 2, fm)        # 16x16 → 32x32
        layers += upsample_block(fm, img_channels, last=True)  # 32x32 → 64x64

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.size(0), self.feature_maps * 8, 4, 4)
        return self.net(h)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator para imagens 64x64.

    Suporta Spectral Normalization opcional nas camadas conv e FC.

    No Discriminator, a spectral norm é especialmente importante:
    limita a capacidade do D de ser demasiado confiante, resolvendo
    o problema de domínio do D de forma matemática — sem precisar
    de truques como label smoothing ou dropout.

    Sem sigmoid na saída — usamos BCEWithLogitsLoss externamente.

    Input:  imagem — (batch_size, 3, 64, 64)
    Output: logit — (batch_size, 1)
    """

    def __init__(
        self,
        feature_maps: int = 32,
        img_channels: int = 3,
        spectral_norm: bool = False,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        fm = feature_maps

        def conv_block(in_ch, out_ch, bn=True):
            layers = [_apply_sn(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                spectral_norm,
            )]
            # BN desactivado para WGAN-GP: o gradient penalty é calculado
            # por amostra e o BN cria dependências entre amostras que o invalidam
            if bn and use_batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return layers

        layers = []
        layers += conv_block(img_channels, fm,      bn=False)  # 64x64 → 32x32  sem BN (padrão DCGAN)
        layers += conv_block(fm,           fm * 2)             # 32x32 → 16x16
        layers += conv_block(fm * 2,       fm * 4)             # 16x16 → 8x8
        layers += conv_block(fm * 4,       fm * 8)             # 8x8   → 4x4

        self.net = nn.Sequential(*layers)
        self.fc = _apply_sn(
            nn.Linear(fm * 8 * 4 * 4, 1),
            spectral_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)
