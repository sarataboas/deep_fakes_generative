import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Convolutional encoder for 64x64 RGB images.

    Input:
        x: (batch_size, 3, 64, 64)

    Output:
        mu:     (batch_size, latent_dim)
        logvar: (batch_size, latent_dim)
    """
    def __init__(
        self,
        img_channels: int = 3,
        feature_maps: int = 32,
        latent_dim: int = 128,
    ):
        super().__init__()

        self.feature_maps = feature_maps
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # 3 x 64 x 64 -> 32 x 32 x 32
            nn.Conv2d(img_channels, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 x 32 x 32 -> 64 x 16 x 16
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 x 16 x 16 -> 128 x 8 x 8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 x 8 x 8 -> 256 x 4 x 4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = feature_maps * 8 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
class Decoder(nn.Module):
    """
    Convolutional decoder for 64x64 RGB images.

    Input:
        z: (batch_size, latent_dim)

    Output:
        x_hat: (batch_size, 3, 64, 64)
    """

    def __init__(
        self,
        img_channels: int = 3,
        feature_maps: int = 32,
        latent_dim: int = 128,
    ):
        super().__init__()

        self.feature_maps = feature_maps
        self.latent_dim = latent_dim

        self.flatten_dim = feature_maps * 8 * 4 * 4

        self.fc = nn.Linear(latent_dim, self.flatten_dim)

        self.net = nn.Sequential(
            # 256 x 4 x 4 -> 128 x 8 x 8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(inplace=True),

            # 128 x 8 x 8 -> 64 x 16 x 16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),

            # 64 x 16 x 16 -> 32 x 32 x 32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True),

            # 32 x 32 x 32 -> 3 x 64 x 64
            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.size(0), self.feature_maps * 8, 4, 4)

        x_hat = self.net(h)
        return x_hat

class BaselineVAE(nn.Module):
    """
    Baseline convolutional VAE.

    Architecture:
        Input: 3 x 64 x 64
        Encoder: 4 convolutional blocks
        Latent dimension: 128
        Decoder: 4 transposed-convolutional blocks
        Output: 3 x 64 x 64 in [-1, 1]
    """

    def __init__(
        self,
        img_channels: int = 3,
        feature_maps: int = 32,
        latent_dim: int = 128,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(
            img_channels=img_channels,
            feature_maps=feature_maps,
            latent_dim=latent_dim,
        )

        self.decoder = Decoder(
            img_channels=img_channels,
            feature_maps=feature_maps,
            latent_dim=latent_dim,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar

    @torch.no_grad()
    def generate(self, num_samples: int, device: torch.device | str) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decoder(z)
        return samples