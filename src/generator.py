import torch
import torch.nn as nn


class BaselineVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for 64x64 RGB face images.
    """

    def __init__(self, latent_dim: int = 128, img_channels: int = 3, feature_maps: int = 32):
        super().__init__()

        self.latent_dim = latent_dim
        self.feature_maps = feature_maps

        # Encoder: 3 x 64 x 64 -> 256 x 4 x 4
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = feature_maps * 8 * 4 * 4

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        # Decoder: 256 x 4 x 4 -> 3 x 64 x 64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feature_maps, img_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(z.size(0), self.feature_maps * 8, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def generate(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)