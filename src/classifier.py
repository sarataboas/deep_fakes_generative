import logging
import torch
import torch.nn as nn
from torchvision import models

from src.setup import get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# -------------------------------------------------------------------
# Modelos
# -------------------------------------------------------------------

class BaselineCNN(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        channels: list[int] = [32, 64, 128],
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        pad = kernel_size // 2 if padding is None else padding

        blocks = []
        in_ch = 3
        for out_ch in channels:
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
    

def build_model(model_config: dict) -> nn.Module:
    """
    Cria o modelo usando a secção 'model' do ficheiro JSON.
    """
    model_type = model_config.get("model_type", "baseline")
    # No teu JSON usas "dropout_rate", vamos garantir que o código lê isso:
    dropout = model_config.get("dropout_rate", 0.3) 
    
    if model_type == "baseline_cnn":
        model = BaselineCNN(
            kernel_size=model_config.get("kernel_size", 3),
            stride=model_config.get("stride", 1),
            padding=model_config.get("padding", None),
            channels=model_config.get("channels", [32, 64, 128]),
            hidden_dim=model_config.get("hidden_dim", 256),
            dropout=dropout,
        )
    else:
        raise ValueError(f"Modelo {model_type} desconhecido.")

    return model.to(get_device())
