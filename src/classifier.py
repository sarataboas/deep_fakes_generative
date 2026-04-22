import logging
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Imports internos do teu projeto
from src.preprocessing import get_test_transforms
from src.setup import get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# -------------------------------------------------------------------
# Modelos
# -------------------------------------------------------------------

class BaselineCNN(nn.Module):
    """
    CNN simples adaptada para aceitar kernel_size e dropout do JSON.
    """
    def __init__(self, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        # Calculamos o padding para manter as dimensões se necessário (ex: kernel 3 -> padding 1)
        padding = kernel_size // 2
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 112x112

            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 56x56

            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 28x28
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1), # NUM_CLASSES fixado em 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 adaptada para aceitar o dropout do JSON.
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        for param in backbone.parameters():
            param.requires_grad = False

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

        self.backbone = backbone
        self._frozen = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._frozen = False
        logging.info("Backbone unfrozen para fine-tuning.")


def build_model(model_config: dict) -> nn.Module:
    """
    Cria o modelo usando a secção 'model' do ficheiro JSON.
    """
    model_type = model_config.get("model_type", "baseline")
    dropout = model_config.get("dropout_rate", 0.3)
    
    if model_type == "baseline":
        kernel_size = model_config.get("kernel_size", 3)
        model = BaselineCNN(kernel_size=kernel_size, dropout=dropout)
    elif model_type == "efficientnet":
        pretrained = model_config.get("pretrained", True)
        model = EfficientNetClassifier(pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Modelo {model_type} desconhecido.")

    return model.to(get_device())

# -------------------------------------------------------------------
# Avaliação
# -------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    """Calcula métricas globais de performance."""
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    total_loss, total = 0.0, 0

    for batch in loader:
        # Nota: O teu DeepFakeDataset devolve um dicionário
        imgs, labels = batch["image"].to(device), batch["label"].to(device).unsqueeze(1).float()
        
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().flatten().astype(int))

    return {
        "loss":      total_loss / total,
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "auc":       roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else None,
        "conf_matrix": confusion_matrix(all_labels, all_preds).tolist()
    }

@torch.no_grad()
def evaluate_per_generator(model: nn.Module, metadata_path: str, device: torch.device) -> dict:
    """Análise detalhada por tipo de gerador (Insight, Inpainting, etc.)."""
    df = pd.read_csv(metadata_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    transform = get_test_transforms()
    model.eval()

    results = {}
    for source in test_df["source_type"].unique():
        sub = test_df[test_df["source_type"] == source]
        probs, labels = [], []

        for _, row in sub.iterrows():
            img = transform(Image.open(row["filepath"]).convert("RGB")).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(img)).item()
            probs.append(prob)
            labels.append(int(row["label"]))

        preds = [int(p >= 0.5) for p in probs]
        results[source] = {
            "acc": accuracy_score(labels, preds),
            "f1":  f1_score(labels, preds, zero_division=0)
        }
    return results