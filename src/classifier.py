"""
Discriminative model for real vs. fake face classification.

Pipeline:
    - Baseline: small custom CNN (fast to train, sets reference)
    - Main model: EfficientNet-B0 with transfer learning

label_mapping: {0: fake, 1: real}

Usage (quick start):
    from classifier import build_model, train_one_epoch, evaluate
    model = build_model("efficientnet", pretrained=True)
    # then plug into your training loop
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.datasets import ImageFolder
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# NOTE to teammate (preprocessing): 
#   - DeepFakeDataset below expects a metadata CSV with columns:
#       'filepath', 'label', 'split'  (as built by build_metadata.py)
#   - get_class_weights() is implemented here but requires those columns
#   - Please make sure preprocessing transforms are importable from preprocessing.py
#       as get_train_transforms() and get_test_transforms()

from preprocessing import get_train_transforms, get_test_transforms
from utils import get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# -------------------------------------------------------------------
# Configuration  (change these without touching the rest of the file)
# -------------------------------------------------------------------
IMG_SIZE      = 224
NUM_CLASSES   = 1       # binary: BCEWithLogitsLoss expects a single logit
DROPOUT_RATE  = 0.3
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 10
BATCH_SIZE    = 32
UNFREEZE_EPOCH = 3      # epoch at which backbone unfreezing starts (fine-tuning)


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
class DeepFakeDataset(torch.utils.data.Dataset):
    """
    Loads images from the metadata CSV produced by build_metadata.py.

    Args:
        metadata_path : path to metadata.csv
        split         : 'train' | 'val' | 'test'
        transform     : torchvision transform pipeline
    """
    def __init__(self, metadata_path: str, split: str, transform=None):
        df = pd.read_csv(metadata_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split='{split}' in {metadata_path}")

        logging.info(
            f"[{split}] {len(self.df)} samples "
            f"| real: {(self.df['label']==1).sum()} "
            f"| fake: {(self.df['label']==0).sum()}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Returns per-sample weights for WeightedRandomSampler.
        Useful to run a balanced-sampling experiment vs. the natural 1:3 ratio.
        """
        counts = self.df["label"].value_counts().to_dict()  # {0: n_fake, 1: n_real}
        total  = len(self.df)
        weight_per_class = {cls: total / count for cls, count in counts.items()}
        weights = self.df["label"].map(weight_per_class).values
        return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    metadata_path: str,
    batch_size: int = BATCH_SIZE,
    use_weighted_sampler: bool = False,   # set True to run the balanced experiment
    num_workers: int = 4,
) -> dict:
    """
    Builds train / val / test DataLoaders.

    Args:
        metadata_path        : path to metadata.csv
        batch_size           : samples per batch
        use_weighted_sampler : if True, oversamples the minority class (real)
                               to counter the 1:3 imbalance — use for experiments
        num_workers          : parallel data loading workers
    Returns:
        dict with keys 'train', 'val', 'test'
    """
    train_ds = DeepFakeDataset(metadata_path, "train", get_train_transforms(IMG_SIZE))
    val_ds   = DeepFakeDataset(metadata_path, "val",   get_test_transforms(IMG_SIZE))
    test_ds  = DeepFakeDataset(metadata_path, "test",  get_test_transforms(IMG_SIZE))

    if use_weighted_sampler:
        weights = train_ds.get_class_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {"train": train_loader, "val": val_loader, "test": test_loader}


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class BaselineCNN(nn.Module):
    """
    Small custom CNN — used only to establish a reference baseline.
    Deliberately simple: easy to explain, fast to train, sets a floor
    for the iterative improvement story.

    Architecture: 3 conv blocks (Conv → BN → ReLU → MaxPool) → MLP head
    """
    def __init__(self, dropout: float = DROPOUT_RATE):
        super().__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 112x112

            # block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 56x56

            # block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # 28x28
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 128x1x1
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B0 with transfer learning.
    Strategy:
        Phase 1 (epochs 0 → UNFREEZE_EPOCH-1): only the classifier head is trained.
        Phase 2 (epochs UNFREEZE_EPOCH → end):  full model fine-tuned at lower LR.

    The model exposes unfreeze_backbone() to switch phases from the training loop.
    """
    def __init__(self, pretrained: bool = True, dropout: float = DROPOUT_RATE):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # freeze all backbone params initially
        for param in backbone.parameters():
            param.requires_grad = False

        # replace the default classifier head
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, NUM_CLASSES),
        )

        self.backbone = backbone
        self._frozen = True
        logging.info("EfficientNet-B0 loaded. Backbone frozen — training head only.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_backbone(self, lr_backbone: float = 1e-5) -> list:
        """
        Unfreezes all backbone parameters for fine-tuning.
        Returns a param group list ready for an optimizer update.

        Call from the training loop at epoch == UNFREEZE_EPOCH.
        """
        if not self._frozen:
            return []
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._frozen = False
        logging.info("Backbone unfrozen — full fine-tuning enabled.")
        return [
            {"params": self.backbone.features.parameters(), "lr": lr_backbone},
            {"params": self.backbone.classifier.parameters(), "lr": LR},
        ]


def build_model(
    model_type: str = "efficientnet",
    pretrained: bool = True,
    dropout: float = DROPOUT_RATE,
) -> nn.Module:
    """
    Factory function.

    Args:
        model_type : 'baseline' | 'efficientnet'
        pretrained : only relevant for efficientnet
        dropout    : dropout rate for the classifier head
    Returns:
        nn.Module on the available device
    """
    if model_type == "baseline":
        model = BaselineCNN(dropout=dropout)
    elif model_type == "efficientnet":
        model = EfficientNetClassifier(pretrained=pretrained, dropout=dropout)
    else:
        raise ValueError(f"Unknown model_type='{model_type}'. Choose 'baseline' or 'efficientnet'.")

    device = get_device()
    logging.info(f"Using device: {device}")
    return model.to(device)


# -------------------------------------------------------------------
# Training & evaluation utilities
# -------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    pos_weight: torch.Tensor | None = None,
) -> dict:
    """
    Runs one full training epoch.

    Returns:
        dict with 'loss', 'acc'
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "val",
) -> dict:
    """
    Evaluates the model — computes loss, accuracy, F1, AUC, and per-class metrics.

    Returns:
        dict with all metrics (ready to log or save to CSV)
    """
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    total_loss, total = 0.0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().flatten().astype(int))

    metrics = {
        "split":     split_name,
        "loss":      total_loss / total,
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "auc":       roc_auc_score(all_labels, all_probs),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }

    logging.info(
        f"[{split_name}] loss={metrics['loss']:.4f} | "
        f"acc={metrics['accuracy']:.4f} | f1={metrics['f1']:.4f} | auc={metrics['auc']:.4f}"
    )
    return metrics


@torch.no_grad()
def evaluate_per_generator(
    model: nn.Module,
    metadata_path: str,
    device: torch.device,
    split: str = "test",
) -> dict:
    """
    Breaks down test performance by fake generator type (insight, inpainting, text2img).
    Real images are reported as a group too.

    This surfaces whether the model struggles with specific generators — 
    a key analysis point for the presentation.

    Returns:
        dict mapping source_type → metrics dict
    """
    df = pd.read_csv(metadata_path)
    df = df[df["split"] == split].reset_index(drop=True)
    transform = get_test_transforms(IMG_SIZE)
    model.eval()

    results = {}
    for source_type in df["source_type"].unique():
        sub = df[df["source_type"] == source_type].reset_index(drop=True)
        labels_list, preds_list, probs_list = [], [], []

        for _, row in sub.iterrows():
            img = Image.open(row["filepath"]).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            prob  = torch.sigmoid(model(img)).item()
            pred  = int(prob >= 0.5)
            label = int(row["label"])
            probs_list.append(prob)
            preds_list.append(pred)
            labels_list.append(label)

        results[source_type] = {
            "n_samples": len(sub),
            "accuracy":  accuracy_score(labels_list, preds_list),
            "f1":        f1_score(labels_list, preds_list, zero_division=0),
            "auc":       roc_auc_score(labels_list, probs_list) if len(set(labels_list)) > 1 else None,
        }
        logging.info(f"  [{source_type}] acc={results[source_type]['accuracy']:.4f} | f1={results[source_type]['f1']:.4f}")

    return results


# -------------------------------------------------------------------
# Training loop (full run)
# -------------------------------------------------------------------
def train(
    model_type: str = "efficientnet",
    metadata_path: str = "data/metadata.csv",
    use_weighted_sampler: bool = False,
    pos_weight_value: float | None = None,   # e.g. 3.0 for 1:3 imbalance experiment
    save_path: str = "checkpoints/best_model.pt",
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """
    Full training loop with:
      - validation-based early stopping (saves best model by val AUC)
      - backbone unfreezing at UNFREEZE_EPOCH (EfficientNet only)
      - per-epoch metric logging

    Returns:
        history — list of metric dicts, one per epoch
    """
    os.makedirs(os.path.dirname(save_path) or "checkpoints", exist_ok=True)
    device = get_device()

    model     = build_model(model_type)
    loaders   = build_dataloaders(metadata_path, batch_size, use_weighted_sampler)

    # pos_weight addresses class imbalance in the loss directly (experiment option)
    pw = torch.tensor([pos_weight_value]).to(device) if pos_weight_value else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history   = []
    best_auc  = 0.0

    for epoch in range(num_epochs):
        logging.info(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        # Unfreeze backbone at the right epoch (EfficientNet only)
        if (
            model_type == "efficientnet"
            and epoch == UNFREEZE_EPOCH
            and hasattr(model, "unfreeze_backbone")
        ):
            param_groups = model.unfreeze_backbone(lr_backbone=1e-5)
            if param_groups:
                optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs - epoch
                )

        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics   = evaluate(model, loaders["val"], criterion, device, split_name="val")
        scheduler.step()

        epoch_log = {"epoch": epoch + 1, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(epoch_log)

        # save best checkpoint by val AUC
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), save_path)
            logging.info(f"  New best model saved (val AUC={best_auc:.4f})")

    logging.info(f"\nTraining complete. Best val AUC: {best_auc:.4f}")
    return history


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    # --- Step 1: Baseline CNN ---
    logging.info("=== STEP 1: Baseline CNN ===")
    history_baseline = train(
        model_type="baseline",
        metadata_path="data/metadata.csv",
        save_path="checkpoints/baseline_best.pt",
        num_epochs=10,
    )

    # --- Step 2: EfficientNet-B0 with frozen backbone, then fine-tuning ---
    logging.info("\n=== STEP 2: EfficientNet-B0 (transfer learning) ===")
    history_efficientnet = train(
        model_type="efficientnet",
        metadata_path="data/metadata.csv",
        save_path="checkpoints/efficientnet_best.pt",
        num_epochs=NUM_EPOCHS,
    )

    # --- Step 3: Per-generator breakdown on test set ---
    logging.info("\n=== STEP 3: Per-generator evaluation ===")
    device = get_device()
    model = build_model("efficientnet")
    model.load_state_dict(torch.load("checkpoints/efficientnet_best.pt", map_location=device))
    per_gen = evaluate_per_generator(model, "data/metadata.csv", device, split="test")

    # --- Step 4 (experiment): weighted sampler vs natural imbalance ---
    # Uncomment to run the comparison experiment
    # logging.info("\n=== STEP 4: Balanced sampler experiment ===")
    # history_balanced = train(
    #     model_type="efficientnet",
    #     metadata_path="data/metadata.csv",
    #     use_weighted_sampler=True,
    #     save_path="checkpoints/efficientnet_balanced_best.pt",
    #     num_epochs=NUM_EPOCHS,
    # )