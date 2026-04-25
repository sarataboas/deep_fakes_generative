import os
import json
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from src.setup import get_device, build_dataloaders
from src.classifier import build_model
from src.preprocessing import get_test_transforms
from src.utils import set_seed, save_history, plot_training

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class Trainer:
    """
    Manages the full lifecycle of an experiment: setup, training, evaluation,
    and checkpointing. Configured entirely from a JSON config dict.
    """

    def __init__(self, config: dict):
        self.config = config
        self.name = config["experiment_name"]

        # Shortcuts to config sections
        self.c_data   = config["data"]
        self.c_model  = config["model"]
        self.c_train  = config["training"]
        self.c_preproc = config["preprocessing"]

        set_seed(config.get("seed", 42))
        self.device = get_device()

        self.loaders = build_dataloaders(self.c_data, self.c_train, self.c_preproc)
        self.model   = build_model(self.c_model)

        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(self.c_train["num_epochs"])

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_criterion(self) -> nn.Module:
        opt_cfg = self.c_train.get("optimizer", {})
        # pos_weight > 1 up-weights the fake class; useful for imbalanced datasets
        pw = torch.tensor([opt_cfg.get("pos_weight", 1.0)]).to(self.device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = self.c_train.get("optimizer", {})
        # filter() skips frozen params (relevant when EfficientNet backbone is frozen)
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        return torch.optim.AdamW(
            trainable,
            lr=opt_cfg.get("lr", 1e-3),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
        )

    def _build_scheduler(self, t_max: int) -> torch.optim.lr_scheduler.LRScheduler:
        sched_cfg = self.c_train.get("scheduler", {})
        # Only cosine is supported for now; extend here if needed
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=sched_cfg.get("T_max", t_max),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch_idx: int) -> dict:
        """Runs one full pass over the training set. Returns loss and accuracy."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(self.loaders["train"], desc=f"Epoch {epoch_idx + 1} [Train]", leave=False)
        for batch in pbar:
            imgs   = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(1).float()

            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds   = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {"loss": total_loss / total, "acc": correct / total}

    def fit(self) -> None:
        """
        Full training loop with:
          - Per-epoch validation and metric logging
          - Best-model checkpointing (based on val AUC)
          - Early stopping
          - EfficientNet backbone unfreeze at a configured epoch
        """
        num_epochs  = self.c_train["num_epochs"]
        es_cfg      = self.c_train.get("early_stopping", {})
        es_enabled  = es_cfg.get("enabled", False)
        es_patience = es_cfg.get("patience", 5)

        best_auc        = 0.0
        epochs_no_improve = 0
        history         = []

        logging.info(f"=== Starting: {self.name} | device: {self.device} ===")

        for epoch in range(num_epochs):

            # Unfreeze EfficientNet backbone mid-training for fine-tuning with a lower LR
            if self.c_model["model_type"] == "efficientnet":
                unfreeze_at = self.c_train.get("unfreeze_epoch", 3)
                if epoch == unfreeze_at:
                    self.model.unfreeze_backbone()
                    ft_lr = self.c_train["optimizer"]["lr"] / 10
                    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=ft_lr)
                    self.scheduler = self._build_scheduler(t_max=num_epochs - epoch)

            train_metrics = self._train_one_epoch(epoch)
            val_metrics   = self.evaluate("val")
            self.scheduler.step()

            epoch_log = {
                "epoch":    epoch + 1,
                "loss":     train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_auc":  val_metrics["auc"],
                "val_f1":   val_metrics["f1"],
            }
            history.append(epoch_log)

            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"loss={epoch_log['loss']:.4f} | "
                f"val_auc={epoch_log['val_auc']:.4f} | "
                f"val_f1={epoch_log['val_f1']:.4f}"
            )

            # Save checkpoint whenever val AUC improves
            if epoch_log["val_auc"] > best_auc:
                best_auc = epoch_log["val_auc"]
                epochs_no_improve = 0
                self._save_checkpoint()
                logging.info(f"  >> Checkpoint saved (AUC={best_auc:.4f})")
            else:
                epochs_no_improve += 1

            if es_enabled and epochs_no_improve >= es_patience:
                logging.info(f"Early stopping after {epoch + 1} epochs.")
                break

        save_history(history, self.name)
        plot_training(history, self.name)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> dict:
        """
        Evaluates the model on a given split ('train', 'val', or 'test').
        Returns a dict with loss, accuracy, precision, recall, F1, AUC, and confusion matrix.
        """
        self.model.eval()
        all_labels, all_probs, all_preds = [], [], []
        total_loss, total = 0.0, 0

        for batch in self.loaders[split]:
            imgs   = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(1).float()

            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            total      += imgs.size(0)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten().astype(int))

        # AUC is undefined when only one class is present in the batch (e.g. tiny splits)
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else None

        return {
            "loss":        total_loss / total,
            "accuracy":    accuracy_score(all_labels, all_preds),
            "precision":   precision_score(all_labels, all_preds, zero_division=0),
            "recall":      recall_score(all_labels, all_preds, zero_division=0),
            "f1":          f1_score(all_labels, all_preds, zero_division=0),
            "auc":         auc,
            "conf_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        }

    @torch.no_grad()
    def evaluate_per_generator(self) -> dict:
        """
        Runs inference on the test split grouped by 'source_type' (e.g. Inpainting, Insight).
        Useful for understanding which generator fools the model most.
        """
        import pandas as pd

        self.model.eval()
        transform = get_test_transforms()

        df = pd.read_csv(self.c_data["metadata_path"])
        test_df = df[df["split"] == "test"].reset_index(drop=True)

        results = {}
        for source in test_df["source_type"].unique():
            sub = test_df[test_df["source_type"] == source]
            probs, labels = [], []

            for _, row in sub.iterrows():
                img = (
                    transform(Image.open(row["filepath"]).convert("RGB"))
                    .unsqueeze(0)
                    .to(self.device)
                )
                probs.append(torch.sigmoid(self.model(img)).item())
                labels.append(int(row["label"]))

            preds = [int(p >= 0.5) for p in probs]
            results[source] = {
                "acc": accuracy_score(labels, preds),
                "f1":  f1_score(labels, preds, zero_division=0),
            }

        return results

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), f"checkpoints/{self.name}.pt")

    def load_best(self) -> None:
        """Loads the best checkpoint saved during fit()."""
        path = f"checkpoints/{self.name}.pt"
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logging.info(f"Loaded best checkpoint from {path}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFake Classifier Training")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    trainer = Trainer(config)
    trainer.fit()

    # Reload best checkpoint and run final test evaluation
    trainer.load_best()
    test_metrics = trainer.evaluate("test")
    logging.info(
        f"Test | AUC={test_metrics['auc']:.4f} | "
        f"F1={test_metrics['f1']:.4f} | "
        f"Acc={test_metrics['accuracy']:.4f}"
    )

    # Per-generator breakdown on the test set
    logging.info("=== Per-generator evaluation ===")
    per_gen = trainer.evaluate_per_generator()
    for gen, m in per_gen.items():
        logging.info(f"  {gen}: Acc={m['acc']:.4f} | F1={m['f1']:.4f}")
