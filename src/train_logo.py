"""
Leave-One-Generator-Out (LOGO) training script.

Trains on real images + all fake generators except `held_out_generator`,
then evaluates the model on the full test set with a per-generator breakdown
to measure how well the model generalises to an unseen generator.

Usage:
    python -m src.train_logo --config configs/cnn_logo_no_insight.json
"""

import os
import json
import argparse
import logging

import torch
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.setup import get_device, get_class_weights
from src.dataset import DeepFakeDataset
from src.preprocessing import get_train_transforms, get_test_transforms
from src.train import Trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def build_logo_dataloaders(config_data, config_train, config_preproc, held_out_generator):
    df = pd.read_csv(config_data["metadata_path"])
    img_size   = config_preproc.get("img_size", 224)
    batch_size = config_train.get("batch_size", 32)
    num_workers = 8
    pin_memory  = torch.cuda.is_available()

    def filter_split(split):
        split_df = df[df["split"] == split]
        if split in ("train", "val"):
            # Remove held-out generator from training signal; keep all real images
            split_df = split_df[
                (split_df["label"] == 1) |
                (split_df["source_type"] != held_out_generator)
            ]
        return split_df.reset_index(drop=True)

    train_df = filter_split("train")
    val_df   = filter_split("val")
    test_df  = filter_split("test")  # all generators present — needed for per-generator eval

    train_generators = sorted(train_df[train_df["label"] == 0]["source_type"].unique().tolist())
    logging.info(f"LOGO: held out = '{held_out_generator}'")
    logging.info(f"Train fake generators: {train_generators}")
    logging.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = DeepFakeDataset(train_df, get_train_transforms(img_size))
    val_ds   = DeepFakeDataset(val_df,   get_test_transforms(img_size))
    test_ds  = DeepFakeDataset(test_df,  get_test_transforms(img_size))

    sampler, shuffle = None, True
    if config_train.get("use_weighted_sampler", False):
        weights = get_class_weights(train_df)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True),
        "val":   DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True),
        "test":  DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True),
    }
    return loaders, test_df


@torch.no_grad()
def evaluate_per_generator(trainer, test_df, img_size=224):
    trainer.model.eval()
    transform = get_test_transforms(img_size)
    results = {}

    for source in sorted(test_df["source_type"].unique()):
        sub = test_df[test_df["source_type"] == source].reset_index(drop=True)
        ds  = DeepFakeDataset(sub, transform)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

        all_probs, all_labels = [], []
        for batch in loader:
            imgs  = batch["image"].to(trainer.device)
            probs = torch.sigmoid(trainer.model(imgs)).squeeze(1).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(batch["label"].tolist())

        preds = [int(p >= 0.5) for p in all_probs]
        auc   = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else None

        results[source] = {
            "n":        len(sub),
            "accuracy": accuracy_score(all_labels, preds),
            "f1_fake":  f1_score(all_labels, preds, pos_label=0, average="binary", zero_division=0),
            "auc":      auc,
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    held_out = config["data"]["held_out_generator"]

    # Build the Trainer as usual — then swap its loaders for LOGO-filtered ones.
    # Trainer.__init__ builds full dataloaders internally (cheap, no images loaded)
    # which we immediately replace before fit() is called.
    trainer = Trainer(config)
    logo_loaders, test_df = build_logo_dataloaders(
        config["data"], config["training"], config["preprocessing"], held_out
    )
    trainer.loaders = logo_loaders

    trainer.fit()
    trainer.load_best()

    # Overall test metrics (all generators)
    test_metrics = trainer.evaluate("test")
    logging.info(
        f"Test (all) | AUC={test_metrics['auc']:.4f} | "
        f"Macro_F1={test_metrics['macro_f1']:.4f} | Acc={test_metrics['accuracy']:.4f}"
    )

    # Per-generator breakdown — the key output of this experiment
    img_size = config["preprocessing"].get("img_size", 224)
    per_gen  = evaluate_per_generator(trainer, test_df, img_size)

    logging.info("=== Per-generator breakdown ===")
    for source, m in per_gen.items():
        marker = "  << HELD OUT (never seen during training)" if source == held_out else ""
        logging.info(
            f"  {source:12s}  n={m['n']}  acc={m['accuracy']:.4f}  "
            f"f1_fake={m['f1_fake']:.4f}{marker}"
        )

    os.makedirs("results", exist_ok=True)
    out_path = f"results/{config['experiment_name']}_test_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"held_out_generator": held_out, "overall": test_metrics, "per_generator": per_gen}, f, indent=2)
    logging.info(f"Saved to {out_path}")
