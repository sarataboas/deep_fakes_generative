import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.dataset import DeepFakeDataset
from src.preprocessing import get_train_transforms, get_test_transforms, get_vae_transforms, get_vae_val_transforms


def get_device():
    """Returns the best available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data_csv(data_path: str) -> pd.DataFrame:
    """Loads the metadata CSV and raises a clear error if the file is missing."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Metadata file not found: {data_path}")
    return pd.read_csv(data_path)


def get_data_split(df: pd.DataFrame, selected_split: str) -> pd.DataFrame:
    """Extracts one split (train / val / test) from the full metadata DataFrame."""
    return df[df["split"] == selected_split].reset_index(drop=True)


def get_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Computes per-sample weights for WeightedRandomSampler.

    Weight = total / (n_classes * count_per_class), i.e. inverse frequency.
    This ensures each class is sampled equally regardless of dataset imbalance.
    """
    counts = df["label"].value_counts().sort_index().values  # sort_index guarantees class order
    total = len(df)
    weights_per_class = total / (len(counts) * counts)
    sample_weights = [weights_per_class[int(label)] for label in df["label"]]
    return torch.tensor(sample_weights, dtype=torch.float32)


def build_dataloaders(config_data: dict, config_train: dict, config_preproc: dict) -> dict:
    """
    Builds train / val / test DataLoaders from a config dict.

    Supports two preprocessing modes:
      - 'classifier': ImageNet normalisation with augmentation
      - 'vae':        [-1, 1] normalisation, flip-only augmentation
    """
    df_full  = load_data_csv(config_data["metadata_path"])
    img_size = config_preproc.get("img_size", 224)
    preprocessing_type = config_preproc.get("type", "classifier")

    train_df = get_data_split(df_full, "train")
    val_df   = get_data_split(df_full, "val")
    test_df  = get_data_split(df_full, "test")

    # VAE and GAN configs may restrict training to a specific source (e.g. wiki-only)
    train_source = config_data.get("train_source")
    if train_source is not None:
        train_df = train_df[train_df["source_type"].isin(train_source)].reset_index(drop=True)
        val_df   = val_df[val_df["source_type"].isin(train_source)].reset_index(drop=True)
        test_df  = test_df[test_df["source_type"].isin(train_source)].reset_index(drop=True)

    if preprocessing_type == "vae":
        train_transform = get_vae_transforms(img_size=img_size)
        val_transform   = get_vae_val_transforms(img_size=img_size)
        test_transform  = get_vae_val_transforms(img_size=img_size)
    elif preprocessing_type == "classifier":
        train_transform = get_train_transforms(img_size=img_size)
        val_transform   = get_test_transforms(img_size=img_size)
        test_transform  = get_test_transforms(img_size=img_size)
    else:
        raise ValueError(
            f"Unknown preprocessing type: '{preprocessing_type}'. Use 'vae' or 'classifier'."
        )

    train_ds = DeepFakeDataset(train_df, transform=train_transform)
    val_ds   = DeepFakeDataset(val_df,   transform=val_transform)
    test_ds  = DeepFakeDataset(test_df,  transform=test_transform)

    sampler = None
    shuffle = True
    if config_train.get("use_weighted_sampler", False):
        weights = get_class_weights(train_df)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False  # shuffle and sampler are mutually exclusive in DataLoader

    batch_size = config_train.get("batch_size", 32)
    num_workers = 8
    pin_memory = torch.cuda.is_available()        # only beneficial with CUDA
    persistent_workers = num_workers > 0          # avoids worker respawn overhead between epochs

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=persistent_workers),
        "val":   DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=persistent_workers),
        "test":  DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=persistent_workers),
    }
