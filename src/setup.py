import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from src.dataset import DeepFakeDataset
from src.preprocessing import *


def get_device():

    """Returns the available device in the following order: CUDA, MPS, CPU."""

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device



def load_data_csv(data_path: str) -> pd.DataFrame:

    """Reads a .csv file. Returns a dataframe"""

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"There is no file in {data_path}")
    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Loading an empty dataframe")
    return df


def get_data_split(df: pd.DataFrame, selected_split: str) -> pd.DataFrame:

    """Gets the data split"""

    if selected_split not in ['train', 'test']:
        raise ValueError("Invalid split name! Please select 'train' or 'test' as split_name")
    split_df = df[df["split"] == selected_split].reset_index(drop=True)
    if split_df.empty:
        raise ValueError("Loading an empty dataframe")
    return split_df


def create_dataset(train_df:pd.DataFrame, test_df: pd.DataFrame) -> tuple[DeepFakeDataset, DeepFakeDataset]:
    
    """Creates the transformed dataset splits - train and test"""

    train_dataset = DeepFakeDataset(df=train_df, transform=get_train_transforms())
    test_dataset = DeepFakeDataset(df=test_df, transform=get_test_transforms())

    return train_dataset, test_dataset


def create_dataloaders(train_dataset: DeepFakeDataset, test_dataset: DeepFakeDataset, batch_size: int, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader