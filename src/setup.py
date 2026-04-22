import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.dataset import DeepFakeDataset
from src.preprocessing import get_train_transforms, get_test_transforms

def get_device():
    """Retorna o dispositivo disponível: CUDA, MPS ou CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_data_csv(data_path: str) -> pd.DataFrame:
    """Lê o ficheiro de metadados CSV."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Ficheiro não encontrado: {data_path}")
    df = pd.read_csv(data_path)
    return df

def get_data_split(df: pd.DataFrame, selected_split: str) -> pd.DataFrame:
    """Extrai o subset (train, val ou test) do DataFrame."""
    split_df = df[df["split"] == selected_split].reset_index(drop=True)
    return split_df

def get_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Calcula pesos para o WeightedRandomSampler."""
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    weight_per_class = {cls: total / count for cls, count in counts.items()}
    weights = df["label"].map(weight_per_class).values
    return torch.tensor(weights, dtype=torch.float32)

def build_dataloaders(config_data: dict, config_train: dict, config_preproc: dict) -> dict:
    """
    Cria os DataLoaders usando as secções 'data', 'training' e 'preprocessing' do JSON.
    """
    # 1. Carregar metadados
    df_full = load_data_csv(config_data["metadata_path"])
    img_size = config_preproc.get("img_size", 224)
    
    # 2. Criar subsets de dados
    train_df = get_data_split(df_full, 'train')
    val_df   = get_data_split(df_full, 'val')
    test_df  = get_data_split(df_full, 'test')

    # 3. Instanciar Datasets com o img_size correto
    train_ds = DeepFakeDataset(train_df, transform=get_train_transforms(img_size=img_size))
    val_ds   = DeepFakeDataset(val_df,   transform=get_test_transforms(img_size=img_size))
    test_ds  = DeepFakeDataset(test_df,  transform=get_test_transforms(img_size=img_size))

    # 4. Configurar Sampler se use_weighted_sampler for true
    sampler = None
    shuffle = True
    if config_train.get("use_weighted_sampler", False):
        weights = get_class_weights(train_df)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    # 5. Criar Loaders
    batch_size = config_train.get("batch_size", 32)
    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler),
        'val':   DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        'test':  DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    }
    
    return loaders