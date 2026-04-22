import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

def set_seed(seed=42):
    """Garante a reprodutibilidade dos resultados."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Seed definida para: {seed}")

def save_history(history, filename):
    """Guarda o histórico de treino num ficheiro CSV."""
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(history)
    path = f"results/{filename}.csv"
    df.to_csv(path, index=False)
    logging.info(f"Histórico guardado em: {path}")

def plot_training(history, model_name):
    """Gera e guarda gráficos de Loss e AUC."""
    os.makedirs("plots", exist_ok=True)
    df = pd.DataFrame(history)
    
    plt.figure(figsize=(12, 5))
    
    # Gráfico de Loss (Treino vs Validação)
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['loss'], label='Train Loss', marker='o')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title(f'Evolução da Loss - {model_name}')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de AUC
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['val_auc'], label='Val AUC', color='orange', marker='s')
    plt.xlabel('Época')
    plt.ylabel('AUC')
    plt.title(f'Evolução do AUC - {model_name}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_curves.png")
    plt.close()