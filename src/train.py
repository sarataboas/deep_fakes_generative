import os
import json
import argparse
import logging
import torch
import torch.nn as nn
from tqdm import tqdm

# Imports do teu projeto
from src.setup import get_device, build_dataloaders
from src.classifier import build_model, evaluate, evaluate_per_generator
from src.utils import set_seed, save_history, plot_training 

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# -------------------------------------------------------------------
# Funções de Treino (Lógica preservada)
# -------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_idx):
    """Executa uma época de treino com barra de progresso."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    # Criamos a barra de progresso para o loader
    pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1} [Train]", leave=False)
    
    for batch in pbar:
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        
        # Atualiza a barra com a loss atual
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return {"loss": total_loss / total, "acc": correct / total}

def run_experiment(config):
    set_seed(config.get("seed", 42))
    device = get_device()
    
    c_data = config["data"]
    c_model = config["model"]
    c_train = config["training"]
    c_preproc = config["preprocessing"]

    logging.info(f"=== A iniciar experiência: {config['experiment_name']} ===")

    loaders = build_dataloaders(c_data, c_train, c_preproc)
    model = build_model(c_model)
    
    # --- CORREÇÃO DO ACESSO AO OPTIMIZER ---
    # Verifica se os dados estão dentro de "optimizer" ou na raiz de "training"
    opt_config = c_train.get("optimizer", c_train) 
    lr = opt_config.get("lr", 1e-3)
    wd = opt_config.get("weight_decay", 1e-4)
    pw_val = opt_config.get("pos_weight", 1.0)

    pw = torch.tensor([pw_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=wd
    )
    
    # --- CORREÇÃO DO SCHEDULER ---
    scheduler_type = c_train["scheduler"].get("type", "cosine")
    t_max = c_train["scheduler"].get("T_max", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    best_auc = 0.0
    history = []
    epochs_no_improve = 0  # Para o Early Stopping

    # 3. Loop de Treino
    for epoch in range(c_train["num_epochs"]):
        
        # Lógica de Unfreeze para EfficientNet
        if c_model["model_type"] == "efficientnet" and epoch == c_train.get("unfreeze_epoch", 3):
            model.unfreeze_backbone()
            # Reinicia otimizador para fine-tuning com LR reduzido
            optimizer = torch.optim.AdamW(model.parameters(), lr=c_train["optimizer"]["lr"]/10)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c_train["num_epochs"] - epoch)

        # Treino e Validação
        train_metrics = train_one_epoch(model, loaders['train'], optimizer, criterion, device, epoch)
        
        # Opcional: Adicionar barra também na avaliação
        logging.info(f"A validar época {epoch+1}...")
        val_metrics = evaluate(model, loaders['val'], criterion, device)
        scheduler.step()

        # Registo de métricas para o histórico
        epoch_log = {
            "epoch": epoch + 1,
            "loss": train_metrics['loss'],
            "val_loss": val_metrics['loss'],
            "val_auc": val_metrics['auc'],
            "val_f1": val_metrics['f1']
        }
        history.append(epoch_log)

        logging.info(f"Epoch {epoch+1}: Loss={epoch_log['loss']:.4f} | Val AUC={epoch_log['val_auc']:.4f}")

        # Lógica de Checkpoint e Early Stopping
        if epoch_log["val_auc"] > best_auc:
            best_auc = epoch_log["val_auc"]
            epochs_no_improve = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{config['experiment_name']}.pt")
            logging.info(f"  >> Checkpoint guardado (AUC: {best_auc:.4f})")
        else:
            epochs_no_improve += 1

        # Early Stopping check
        if c_train["early_stopping"]["enabled"]:
            if epochs_no_improve >= c_train["early_stopping"]["patience"]:
                logging.info(f"Early Stopping ativado após {epoch+1} épocas.")
                break

    # 4. Finalização: CSV e Gráficos
    save_history(history, config['experiment_name'])
    plot_training(history, config['experiment_name'])
    
    return model

# -------------------------------------------------------------------
# Ponto de Entrada (Argparse para carregar o JSON)
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino de Classificador DeepFake")
    parser.add_argument("--config", type=str, required=True, help="Caminho para o JSON de configuração")
    args = parser.parse_args()

    # Carregar configuração
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Correr experiência
    model = run_experiment(config)

    # Avaliação final por gerador usando o melhor modelo salvo
    logging.info("\n=== AVALIAÇÃO FINAL POR GERADOR ===")
    device = get_device()
    model.load_state_dict(torch.load(f"checkpoints/{config['experiment_name']}.pt", map_location=device))
    per_gen = evaluate_per_generator(model, config["data"]["metadata_path"], device)
    
    for gen, m in per_gen.items():
        logging.info(f"  {gen}: Acc={m['acc']:.4f} | F1={m['f1']:.4f}")