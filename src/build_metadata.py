import os
import json
import logging
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# Mapeamento das pastas (mantém-se como referência do projeto)
SOURCE_TYPE_DICT = {
    "wiki": "wiki",
    "inpainting": "inpainting",
    "insight": "insight",
    "text2image": "text2img",
}

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def build_metadata(data_dir, source_map):
    """Explora as pastas e cria a lista inicial de ficheiros."""
    records = []
    for source_type, folder_name in source_map.items():
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            logging.warning(f"Pasta não encontrada: {folder_path}")
            continue
        
        for root, _, files in os.walk(folder_path):
            for fname in sorted(files):
                if fname.lower().endswith(".jpg"):
                    filepath = os.path.join(root, fname)
                    rel_path = os.path.relpath(filepath, data_dir)
                    image_id = os.path.splitext(rel_path)[0].replace(os.sep, "_")
                    label = 1 if source_type == "wiki" else 0
                    
                    records.append({
                        "image_id": image_id,
                        "filepath": filepath,
                        "source_type": source_type,
                        "label": label,
                    })
    return pd.DataFrame(records)

def select_samples(df, samples_per_source):
    """Lógica original: garante N imagens por cada fonte (wiki, insight, etc)."""
    return df.groupby("source_type", group_keys=False).head(samples_per_source)

def split_data(df, train_size, val_size):
    """Split estratificado triplo (Train/Val/Test)."""
    df = df.copy()
    # Primeiro split: Treino vs Resto
    train_idx, temp_idx = train_test_split(
        df.index, train_size=train_size, stratify=df["source_type"], random_state=42
    )
    # Segundo split: Metade do resto para Val e metade para Test
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=0.5, stratify=df.loc[temp_idx, "source_type"], random_state=42
    )
    
    df["split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Caminho para o ficheiro JSON de config")
    args = parser.parse_args()

    # 1. Carregar valores do Config
    config = load_config(args.config)
    
    # Extraímos o que precisamos da secção "data"
    data_cfg = config["data"]
    data_dir = "data" # Ou podes adicionar "data_dir" ao JSON se quiseres
    samples = data_cfg.get("number_samples_per_class", 1000)
    train_pct = data_cfg.get("train_size", 0.7)
    val_pct = data_cfg.get("val_size", 0.15)
    output_csv = data_cfg.get("metadata_path", "data/metadata.csv")

    logging.info(f"A gerar metadados usando config: {args.config}")
    logging.info(f"Samples por fonte: {samples}")

    # 2. Processamento
    df = build_metadata(data_dir, SOURCE_TYPE_DICT)
    df = select_samples(df, samples)
    df = split_data(df, train_pct, val_pct)

    # 3. Guardar
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logging.info(f"Sucesso! Metadados guardados em {output_csv} ({len(df)} imagens).")

if __name__ == "__main__":
    main()