import os
import json
import logging
import argparse
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

SEED = 42

# 70/15/15 — the only combination where val=15% and test=15% sum to ≤ 100%
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

FAKE_GENERATORS = ["inpainting", "insight", "text2image"]

SOURCE_FOLDER_MAP = {
    "wiki":       "wiki",
    "inpainting": "inpainting",
    "insight":    "insight",
    "text2image": "text2img",
}


# ──────────────────────────────────────────────────────────────────────
# Step 1 — Collect and balance
# ──────────────────────────────────────────────────────────────────────

def _scan_source(data_dir: str, folder_name: str, source_type: str) -> pd.DataFrame:
    """Returns a sorted DataFrame of all .jpg files found in one source folder."""
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.isdir(folder_path):
        logging.warning(f"Folder not found, skipping: {folder_path}")
        return pd.DataFrame(columns=["filepath", "source_type", "label"])

    records = []
    for root, _, files in os.walk(folder_path):
        for fname in sorted(files):
            if fname.lower().endswith(".jpg"):
                records.append({
                    "filepath":    os.path.join(root, fname),
                    "source_type": source_type,
                    "label":       1 if source_type == "wiki" else 0,
                })
    return pd.DataFrame(records)


def collect_and_balance(data_dir: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Scans all source folders and returns:
      - real_df: all wiki samples, shuffled with SEED
      - fake_gen_dfs: {generator_name: df} — each generator shuffled with SEED
                      and capped to the minimum count across generators

    Capping + per-source shuffle ensures:
      - Equal generator representation at every experiment size
      - Any trainval_fraction takes a deterministic prefix of each generator's pool
    """
    real_df = (
        _scan_source(data_dir, SOURCE_FOLDER_MAP["wiki"], "wiki")
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    fake_gen_dfs = {
        gen: _scan_source(data_dir, SOURCE_FOLDER_MAP[gen], gen)
              .sample(frac=1, random_state=SEED)
              .reset_index(drop=True)
        for gen in FAKE_GENERATORS
    }

    min_per_gen = min(len(df) for df in fake_gen_dfs.values())
    fake_gen_dfs = {gen: df.iloc[:min_per_gen].copy() for gen, df in fake_gen_dfs.items()}

    logging.info(
        f"Collected {len(real_df)} real | "
        f"{min_per_gen * len(FAKE_GENERATORS)} fake "
        f"({min_per_gen} per generator: {FAKE_GENERATORS})"
    )
    return real_df, fake_gen_dfs


# ──────────────────────────────────────────────────────────────────────
# Step 2 — Build fixed pools (train / val / test)
# ──────────────────────────────────────────────────────────────────────

def _split_source(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits one source's (already-shuffled) DataFrame into train/val/test
    using index-based slicing. This is deterministic and gives exact counts.
    """
    n       = len(df)
    n_train = round(TRAIN_FRAC * n)
    n_val   = round(VAL_FRAC * n)
    # Test gets whatever remains to avoid rounding drift
    return (
        df.iloc[:n_train].copy(),
        df.iloc[n_train : n_train + n_val].copy(),
        df.iloc[n_train + n_val :].copy(),
    )


def _interleave(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Round-robin interleave: [df0[0], df1[0], df2[0], df0[1], df1[1], ...]

    This ordering guarantees that taking the first k rows always yields
    ≈ equal samples from every source (exactly equal when k is a multiple
    of len(dfs)), which is the subset-consistency guarantee for fake generators.
    """
    max_len = max(len(df) for df in dfs)
    rows = [
        df.iloc[i]
        for i in range(max_len)
        for df in dfs
        if i < len(df)
    ]
    return pd.DataFrame(rows).reset_index(drop=True)


def make_pools(
    real_df: pd.DataFrame,
    fake_gen_dfs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits every source independently at TRAIN/VAL/TEST_FRAC, then
    assembles three ordered pools.

    Splitting each source independently guarantees that each pool has
    exactly the same number of samples from every generator — no rounding
    skew from splitting the combined fake DataFrame.

    Within each pool the fake generators are interleaved so that any prefix
    (i.e. any trainval_fraction) preserves per-generator balance.

    Returns (train_pool, val_pool, test_df) — all indexed from 0,
    split column not yet assigned.
    """
    real_train, real_val, real_test = _split_source(real_df)

    gen_trains, gen_vals, gen_tests = [], [], []
    for gen in FAKE_GENERATORS:
        tr, va, te = _split_source(fake_gen_dfs[gen])
        gen_trains.append(tr)
        gen_vals.append(va)
        gen_tests.append(te)

    fake_train = _interleave(gen_trains)
    fake_val   = _interleave(gen_vals)
    fake_test  = _interleave(gen_tests)

    train_pool = pd.concat([real_train, fake_train]).reset_index(drop=True)
    val_pool   = pd.concat([real_val,   fake_val  ]).reset_index(drop=True)
    test_df    = pd.concat([real_test,  fake_test ]).reset_index(drop=True)

    for name, pool in [("train", train_pool), ("val", val_pool), ("test", test_df)]:
        gen_counts = pool[pool["label"] == 0]["source_type"].value_counts().to_dict()
        logging.info(
            f"  {name}: {len(pool)} total "
            f"({pool['label'].sum()} real, {(pool['label']==0).sum()} fake) "
            f"| generators: {gen_counts}"
        )

    return train_pool, val_pool, test_df


# ──────────────────────────────────────────────────────────────────────
# Step 3 — Subsample with trainval_fraction
# ──────────────────────────────────────────────────────────────────────

def subsample_pool(pool: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """
    Takes the first `fraction` of real samples and the first `fraction` of
    fake samples independently, preserving real/fake balance.

    Because fake samples are interleaved across generators in the pool,
    taking any prefix also preserves per-generator balance (±1 sample).
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"trainval_fraction must be in (0.0, 1.0], got {fraction}")

    real_pool = pool[pool["label"] == 1]
    fake_pool = pool[pool["label"] == 0]

    n_real = max(1, round(fraction * len(real_pool)))
    n_fake = max(1, round(fraction * len(fake_pool)))

    return pd.concat([
        real_pool.iloc[:n_real],
        fake_pool.iloc[:n_fake],
    ]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def _log_summary(df: pd.DataFrame) -> None:
    total  = len(df)
    counts = df["split"].value_counts()
    logging.info(
        f"Final split — "
        f"train: {counts.get('train', 0)} ({counts.get('train', 0) / total:.1%}) | "
        f"val:   {counts.get('val',   0)} ({counts.get('val',   0) / total:.1%}) | "
        f"test:  {counts.get('test',  0)} ({counts.get('test',  0) / total:.1%})"
    )
    for split_name in ["train", "val", "test"]:
        fake_counts = (
            df[(df["split"] == split_name) & (df["label"] == 0)]["source_type"]
            .value_counts().to_dict()
        )
        logging.info(f"  {split_name} fake generators: {fake_counts}")


def main():
    parser = argparse.ArgumentParser(description="Build and split dataset metadata CSV")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    config   = load_config(args.config)
    data_cfg = config["data"]

    trainval_fraction = data_cfg.get("trainval_fraction", 1.0)
    output_csv        = data_cfg.get("metadata_path", "data/metadata_cropped.csv")

    logging.info(f"trainval_fraction={trainval_fraction} | output={output_csv}")

    # 1. Collect all samples; balance fake generators to min available count
    #real_df, fake_gen_dfs = collect_and_balance(data_dir="data")

    real_df, fake_gen_dfs = collect_and_balance(data_dir="face_crop_final")

    # 2. Build fixed full-size pools (70/15/15, interleaved for subset consistency)
    train_pool, val_pool, test_df = make_pools(real_df, fake_gen_dfs)

    # 3. Subsample train and val; test is never subsampled
    train_subset = subsample_pool(train_pool, trainval_fraction)
    val_subset   = subsample_pool(val_pool,   trainval_fraction)

    # 4. Assign split labels and combine
    train_subset = train_subset.copy(); train_subset["split"] = "train"
    val_subset   = val_subset.copy();   val_subset["split"]   = "val"
    test_df      = test_df.copy();      test_df["split"]      = "test"

    final_df = pd.concat([train_subset, val_subset, test_df]).reset_index(drop=True)

    _log_summary(final_df)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    logging.info(f"Saved {len(final_df)} rows to {output_csv}.")


if __name__ == "__main__":
    main()
