#!/usr/bin/env bash
# Quick reference — common commands for this project
# Not meant to be run all at once; pick the line you need.

PYTHON=".venv/bin/python"

# ── Build metadata CSV ────────────────────────────────────────────────
$PYTHON -m src.build_metadata --config configs/baseline_chapter_2.json

# ── Train ─────────────────────────────────────────────────────────────
$PYTHON -m src.train --config configs/cnn_v5_warmup_gradclip.json

# ── Evaluate a checkpoint (test split + confusion matrix) ─────────────
$PYTHON -c "
import json
from src.train import Trainer

with open('configs/cnn_v4_early_stopping.json') as f:
    config = json.load(f)

trainer = Trainer(config)
trainer.load_best()

m = trainer.evaluate('test')
print(f"AUC={m['auc']:.4f}  Macro_F1={m['macro_f1']:.4f}  Acc={m['accuracy']:.4f}")

tn, fp = m['conf_matrix'][0]
fn, tp = m['conf_matrix'][1]
print()
print('Confusion matrix (rows=true, cols=predicted):')
print('                pred FAKE   pred REAL')
print(f'  true FAKE      {tn:>7}     {fp:>7}')
print(f'  true REAL      {fn:>7}     {tp:>7}')
"

# ── Per-generator evaluation ──────────────────────────────────────────
$PYTHON -c "
import json
from src.train import Trainer

with open('configs/cnn_v4_early_stopping.json') as f:
    config = json.load(f)

trainer = Trainer(config)
trainer.load_best()

for gen, m in trainer.evaluate_per_generator().items():
    print(f'{gen}: Acc={m[\"acc\"]:.4f}  Macro_F1={m[\"macro_f1\"]:.4f}  AUC={m[\"auc\"]:.4f}')
    print(m['crosstab'].to_string())
    print()
"

# ── Sanity check (prediction distribution + dummy baseline) ───────────
$PYTHON -m src.sanity_check --config configs/cnn_v5_warmup_gradclip.json

# ── Single image prediction ───────────────────────────────────────────
$PYTHON predict.py --image path/to/image.jpg --checkpoint checkpoints/cnn_v5_warmup_gradclip.pt

# ── Create a new experiment config from an existing one ───────────────
$PYTHON -m src.new_experiment \
    --from configs/cnn_v5_warmup_gradclip.json \
    --name cnn_v6_new_experiment \
    --set training.optimizer.lr=1e-4 \
    --set training.num_epochs=30
