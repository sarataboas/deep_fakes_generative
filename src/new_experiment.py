"""
Creates a new experiment config from an existing one.

Usage:
    # Minimal — just clone and rename
    python -m src.new_experiment --from configs/baseline_chapter_2.json --name cnn_v3_test

    # With parameter overrides (dot notation, any depth)
    python -m src.new_experiment --from configs/baseline_chapter_2.json --name cnn_v3_low_lr \\
        --set training.optimizer.lr=1e-4 \\
        --set training.num_epochs=50 \\
        --set model.dropout_rate=0.3
"""
import argparse
import json
import sys
from pathlib import Path


def _set_nested(d: dict, key_path: str, raw_value: str) -> None:
    """Sets d[a][b][c] = value given key_path='a.b.c' and a raw string value."""
    keys = key_path.split(".")
    node = d
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            print(f"ERROR: key path '{key_path}' does not exist in config.", file=sys.stderr)
            sys.exit(1)
        node = node[k]

    # Parse the raw string into the right Python type
    leaf_key = keys[-1]
    if leaf_key not in node:
        print(f"ERROR: key '{key_path}' does not exist in config.", file=sys.stderr)
        sys.exit(1)

    original = node[leaf_key]
    try:
        if isinstance(original, bool):
            value = raw_value.lower() in ("true", "1", "yes")
        elif isinstance(original, int):
            value = int(raw_value)
        elif isinstance(original, float):
            value = float(raw_value)
        elif isinstance(original, list):
            value = json.loads(raw_value)
        else:
            value = raw_value
    except (ValueError, json.JSONDecodeError):
        value = raw_value   # fall back to string if parsing fails

    node[leaf_key] = value


def main():
    parser = argparse.ArgumentParser(description="Clone a config with a new experiment name")
    parser.add_argument("--from",  dest="base",  required=True, help="Path to the base config JSON")
    parser.add_argument("--name",  required=True, help="New experiment name (also becomes the filename)")
    parser.add_argument("--set",   action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value using dot notation (repeatable)")
    args = parser.parse_args()

    base_path = Path(args.base)
    if not base_path.exists():
        print(f"ERROR: base config not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    with open(base_path) as f:
        config = json.load(f)

    config["experiment_name"] = args.name

    for override in args.set:
        if "=" not in override:
            print(f"ERROR: --set value must be KEY=VALUE, got: '{override}'", file=sys.stderr)
            sys.exit(1)
        key_path, raw_value = override.split("=", 1)
        _set_nested(config, key_path.strip(), raw_value.strip())

    out_path = base_path.parent / f"{args.name}.json"
    if out_path.exists():
        print(f"ERROR: {out_path} already exists. Choose a different name.", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Created {out_path}")
    if args.set:
        print("Overrides applied:")
        for o in args.set:
            print(f"  {o}")


if __name__ == "__main__":
    main()
