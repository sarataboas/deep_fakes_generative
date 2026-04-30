import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.classifier import build_model
from src.preprocessing import get_test_transforms


def predict(image_path: str, checkpoint_path: str) -> None:
    ckpt   = Path(checkpoint_path)
    config_path = ckpt.with_suffix(".json")

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Expected a JSON file alongside the checkpoint with the same name."
        )

    with open(config_path) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config["model"])
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    img_size  = config.get("preprocessing", {}).get("img_size", 224)
    transform = get_test_transforms(img_size=img_size)

    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    # label=1 → REAL, label=0 → FAKE; prob is P(real)
    label      = "REAL" if prob >= 0.5 else "FAKE"
    confidence = prob if prob >= 0.5 else 1.0 - prob

    print(f"Prediction:  {label}")
    print(f"Confidence:  {confidence * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFake single-image classifier")
    parser.add_argument("--image",      required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pt checkpoint file")
    args = parser.parse_args()

    predict(args.image, args.checkpoint)
