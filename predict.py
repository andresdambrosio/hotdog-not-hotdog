"""Clasifica una o varias imágenes: ¿hot dog o no hot dog?

Uso:
    .venv/bin/python predict.py foto1.jpg foto2.png
"""
import sys
from pathlib import Path

import timm
import torch
from PIL import Image

from train import eval_transform, get_device

ROOT = Path(__file__).parent


def main(paths):
    device = get_device()
    ckpt_path = ROOT / "hotdog_model.pt"
    if not ckpt_path.exists():
        sys.exit(f"No encuentro {ckpt_path.name}. Descargalo del Release de GitHub "
                 "o entrenalo con: python train.py")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    tf = eval_transform()

    for p in paths:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model(x).softmax(1)[0, 1].item()  # prob. de hot dog
        verdict = "🌭 HOT DOG" if prob >= 0.5 else "❌ NOT HOT DOG"
        print(f"{p}: {verdict}  (p(hot dog) = {prob:.3f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Uso: python predict.py imagen1.jpg [imagen2.jpg ...]")
    main(sys.argv[1:])
