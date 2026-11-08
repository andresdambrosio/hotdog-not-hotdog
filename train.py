"""Entrena un clasificador Hot Dog / Not Hot Dog con transfer learning (ConvNeXt-Tiny).

Uso:
    .venv/bin/python train.py --epochs 10
"""
import argparse
import json
import random
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CLASSES = ["not_hot_dog", "hot_dog"]  # índice 1 = hot dog
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMG_SIZE = 224


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def eval_transform():
    return T.Compose([
        T.Resize(int(IMG_SIZE * 1.14)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def train_transform():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25),
    ])


class ImageList(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def load_items():
    items = []
    for label, cls in enumerate(CLASSES):
        for p in sorted((DATA_DIR / cls).glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                items.append((str(p), label))
            except Exception:
                print(f"Imagen corrupta ignorada: {p}")
    return items


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels, loss_sum = [], [], 0.0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * len(y)
        preds += logits.argmax(1).cpu().tolist()
        labels += y.cpu().tolist()
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    return loss_sum / len(labels), acc, preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Dispositivo: {device}")

    items = load_items()
    labels = [l for _, l in items]
    train_items, rest = train_test_split(items, test_size=0.3, stratify=labels, random_state=args.seed)
    val_items, test_items = train_test_split(rest, test_size=0.5, stratify=[l for _, l in rest], random_state=args.seed)
    print(f"Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")

    dl_kw = dict(batch_size=args.batch_size, num_workers=4, persistent_workers=True)
    train_dl = DataLoader(ImageList(train_items, train_transform()), shuffle=True, drop_last=True, **dl_kw)
    val_dl = DataLoader(ImageList(val_items, eval_transform()), **dl_kw)
    test_dl = DataLoader(ImageList(test_items, eval_transform()), **dl_kw)

    model = timm.create_model(args.model, pretrained=True, num_classes=2, drop_path_rate=0.1).to(device)

    # LR más bajo para el backbone preentrenado, más alto para la cabeza nueva
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * 10},
    ], weight_decay=0.05)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.lr * 10], total_steps=steps, pct_start=0.1)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, ckpt = 0.0, ROOT / "hotdog_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        bar = tqdm(train_dl, desc=f"Época {epoch:2d}/{args.epochs}", unit="batch",
                   bar_format="{l_bar}{bar:30}{r_bar}")
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)
            bar.set_postfix(loss=f"{run_loss / n:.4f}")
        val_loss, val_acc, _, _ = evaluate(model, val_dl, device)
        tqdm.write(f"   ↳ train_loss {run_loss / n:.4f} | val_loss {val_loss:.4f} | "
                   f"val_acc {val_acc:.4f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({"model_name": args.model, "classes": CLASSES,
                        "state_dict": model.state_dict()}, ckpt)

    print(f"\nMejor val_acc: {best_acc:.4f} -> guardado en {ckpt.name}")
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    test_loss, test_acc, preds, ys = evaluate(model, test_dl, device)
    print(f"\n=== TEST (imágenes nunca vistas) ===\nAccuracy: {test_acc:.4f}")
    print(classification_report(ys, preds, target_names=CLASSES, digits=4))
    print("Matriz de confusión (filas=real, columnas=predicho):")
    print(confusion_matrix(ys, preds))

    (ROOT / "metrics.json").write_text(json.dumps(
        {"model": args.model, "val_acc": best_acc, "test_acc": test_acc}, indent=2))


if __name__ == "__main__":
    main()
