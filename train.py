"""Train SkinCNN (ResNet18) on HAM10000 and save weights/best.pt."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ham10000"
WEIGHTS = ROOT / "weights" / "best.pt"
SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LabeledImageFolder(ImageFolder):
    def find_classes(self, directory):
        classes, _ = super().find_classes(directory)
        classes = [c for c in classes if c != "unlabeled"]
        return classes, {name: i for i, name in enumerate(classes)}


class SkinCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feats, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def train(
    data_dir=DATA,
    epochs=100,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4,
    patience=10,
    save_path=WEIGHTS,
):
    dev = pick_device()
    train_tf = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    full = LabeledImageFolder(str(data_dir), transform=train_tf)
    n_cls = len(full.class_to_idx)
    n_val = max(1, int(len(full) * 0.2))
    train_ds, val_split = random_split(
        full,
        [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    val_ds = Subset(
        LabeledImageFolder(str(data_dir), transform=eval_tf),
        val_split.indices,
    )

    train_labels = torch.tensor([full.targets[i] for i in train_ds.indices])
    counts = torch.bincount(train_labels, minlength=n_cls).float()
    class_weights = (counts.sum() / (counts.clamp(min=1) * n_cls)).to(dev)
    crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Oversample rare classes so batches aren't flooded by nevi.
    sample_weights = (1.0 / counts.clamp(min=1))[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    model = SkinCNN(n_cls).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best, stale = 0.0, 0
    print(f"device={dev} classes={n_cls} train={len(train_ds)} val={n_val}")
    print("class counts:", {name: int(counts[i]) for name, i in full.class_to_idx.items()})

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_ok, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
            tr_ok += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        tr_loss /= n
        tr_acc = tr_ok / n * 100

        model.eval()
        va_loss, va_ok, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                va_loss += crit(logits, y).item() * x.size(0)
                va_ok += (logits.argmax(1) == y).sum().item()
                n += x.size(0)
        va_loss /= n
        va_acc = va_ok / n * 100

        mark = ""
        if va_acc > best:
            best, stale, mark = va_acc, 0, " *"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": dict(full.class_to_idx),
                    "num_classes": n_cls,
                    "best_acc": va_acc,
                },
                save_path,
            )
            print(f"Best accuracy saved: {va_acc:.1f}%")
        else:
            stale += 1

        print(
            f"epoch {epoch}/{epochs}  "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.1f}%  "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.1f}%{mark}"
        )
        print(f"Current saved accuracy: {best:.1f}%")
        if stale >= patience:
            print(f"early stop (best_acc={best:.1f}%)")
            break

    print(f"best val_acc={best:.1f}% → {save_path}")
    return save_path


if __name__ == "__main__":
    train()
