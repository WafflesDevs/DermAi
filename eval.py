"""Predict a skin-lesion image from the CLI, or serve a small FastAPI demo."""

import sys
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
STATIC = ROOT / "static"
SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EVAL_TF = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
DISCLAIMER = (
    "SkinNN is a prototype for educational use only. "
    "It is not medical advice and does not diagnose disease. "
    "See a clinician for any skin concerns."
)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def predict(image, weights_path=WEIGHTS):
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    dev = pick_device()
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    idx_to_class = {i: name for name, i in ckpt["class_to_idx"].items()}
    n = ckpt.get("num_classes", len(idx_to_class))

    model = SkinCNN(n).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    x = EVAL_TF(image.convert("RGB")).unsqueeze(0).to(dev)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).squeeze(0).cpu()

    pred = int(probs.argmax())
    return {
        "class": idx_to_class[pred],
        "confidence": float(probs[pred]),
        "probabilities": {idx_to_class[i]: float(probs[i]) for i in range(n)},
    }



def as_percents(result):
    return {
        "class": result["class"],
        "confidence": round(result["confidence"] * 100, 1),
        "probabilities": {
            name: round(prob * 100, 1)
            for name, prob in result["probabilities"].items()
        },
        "disclaimer": DISCLAIMER,
    }


app = FastAPI(title="SkinNN")


@app.post("/predict")
async def api_predict(file: UploadFile = File(...)):
    if not WEIGHTS.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model weights missing. Train first with: python train.py",
        )

    data = await file.read()
    try:
        image = Image.open(BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image upload") from exc

    return as_percents(predict(image))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


def run_cli(image_path):
    result = as_percents(predict(image_path))
    print(f"Predicted: {result['class']} ({result['confidence']}%)")
    print("Chances:")
    ranked = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)
    for name, pct in ranked:
        print(f"  {name}: {pct}%")
    print(DISCLAIMER)


def run_web():
    print("SkinNN web demo → http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli(sys.argv[1])
    else:
        run_web()
