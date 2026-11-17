# SkinNN

**Track:** Health  
**Team:** Ayaan Ali (A Tech)

Dermoscopic skin-lesion image classifier (7 HAM10000 classes) using a fine-tuned ResNet18

> **Prototype / not medical advice.** SkinNN is a student hackathon prototype for learning and experimentation only. It is **not** a medical device, **not** a diagnosis, and **not** a substitute for a licensed clinician. Do not use it to make health decisions.

## The problem

Skin lesion appearance is hard for non-specialists to interpret, and many public datasets exist for research on dermoscopic classification. We built a small, runnable prototype that takes a lesion image and returns a predicted class with probabilities—so we could practice transfer learning end-to-end, not to give clinical advice.

## What we built

- **In:** a dermoscopic RGB image  
- **Out:** predicted class name, confidence, and per-class chances (%)  
- **Stack:** PyTorch + torchvision ResNet18 (ImageNet pretrained), HAM10000 folder layout, FastAPI + a single HTML page

## How it works

**Data:** [HAM10000](https://doi.org/10.1038/sdata.2018.161) arranged as ImageFolder classes under `data/ham10000/` (7 labeled classes; `unlabeled` skipped).

**Model:** `SkinCNN` wraps torchvision `resnet18` with `ResNet18_Weights.IMAGENET1K_V1`, replaces the final FC with Dropout(0.5) + Linear(7). Training uses an 80/20 split, class-weighted cross-entropy, Adam + weight decay, augmentation, and early stopping on val loss. Best checkpoint: `weights/best.pt`.

**Demo:** `eval.py` either predicts one image from the CLI or serves `static/index.html` with `POST /predict`.

## How to run

```bash
# Install deps
uv sync

# Train once (needs data/ham10000 populated)
uv run python train.py

# Web demo
uv run python eval.py
# open http://127.0.0.1:8000

# CLI check
uv run python eval.py path/to/photo.jpg
```

**Data setup:** Download HAM10000 (e.g. from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) or a mirror), place images into class folders under `data/ham10000/`.

## Results

Honest numbers from our training runs:

- Early from-scratch CNN plateaued around **~77–78% validation accuracy**; train accuracy climbed much higher while val stuck → clear overfitting, so we kept the best-val-loss checkpoint and added augmentation, dropout, weight decay, class weights, and early stopping.
- We then switched to **ResNet18 transfer learning** (224px, full fine-tune). A `best.pt` checkpoint exists locally; re-report the exact ResNet val accuracy from your latest training log before submitting if you want a single headline number.
- **Limitations:** HAM10000 is class-imbalanced (melanocytic nevi dominate); overall accuracy can hide weak rare-class performance. No clinical validation. Images outside dermoscopic style will fail more often.

## Credits

| What | Link |
|------|------|
| HAM10000 dataset (Tschandl et al., Scientific Data 2018) | https://doi.org/10.1038/sdata.2018.161 · https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T |
| PyTorch | https://pytorch.org/ |
| torchvision ResNet18 + ImageNet weights (`ResNet18_Weights.IMAGENET1K_V1`) | https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html |
| FastAPI | https://fastapi.tiangolo.com/ |
| uv (package manager) | https://docs.astral.sh/uv/ |

Public datasets, open-source libraries, and pretrained models only. No personal patient data was collected for this project.
