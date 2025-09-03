# Histopathology Image Classification (Benign vs Malignant)

A complete, ready-to-run project for classifying histopathology image patches using deep learning (PyTorch), with Grad-CAM explainability and a Streamlit demo app.

> Suited for portfolios and interviews. Easily extensible to whole-slide images (WSIs) via patching.

---

## Features
- ResNet/EfficientNet backbones with transfer learning
- Patch-level classification pipeline
- Albumentations-based augmentations
- Grad-CAM visual explanations
- Balanced sampling & metric reporting (Accuracy, F1, AUC)
- Config-driven training (YAML)
- Streamlit app for quick demos

---

## Datasets

You can use one of the following public datasets:

1. **PatchCamelyon (PCam)** (fastest to start)
   - GitHub: https://github.com/basveeling/pcam
   - Contains 96x96 patches extracted from CAMELYON16 WSIs.
   - Labels: 1 = tumor, 0 = normal.

2. **Kaggle - Histopathologic Cancer Detection**
   - https://www.kaggle.com/competitions/histopathologic-cancer-detection
   - 96x96 H&E stained patches of lymph node sections.
   - Labels in CSV; images named by id.

> Place images in the `data/` directory as described below.

### Folder Layout (for generic folder dataset)
```
data/
  train/
    0/  # benign
    1/  # malignant
  val/
    0/
    1/
  test/
    0/
    1/
```

Alternatively, for **Kaggle HCD**, use CSV + images in a single folder; see `src/dataset.py` for CSV mode.

---

## Quickstart

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Edit config
# configs/default.yaml

# 4) Train
python src/train.py --config configs/default.yaml

# 5) Inference on a folder of images
python src/inference.py --checkpoint checkpoints/best_model.pt --input_dir ./data/test/1 --output_dir ./outputs

# 6) Run Streamlit app
streamlit run app/streamlit_app.py
```

---

## Explainability (Grad-CAM)
We include `src/gradcam.py` for generating Grad-CAM heatmaps over input patches to visualize regions driving the prediction. The Streamlit app overlays Grad-CAM automatically on uploaded images.

---

## Project Structure
```
histopathology-classification/
├─ app/
│  └─ streamlit_app.py
├─ configs/
│  └─ default.yaml
├─ data/
│  └─ README_DATA.md
├─ src/
│  ├─ dataset.py
│  ├─ gradcam.py
│  ├─ inference.py
│  ├─ model.py
│  ├─ train.py
│  ├─ transforms.py
│  └─ utils.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```

---

## Notes
- For WSIs (50K×50K): use OpenSlide to patch tiles, then feed patches to this classifier. The same augmentations and model code apply.
- Consider stain normalization for domain robustness (see `transforms.py` for a Macenko-style placeholder).
- For best results, ensure class balancing (we include weighted loss and sampling options).

---

## License
MIT