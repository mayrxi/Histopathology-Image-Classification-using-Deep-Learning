🧪 Histopathology Image Classification
📌 Overview

This project applies Deep Learning to classify histopathology images (microscope tissue samples) into benign (non-cancerous) and malignant (cancerous) categories.
It demonstrates how AI can support digital pathology by speeding up diagnosis, improving accuracy, and providing explainable results for medical experts.

The pipeline uses CNN architectures (ResNet, EfficientNet) with transfer learning, Grad-CAM visualizations for interpretability, and includes a Streamlit web app for easy demo.
🎯 Motivation

Traditional cancer diagnosis via histopathology requires trained pathologists to manually examine thousands of microscopic slides. This process is:

Time-consuming ⏳

Prone to inter-observer variability 👨‍⚕️👩‍⚕️

Difficult to scale in resource-limited settings 🌍

By leveraging Deep Learning & Computer Vision, we can:

Assist doctors with AI-driven second opinions

Highlight regions of interest (via Grad-CAM)

Improve diagnostic efficiency and consistency

⚙️ Features

Patch-level classification (benign vs malignant)

Pretrained ResNet / EfficientNet backbones

Data augmentations for generalization (flips, rotations, color jitter)

Grad-CAM explainability to highlight regions influencing predictions

Config-driven training (YAML for easy experimentation)

Streamlit app for real-time demo (upload → prediction + heatmap overlay)

📂 Dataset

This project supports multiple datasets:

PatchCamelyon (PCam) – GitHub

96×96 patches extracted from CAMELYON16 breast cancer slides

327,680 labeled patches (balanced benign/malignant)

Kaggle – Histopathologic Cancer Detection – Competition Link

96×96 H&E stained tissue patches

CSV file with image IDs and labels (0 = benign, 1 = malignant)

📌 Place data under data/ as explained in data/README_DATA.md
.

🚀 Quickstart
# 1) Setup environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Train model
python src/train.py --config configs/default.yaml

# 3) Run inference with Grad-CAM overlays
python src/inference.py --checkpoint checkpoints/best_model.pt --input_dir data/test/1 --output_dir outputs

# 4) Launch web demo
streamlit run app/streamlit_app.py

📊 Example Output

Prediction: Malignant (probability = 0.93)
Grad-CAM Overlay: Highlights tumor regions that influenced the decision.

(Insert sample images here when available)

🛠 Tech Stack

Python, PyTorch, Torchvision – deep learning

Albumentations – image augmentations

OpenSlide – (optional) for whole-slide images

scikit-learn – metrics (AUC, F1)

Streamlit – interactive web demo

YAML configs – reproducible experiments

🔮 Extensions

Adapt pipeline to whole-slide images (WSIs) by tiling patches

Add stain normalization (Macenko/Vahadane) for domain robustness

Explore Vision Transformers (ViTs) for higher accuracy

Deploy as a cloud-based pathology assistant tool

📖 References

Bandi, P., et al. (2018). CAMELYON16: Grand Challenge on Cancer Metastasis Detection in Lymph Node Histopathology Images.

Kaggle: Histopathologic Cancer Detection

📜 License
