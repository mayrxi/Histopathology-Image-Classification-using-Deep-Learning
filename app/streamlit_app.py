import os
import tempfile
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from pathlib import Path
import numpy as np
import cv2

from sys import path as sys_path
sys_path.append(str(Path(__file__).resolve().parents[1] / "src"))
from model import build_model
from gradcam import GradCAM

st.set_page_config(page_title="Histopathology Classifier", layout="centered")

st.title("🧪 Histopathology Patch Classifier")
st.write("Upload a patch image (e.g., 96×96 or 224×224). The app will predict benign vs malignant and show a Grad-CAM overlay.")

ckpt_path = st.text_input("Checkpoint path", "checkpoints/best_model.pt")
target_layer = st.text_input("Target layer (ResNet)", "layer4")

uploaded = st.file_uploader("Upload image", type=['png','jpg','jpeg','tif','tiff'])

if st.button("Run Inference") and uploaded is not None:
    if not os.path.exists(ckpt_path):
        st.error(f"Checkpoint not found: {ckpt_path}")
    else:
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu')
        cfg = ckpt['config']
        model = build_model(backbone=cfg['model']['backbone'],
                            num_classes=cfg['model']['num_classes'],
                            pretrained=False,
                            dropout=cfg['model']['dropout'])
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        tfm = T.Compose([
            T.Resize((cfg['data']['input_size'], cfg['data']['input_size'])),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        image = Image.open(uploaded).convert("RGB")
        x = tfm(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).numpy()[0]
        pred_class = int(probs.argmax())
        prob_pos = float(probs[1])

        st.write(f"**Prediction:** {'Malignant (1)' if pred_class==1 else 'Benign (0)'}")
        st.write(f"**Probability (class 1):** {prob_pos:.3f}")

        gradcam = GradCAM(model, target_layer_name=target_layer)
        cam, _ = gradcam(x, class_idx=pred_class)
        cam = cam.squeeze().numpy()

        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap = cv2.applyColorMap((cam_resized*255).astype('uint8'), cv2.COLORMAP_JET)
        overlay = (0.4*heatmap + 0.6*img_np).astype('uint8')
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Grad-CAM Overlay", use_container_width=True)