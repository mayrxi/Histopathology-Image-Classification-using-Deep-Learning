import os
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from model import build_model
from gradcam import GradCAM
from utils import ensure_dir

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pt')
    p.add_argument('--input_dir', type=str, required=True, help='Folder with images to predict')
    p.add_argument('--output_dir', type=str, default='outputs')
    p.add_argument('--target_layer', type=str, default='layer4')  # for ResNet
    return p.parse_args()

def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    return ckpt['model_state_dict'], ckpt['config']

def main():
    args = parse_args()
    state_dict, cfg = load_checkpoint(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone=cfg['model']['backbone'],
                        num_classes=cfg['model']['num_classes'],
                        pretrained=False,
                        dropout=cfg['model']['dropout']).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    tfm = T.Compose([
        T.Resize((cfg['data']['input_size'], cfg['data']['input_size'])),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    ensure_dir(args.output_dir)
    gradcam = GradCAM(model, target_layer_name=args.target_layer)

    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
            continue
        path = os.path.join(args.input_dir, fname)
        img = Image.open(path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(probs.argmax())
        prob_pos = float(probs[1])

        # Grad-CAM
        cam, _ = gradcam(x, class_idx=pred_class)
        cam = cam.squeeze().cpu().numpy()

        # Save overlay
        import cv2
        import numpy as np
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap = cv2.applyColorMap((cam_resized*255).astype('uint8'), cv2.COLORMAP_JET)
        overlay = (0.4*heatmap + 0.6*img_np).astype('uint8')

        out_path = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}_pred{pred_class}_p{prob_pos:.2f}.jpg")
        cv2.imwrite(out_path, overlay)
        print(f"Saved: {out_path} (pred={pred_class}, prob_pos={prob_pos:.3f})")

if __name__ == '__main__':
    main()