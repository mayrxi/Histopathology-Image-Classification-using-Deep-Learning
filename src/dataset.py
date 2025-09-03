import os
import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Expect subfolders 0/ and 1/
        self.samples = []
        for label in ["0", "1"]:
            class_dir = os.path.join(root_dir, label)
            if not os.path.isdir(class_dir):
                continue
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
                for p in glob.glob(os.path.join(class_dir, ext)):
                    self.samples.append((p, int(label)))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root_dir}. Expected subfolders '0' and '1'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, os.path.basename(path)

class CSVDataset(Dataset):
    def __init__(self, csv_path, image_dir, image_col="id", label_col="label", transform=None, ext=".tif"):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.ext = ext

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row[self.image_col]
        label = int(row[self.label_col])
        path = os.path.join(self.image_dir, f"{img_id}{self.ext}")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, os.path.basename(path)