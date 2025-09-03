import os
import argparse
import yaml
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import classification_report

from dataset import FolderDataset, CSVDataset
from transforms import get_transforms
from model import build_model
from utils import set_seed, compute_metrics, ensure_dir

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/default.yaml')
    return p.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def make_dataloaders(cfg):
    input_size = cfg['data']['input_size']
    train_tfms, val_tfms = get_transforms(input_size=input_size)

    if cfg['data']['mode'] == 'folders':
        train_ds = FolderDataset(cfg['data']['train_dir'], transform=train_tfms)
        val_ds   = FolderDataset(cfg['data']['val_dir'], transform=val_tfms)
    else:
        csv = cfg['data']['csv']
        train_ds = CSVDataset(csv['train_csv'], image_dir=csv['image_dir'],
                              image_col=csv['image_col'], label_col=csv['label_col'],
                              transform=train_tfms)
        val_ds   = CSVDataset(csv['val_csv'], image_dir=csv['image_dir'],
                              image_col=csv['image_col'], label_col=csv['label_col'],
                              transform=val_tfms)

    # Compute class weights
    if cfg['training']['use_class_weights']:
        labels = [y for _, y, _ in train_ds]
        num_pos = sum(1 for y in labels if y == 1)
        num_neg = sum(1 for y in labels if y == 0)
        total = len(labels)
        weight_for_0 = total / (2.0 * max(num_neg, 1))
        weight_for_1 = total / (2.0 * max(num_pos, 1))
        class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32)
    else:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)

    # Optionally use a WeightedRandomSampler
    sample_weights = [class_weights[y].item() for _, y, _ in train_ds]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'],
                              sampler=sampler, num_workers=cfg['data']['num_workers'])
    val_loader   = DataLoader(val_ds, batch_size=cfg['training']['batch_size'],
                              shuffle=False, num_workers=cfg['data']['num_workers'])
    return train_loader, val_loader, class_weights

def get_optimizer(params, cfg):
    if cfg['training']['optimizer'] == 'adam':
        return torch.optim.Adam(params, lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    else:
        return torch.optim.SGD(params, lr=cfg['training']['lr'], momentum=0.9, weight_decay=cfg['training']['weight_decay'])

def get_scheduler(optimizer, cfg):
    if cfg['training']['scheduler'] == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif cfg['training']['scheduler'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['max_epochs'])
    else:
        return None

def train_one_epoch(model, loader, criterion, optimizer, device, log_every=50):
    model.train()
    all_probs, all_labels = [], []
    running_loss = 0.0
    for i, (x, y, _) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:,1]
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

        if (i+1) % log_every == 0:
            print(f"[train] step {i+1}/{len(loader)} loss={running_loss/(i+1):.4f}")
    return running_loss/ max(1,len(loader)), all_probs, all_labels

def validate(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels = [], []
    val_loss = 0.0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:,1]
            loss = criterion(logits, y)
            val_loss += loss.item()
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
    return val_loss/ max(1,len(loader)), all_probs, all_labels

def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, class_weights = make_dataloaders(cfg)
    model = build_model(backbone=cfg['model']['backbone'],
                        num_classes=cfg['model']['num_classes'],
                        pretrained=cfg['model']['pretrained'],
                        dropout=cfg['model']['dropout']).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = get_optimizer(model.parameters(), cfg)
    scheduler = get_scheduler(optimizer, cfg)

    best_metric = -1.0
    best_path = os.path.join(cfg['training']['checkpoint_dir'], "best_model.pt")
    ensure_dir(cfg['training']['checkpoint_dir'])

    patience = cfg['training']['early_stopping_patience']
    patience_counter = 0
    best_name = cfg['logging']['save_best_metric']

    for epoch in range(1, cfg['training']['max_epochs']+1):
        print(f"\nEpoch {epoch}/{cfg['training']['max_epochs']}")
        tr_loss, tr_probs, tr_labels = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                                       log_every=cfg['logging']['log_every_n_steps'])
        tr_metrics = compute_metrics(tr_labels, tr_probs)

        val_loss, val_probs, val_labels = validate(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_labels, val_probs)

        print(f"[train] loss={tr_loss:.4f} acc={tr_metrics['accuracy']:.4f} f1={tr_metrics['f1']:.4f} auc={tr_metrics['auc']:.4f}")
        print(f"[val]   loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f} auc={val_metrics['auc']:.4f}")

        # Track best by chosen metric
        current = val_metrics.get(best_name, float('nan'))
        if current > best_metric:
            best_metric = current
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg
            }, best_path)
            print(f"Saved new best model to {best_path} (best {best_name}={best_metric:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if scheduler is not None:
            scheduler.step()

    # Print classification report on val as a summary
    y_pred = (torch.tensor(val_probs) >= 0.5).int().tolist()
    print("\nValidation Classification Report:")
    print(classification_report(val_labels, y_pred, digits=4))

if __name__ == "__main__":
    main()