import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
from dataset import ODIRDataset
from model import ODIRModel

# ── Config ─────────────────────────────────────────
DATA_CSV   = "../data/odir/full_df.csv"
IMAGE_DIR  = "../data/odir/ODIR-5K"
OUTPUT_DIR = "../outputs"
EPOCHS     = 10
BATCH_SIZE = 16
LR         = 3e-4
IMAGE_SIZE = 224
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_COLS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
DISEASE_NAMES = ['Normal','Diabetes','Glaucoma','Cataract',
                 'AMD','Hypertension','Myopia','Other']

os.makedirs(OUTPUT_DIR, exist_ok=True)


def train():
    print(f"Using device: {DEVICE}")

    # ── Load & split data ───────────────────────────
    df = pd.read_csv(DATA_CSV)
    print(f"Total samples: {len(df)}")

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, shuffle=True
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = ODIRDataset(train_df, IMAGE_DIR, phase="train",
                           image_size=IMAGE_SIZE)
    val_ds   = ODIRDataset(val_df,   IMAGE_DIR, phase="val",
                           image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── Model ───────────────────────────────────────
    model     = ODIRModel(num_classes=8, pretrained=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()  # multi-label loss

    best_auc = 0.0

    for epoch in range(EPOCHS):
        # ── Train ───────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx}/"
                      f"{len(train_loader)} | Loss: {loss.item():.4f}")

        # ── Validate ─────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs  = model(images)
                val_loss += criterion(outputs, labels).item()
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        all_preds  = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # AUC per disease
        aucs = []
        for i, name in enumerate(DISEASE_NAMES):
            try:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aucs.append(auc)
            except Exception:
                aucs.append(0.0)

        mean_auc   = np.mean(aucs)
        avg_train  = train_loss / len(train_loader)
        avg_val    = val_loss   / len(val_loader)

        print(f"\nEpoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train:.4f} | "
              f"Val Loss: {avg_val:.4f} | "
              f"Mean AUC: {mean_auc:.4f}")
        for name, auc in zip(DISEASE_NAMES, aucs):
            print(f"  {name:<14}: AUC = {auc:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(),
                       f"{OUTPUT_DIR}/best_model.pth")
            print(f"  ✓ Saved best model (AUC={best_auc:.4f})\n")

        scheduler.step()

    print(f"\nTraining complete! Best Mean AUC: {best_auc:.4f}")
    
    

if __name__ == "__main__":
    train()