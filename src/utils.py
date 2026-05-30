import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
import torch
import seaborn as sns

DISEASE_NAMES = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract',
                 'AMD', 'Hypertension', 'Myopia', 'Other']


# ── 1. Plot AUC curves for all diseases ─────────────────────────
def plot_roc_curves(all_labels, all_preds, save_path="../outputs/roc_curves.png"):
    """Plot ROC curve for each disease and save to file."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("ROC Curves — Retinal Disease Detection", fontsize=15, fontweight='bold')

    for i, (name, ax) in enumerate(zip(DISEASE_NAMES, axes.flat)):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_preds[:, i])
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        ax.plot(fpr, tpr, color='navy', lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"ROC curves saved to {save_path}")


# ── 2. Plot training loss curve ──────────────────────────────────
def plot_loss_curve(train_losses, val_losses, save_path="../outputs/loss_curve.png"):
    """Plot train vs validation loss over epochs."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses,   'r-o', label='Val Loss',   linewidth=2)
    plt.title("Training vs Validation Loss", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Loss curve saved to {save_path}")


# ── 3. Plot AUC bar chart ────────────────────────────────────────
def plot_auc_bars(aucs, save_path="../outputs/auc_bars.png"):
    """Bar chart of per-disease AUC scores."""
    colors = ['#2ecc71' if a >= 0.85 else '#f39c12' if a >= 0.75 else '#e74c3c' for a in aucs]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(DISEASE_NAMES, aucs, color=colors, edgecolor='white', linewidth=0.5)
    plt.axhline(y=0.85, color='green',  linestyle='--', alpha=0.6, label='0.85 threshold')
    plt.axhline(y=0.75, color='orange', linestyle='--', alpha=0.6, label='0.75 threshold')
    plt.ylim([0.5, 1.0])
    plt.title("Per-Disease AUC-ROC Scores", fontsize=13, fontweight='bold')
    plt.ylabel("AUC Score")
    plt.xticks(rotation=15)
    for bar, auc in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{auc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    patches = [
        mpatches.Patch(color='#2ecc71', label='AUC ≥ 0.85 (Excellent)'),
        mpatches.Patch(color='#f39c12', label='AUC ≥ 0.75 (Good)'),
        mpatches.Patch(color='#e74c3c', label='AUC < 0.75 (Needs work)'),
    ]
    plt.legend(handles=patches, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"AUC bar chart saved to {save_path}")


# ── 4. Confusion matrix for one disease ─────────────────────────
def plot_confusion_matrix(all_labels, all_preds, disease_idx=2,
                          threshold=0.5, save_path="../outputs/confusion_matrix.png"):
    """Plot confusion matrix for a single disease at given threshold."""
    name = DISEASE_NAMES[disease_idx]
    y_true = all_labels[:, disease_idx]
    y_pred = (all_preds[:, disease_idx] >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'])
    plt.title(f"Confusion Matrix — {name}", fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {save_path}")


# ── 5. Save training log to CSV ──────────────────────────────────
def save_training_log(epoch, train_loss, val_loss, aucs,
                      log_path="../outputs/training_log.csv"):
    """Append one epoch's results to a CSV log file."""
    row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
           "mean_auc": np.mean(aucs)}
    for name, auc in zip(DISEASE_NAMES, aucs):
        row[name] = auc

    df_new = pd.DataFrame([row])
    if os.path.exists(log_path):
        df_existing = pd.read_csv(log_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(log_path, index=False)


# ── 6. Print full classification report ─────────────────────────
def print_classification_report(all_labels, all_preds, threshold=0.5):
    """Print precision, recall, F1 for each disease."""
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (threshold = 0.5)")
    print("="*60)
    for i, name in enumerate(DISEASE_NAMES):
        y_true = all_labels[:, i]
        y_pred = (all_preds[:, i] >= threshold).astype(int)
        report = classification_report(y_true, y_pred,
                                       target_names=['No', 'Yes'],
                                       output_dict=True)
        print(f"\n{name}:")
        print(f"  Precision: {report['Yes']['precision']:.4f}")
        print(f"  Recall:    {report['Yes']['recall']:.4f}")
        print(f"  F1-Score:  {report['Yes']['f1-score']:.4f}")


# ── 7. Generate all charts at once ──────────────────────────────
def generate_all_charts(all_labels, all_preds, train_losses, val_losses):
    """Run all visualizations in one call."""
    aucs = [roc_auc_score(all_labels[:, i], all_preds[:, i])
            for i in range(len(DISEASE_NAMES))]

    print("\nGenerating all charts...")
    plot_roc_curves(all_labels, all_preds)
    plot_auc_bars(aucs)
    plot_loss_curve(train_losses, val_losses)
    plot_confusion_matrix(all_labels, all_preds, disease_idx=2)  # Glaucoma
    print_classification_report(all_labels, all_preds)
    print("\nAll charts saved to outputs/ folder!")