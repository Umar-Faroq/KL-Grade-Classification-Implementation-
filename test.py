"""
test.py — Evaluation script for KL-grade classification.

Loads a saved checkpoint, runs inference on the test split, and reports:
  • Overall accuracy
  • Per-class precision / recall / F1 (sklearn classification_report)
  • Macro & weighted averages
  • Confusion matrix saved as PNG

Usage:
    python test.py --model medmamba_t --checkpoint checkpoints/medmamba_t_best.pth
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KLGradeDataset, get_transforms
from train import build_model, NUM_CLASSES, CLASS_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  infer", file=sys.stdout):
        images = images.to(device)
        outputs = model(images)
        preds   = outputs.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


def save_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="KL-grade classification evaluator")
    p.add_argument("--model", required=True,
                   choices=["medmamba_t", "medmamba_s", "medmamba_b",
                            "resnet18", "vgg11", "efficientnet_b0",
                            "inception_v3", "densenet201"])
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--data_dir",   type=str, default="data/splits")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers",type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] device:     {device}")
    print(f"[test] model:      {args.model}")
    print(f"[test] checkpoint: {args.checkpoint}")

    img_size = 299 if args.model == "inception_v3" else 224

    # ---- Dataset ----
    test_csv = os.path.join(args.data_dir, "split_seed42_test.csv")
    test_ds  = KLGradeDataset(test_csv, transform=get_transforms("test", img_size))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"[test] test set size: {len(test_ds)}")

    # ---- Model ----
    model = build_model(args.model, num_classes=NUM_CLASSES)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    # Support both raw state_dict and wrapped checkpoint dicts
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    print(f"[test] Loaded checkpoint (epoch={ckpt.get('epoch', '?')},"
          f" val_acc={ckpt.get('val_acc', '?'):.4f})" if "epoch" in ckpt else
          f"[test] Loaded raw state_dict checkpoint")

    # ---- Inference ----
    y_true, y_pred = run_inference(model, test_loader, device)

    # ---- Metrics ----
    acc = (y_true == y_pred).mean()
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
    cm = confusion_matrix(y_true, y_pred)

    result_lines = [
        f"Model: {args.model}",
        f"Checkpoint: {args.checkpoint}",
        f"Test samples: {len(test_ds)}",
        f"\nOverall Accuracy: {acc:.4f}",
        f"\nClassification Report:\n{report}",
        f"\nConfusion Matrix (rows=true, cols=pred):\n{cm}",
    ]
    result_text = "\n".join(result_lines)

    print("\n" + result_text)

    # ---- Save results ----
    os.makedirs("results", exist_ok=True)
    txt_path = os.path.join("results", f"{args.model}_test_results.txt")
    with open(txt_path, "w") as f:
        f.write(result_text + "\n")
    print(f"\n[test] Results saved → {txt_path}")

    cm_path = os.path.join("results", f"{args.model}_confusion_matrix.png")
    save_confusion_matrix(cm, CLASS_NAMES, cm_path)


if __name__ == "__main__":
    main()
