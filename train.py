"""
train.py — Training script for KL-grade classification.

Supports MedMamba variants and torchvision baselines.

Usage:
    python train.py --model medmamba_t
    python train.py --model resnet18 --epochs 50 --batch_size 64
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm

from dataset import KLGradeDataset, get_transforms

NUM_CLASSES = 5
CLASS_NAMES = ["KL-0", "KL-1", "KL-2", "KL-3", "KL-4"]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str, num_classes: int = NUM_CLASSES):
    """Instantiate and return a model ready for training."""
    name = name.lower()

    # ---- MedMamba variants ----
    if name in ("medmamba_t", "medmamba_s", "medmamba_b"):
        from MedMamba import VSSM
        configs = {
            "medmamba_t": dict(depths=[2, 2, 4, 2],  dims=[96, 192, 384, 768]),
            "medmamba_s": dict(depths=[2, 2, 8, 2],  dims=[96, 192, 384, 768]),
            "medmamba_b": dict(depths=[2, 2, 12, 2], dims=[128, 256, 512, 1024]),
        }
        cfg = configs[name]
        return VSSM(depths=cfg["depths"], dims=cfg["dims"], num_classes=num_classes)

    # ---- torchvision baselines ----
    if name == "resnet18":
        m = models.resnet18(pretrained=True)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "vgg11":
        m = models.vgg11(pretrained=True)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    if name == "efficientnet_b0":
        m = models.efficientnet_b0(pretrained=True)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    if name == "inception_v3":
        m = models.inception_v3(pretrained=True, aux_logits=True)
        m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, num_classes)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "densenet201":
        m = models.densenet201(pretrained=True)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m

    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, is_inception=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  train", leave=False, file=sys.stdout):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if is_inception:
            outputs, aux = model(images)
            loss = criterion(outputs, labels) + 0.4 * criterion(aux, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  val  ", leave=False, file=sys.stdout):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# TensorBoard helpers
# ---------------------------------------------------------------------------

def log_sample_images(writer, loader, device, tag="sample_images"):
    """Log a batch of images to TensorBoard (called once, first epoch)."""
    import torchvision
    images, labels = next(iter(loader))
    grid = torchvision.utils.make_grid(images[:16], normalize=True, scale_each=True)
    writer.add_image(tag, grid)


@torch.no_grad()
def log_pr_curves(writer, model, loader, device, num_classes, global_step):
    """Log per-class PR curves to TensorBoard."""
    import torch.nn.functional as F
    model.eval()
    all_probs = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)

    all_probs  = torch.cat(all_probs, dim=0)   # (N, C)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)

    for c in range(num_classes):
        gt    = (all_labels == c).int()
        probs = all_probs[:, c]
        writer.add_pr_curve(f"PR/{CLASS_NAMES[c]}", gt, probs, global_step=global_step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="KL-grade classification trainer")
    p.add_argument("--model", required=True,
                   choices=["medmamba_t", "medmamba_s", "medmamba_b",
                            "resnet18", "vgg11", "efficientnet_b0",
                            "inception_v3", "densenet201"])
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--data_dir",    type=str,   default="data/splits")
    p.add_argument("--run_name",    type=str,   default="",
                   help="Optional tag appended to TensorBoard run directory")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")
    print(f"[train] model:  {args.model}")

    # Inception_v3 requires 299×299
    img_size = 299 if args.model == "inception_v3" else 224
    is_inception = args.model == "inception_v3"

    # ---- Datasets ----
    train_csv = os.path.join(args.data_dir, "split_seed42_train.csv")
    val_csv   = os.path.join(args.data_dir, "split_seed42_val.csv")

    train_ds = KLGradeDataset(train_csv, transform=get_transforms("train", img_size))
    val_ds   = KLGradeDataset(val_csv,   transform=get_transforms("val",   img_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    # ---- Model ----
    model = build_model(args.model, num_classes=NUM_CLASSES)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- TensorBoard ----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_tag   = f"_{args.run_name}" if args.run_name else ""
    tb_dir    = os.path.join("runs", f"{args.model}{run_tag}_{timestamp}")
    writer    = SummaryWriter(log_dir=tb_dir)
    print(f"[train] TensorBoard dir: {tb_dir}")

    # ---- Checkpoints dir ----
    os.makedirs("checkpoints", exist_ok=True)
    best_ckpt = os.path.join("checkpoints", f"{args.model}_best.pth")
    last_ckpt = os.path.join("checkpoints", f"{args.model}_last.pth")

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Log sample images once
        if epoch == 1:
            log_sample_images(writer, train_loader, device)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, is_inception
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        # Console
        print(
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
            f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            f"  lr={current_lr:.6f}"
        )

        # TensorBoard scalars
        writer.add_scalar("Loss/train",     train_loss, epoch)
        writer.add_scalar("Loss/val",       val_loss,   epoch)
        writer.add_scalar("Accuracy/train", train_acc,  epoch)
        writer.add_scalar("Accuracy/val",   val_acc,    epoch)
        writer.add_scalar("LR/epoch",       current_lr, epoch)

        # Checkpoints
        ckpt = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc":              val_acc,
            "val_loss":             val_loss,
        }
        torch.save(ckpt, last_ckpt)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, best_ckpt)
            print(f"  [*] best val_acc improved → {best_val_acc:.4f}  (saved {best_ckpt})")

    # PR curves on val set at end of training
    log_pr_curves(writer, model, val_loader, device, NUM_CLASSES, global_step=args.epochs)

    writer.close()
    print(f"\n[train] Finished. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
