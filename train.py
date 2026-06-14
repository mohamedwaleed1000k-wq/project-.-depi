
import os
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score, f1_score

import config
import data_load as dl
import data_pipeline as dp
from model import build_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Trainer")


def compute_metrics(logits: np.ndarray, targets: np.ndarray, threshold: float = 0.5):
    """
    logits  : (N, C) raw model output  (before sigmoid)
    targets : (N, C) binary labels
    Returns dict with auroc_macro, f1_macro
    """
    probs = 1 / (1 + np.exp(-logits))   # sigmoid

    # AUROC — skip classes with only one label value present
    try:
        auroc = roc_auc_score(targets, probs, average="macro")
    except ValueError:
        auroc = float("nan")

    preds = (probs >= threshold).astype(int)
    f1    = f1_score(targets, preds, average="macro", zero_division=0)

    return {"auroc": auroc, "f1": f1}



def run_epoch(model, loader, criterion, optimizer, scheduler, device, train: bool):
    model.train() if train else model.eval()

    total_loss   = 0.0
    all_logits   = []
    all_targets  = []

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)

            logits = model(X)
            loss   = criterion(logits, Y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss  += loss.item() * X.size(0)
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(Y.detach().cpu().numpy())

    all_logits  = np.concatenate(all_logits,  axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_loss    = total_loss / len(loader.dataset)
    metrics     = compute_metrics(all_logits, all_targets)

    return avg_loss, metrics


def train(
    n_epochs:        int   = 50,
    lr:              float = 1e-3,
    weight_decay:    float = 1e-4,
    patience:        int   = 10,
    checkpoint_dir:  str   = "checkpoints",
    model_variant:   str   = "resnet18",
):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    
    log.info("Loading metadata...")
    meta     = dl.load_metadata(config.CFG["data_dir"])
    scp_diag = dl.load_scp_statements(config.CFG["data_dir"])

    Y, mlb, class_names, meta_labeled = dp.build_label_matrix(
        meta, scp_diag, threshold=config.CFG["label_thresh"]
    )
    n_classes = len(class_names)
    log.info(f"Classes ({n_classes}): {class_names}")

    log.info("Creating DataLoaders...")
    train_dl, val_dl, _ = dl.create_dataloaders(
        meta_labeled, Y, config.CFG["data_dir"]
    )

    
    model = build_model(n_classes=n_classes, variant=model_variant).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: ResNet1D-{model_variant} | Trainable params: {total_params:,}")

    raw_weights = dp.compute_class_weights(Y, class_names)
    pos_weight  = torch.tensor(
        [raw_weights[c] for c in class_names], dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_dl),
        epochs=n_epochs,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_ckpt = os.path.join(checkpoint_dir, "best_model.pt")

    history = {
        "train_loss": [], "val_loss":   [],
        "train_auroc":[], "val_auroc":  [],
        "train_f1":   [], "val_f1":     [],
    }

    best_val_auroc  = -1.0
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = run_epoch(
            model, train_dl, criterion, optimizer, scheduler, device, train=True
        )
        val_loss, val_metrics = run_epoch(
            model, val_dl, criterion, None, None, device, train=False
        )

        elapsed = time.time() - t0

        # Store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auroc"].append(train_metrics["auroc"])
        history["val_auroc"].append(val_metrics["auroc"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        log.info(
            f"Epoch {epoch:03d}/{n_epochs} | "
            f"Train loss={train_loss:.4f} auroc={train_metrics['auroc']:.4f} f1={train_metrics['f1']:.4f} | "
            f"Val   loss={val_loss:.4f} auroc={val_metrics['auroc']:.4f} f1={val_metrics['f1']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc   = val_metrics["auroc"]
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_auroc":   best_val_auroc,
                "class_names": class_names,
                "config":      config.CFG,
            }, best_ckpt)
            log.info(f"  ✅ New best val AUROC={best_val_auroc:.4f} — checkpoint saved.")
        else:
            patience_counter += 1
            log.info(f"  No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                log.info(f"Early stopping at epoch {epoch}.")
                break

    log.info(f"Training finished. Best val AUROC: {best_val_auroc:.4f}")
    log.info(f"Best checkpoint: {best_ckpt}")

    return history, class_names


if __name__ == "__main__":
    history, class_names = train(
        n_epochs      = 50,
        lr            = 1e-3,
        weight_decay  = 1e-4,
        patience      = 10,
        checkpoint_dir= "checkpoints",
        model_variant = "resnet18",
    )