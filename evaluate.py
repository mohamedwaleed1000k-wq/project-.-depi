
import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
    multilabel_confusion_matrix,
    classification_report,
)

import config
import data_load as dl
import data_pipeline as dp
from model import build_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Evaluator")


def get_predictions(model, loader, device):
    """Run inference and collect logits + ground-truth labels."""
    model.eval()
    all_logits  = []
    all_targets = []

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            logits = model(X)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(Y.numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)



def find_best_threshold(logits_val, targets_val,
                        thresholds=np.arange(0.1, 0.9, 0.05)):
    """Grid-search a single global threshold on the validation set."""
    probs = 1 / (1 + np.exp(-logits_val))
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1    = f1_score(targets_val, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    log.info(f"Best threshold (val): {best_t:.2f}  macro-F1={best_f1:.4f}")
    return best_t


def evaluate(checkpoint_path: str = "checkpoints/best_model.pt",
             threshold: float = None,
             save_plots: bool = True,
             plot_dir: str    = "eval_plots"):

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ───────────────────────────────────────────────────
    log.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt        = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt["class_names"]
    n_classes   = len(class_names)

    model = build_model(n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    log.info(f"Loaded model from epoch {ckpt['epoch']}  |  val AUROC={ckpt['val_auroc']:.4f}")

    meta     = dl.load_metadata(config.CFG["data_dir"])
    scp_diag = dl.load_scp_statements(config.CFG["data_dir"])
    Y, mlb, _, meta_labeled = dp.build_label_matrix(
        meta, scp_diag, threshold=config.CFG["label_thresh"]
    )

    _, val_dl, test_dl = dl.create_dataloaders(
        meta_labeled, Y, config.CFG["data_dir"]
    )
    
    log.info("Running inference on validation set (for threshold search)...")
    logits_val, targets_val = get_predictions(model, val_dl, device)

    log.info("Running inference on test set...")
    logits_test, targets_test = get_predictions(model, test_dl, device)

    probs_test = 1 / (1 + np.exp(-logits_test))

    if threshold is None:
        threshold = find_best_threshold(logits_val, targets_val)

    preds_test = (probs_test >= threshold).astype(int)

    log.info("=" * 60)
    log.info("PER-CLASS METRICS (test set)")
    log.info("=" * 60)

    results = {}
    for i, cls in enumerate(class_names):
        auroc = roc_auc_score(targets_test[:, i], probs_test[:, i])
        ap    = average_precision_score(targets_test[:, i], probs_test[:, i])
        f1    = f1_score(targets_test[:, i], preds_test[:, i], zero_division=0)
        prec  = precision_score(targets_test[:, i], preds_test[:, i], zero_division=0)
        rec   = recall_score(targets_test[:, i], preds_test[:, i], zero_division=0)
        results[cls] = {"AUROC": auroc, "AP": ap, "F1": f1, "Precision": prec, "Recall": rec}
        log.info(f"  {cls:<6}  AUROC={auroc:.4f}  AP={ap:.4f}  F1={f1:.4f}  "
                 f"Prec={prec:.4f}  Rec={rec:.4f}")

    macro_auroc = roc_auc_score(targets_test, probs_test, average="macro")
    macro_ap    = average_precision_score(targets_test, probs_test, average="macro")
    macro_f1    = f1_score(targets_test, preds_test, average="macro", zero_division=0)

    log.info("-" * 60)
    log.info(f"  MACRO  AUROC={macro_auroc:.4f}  AP={macro_ap:.4f}  F1={macro_f1:.4f}")
    log.info("=" * 60)

    print("\n" + classification_report(targets_test, preds_test,
                                       target_names=class_names, zero_division=0))

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        _plot_roc_curves(targets_test, probs_test, class_names, plot_dir)
        _plot_pr_curves(targets_test, probs_test, class_names, plot_dir)
        _plot_confusion_matrices(targets_test, preds_test, class_names, plot_dir)
        log.info(f"Plots saved to {plot_dir}/")

    return results, macro_auroc, macro_f1



COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]


def _plot_roc_curves(targets, probs, class_names, plot_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, color) in enumerate(zip(class_names, COLORS)):
        fpr, tpr, _ = roc_curve(targets[:, i], probs[:, i])
        auc = roc_auc_score(targets[:, i], probs[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "roc_curves.png"), dpi=150)
    plt.close()


def _plot_pr_curves(targets, probs, class_names, plot_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, color) in enumerate(zip(class_names, COLORS)):
        prec, rec, _ = precision_recall_curve(targets[:, i], probs[:, i])
        ap = average_precision_score(targets[:, i], probs[:, i])
        ax.plot(rec, prec, color=color, lw=2, label=f"{cls} (AP={ap:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "pr_curves.png"), dpi=150)
    plt.close()


def _plot_confusion_matrices(targets, preds, class_names, plot_dir):
    cms = multilabel_confusion_matrix(targets, preds)   # (n_classes, 2, 2)
    n   = len(class_names)
    fig = plt.figure(figsize=(4 * n, 4))
    gs  = gridspec.GridSpec(1, n, figure=fig)

    for i, (cls, color) in enumerate(zip(class_names, COLORS)):
        ax  = fig.add_subplot(gs[i])
        cm  = cms[i]                   # [[TN, FP], [FN, TP]]
        im  = ax.imshow(cm, cmap="Blues")
        ax.set_title(cls, fontsize=13, fontweight="bold", color=color)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]),
                        ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if cm[r, c] > cm.max() / 2 else "black")

    fig.suptitle("Confusion Matrices (per class) — Test Set",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "confusion_matrices.png"),
                dpi=150, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    evaluate(
        checkpoint_path = "checkpoints/best_model.pt",
        threshold       = None,      # auto-search on val set
        save_plots      = True,
        plot_dir        = "eval_plots",
    )