


import wfdb
from scipy.signal import butter, sosfiltfilt, resample_poly
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")



"""
def load_signals(meta: pd.DataFrame, data_dir: str, fs: int = 100) -> np.ndarray:
# 
    Load all WFDB records.
    Returns X  shape: (N, T, 12)
        N  = number of records
        T  = 1000 at 100 Hz  |  5000 at 500 Hz
        12 = leads
#
    col = "filename_lr" if fs == 100 else "filename_hr"
    signals = []
    for idx, (_, row) in enumerate(meta.iterrows()):
        signal, _ = wfdb.rdsamp(os.path.join(data_dir, row[col]))
        signals.append(signal)
        if (idx + 1) % 2000 == 0:
            print(f"   Loaded {idx+1}/{len(meta)} …")
    X = np.array(signals, dtype=np.float32)   # (N, T, 12)
    print(f"[1] Signals loaded → shape {X.shape}  dtype {X.dtype}")
    return X
"""









# =============================================================================
# STEP 8 — MODEL ARCHITECTURE RECOMMENDATION  (reference only)
# =============================================================================

def recommended_model_stub(num_leads: int = 12, num_classes: int = 5) -> str:
    """
    Return a description of the recommended architecture.

    Best architecture for PTB-XL (based on published benchmarks):
    ► 1D-ResNet (e.g. ResNet-1D with depth 34 or 50)
      - Input : (batch, 12, 1000)
      - Conv1D stem → stacked residual blocks (kernel 15–17, dilated)
      - Global Average Pooling → FC → sigmoid (multi-label)
      - Loss: BCEWithLogitsLoss with pos_weight
      - Optimizer: AdamW + cosine LR schedule
      - Augmentation: random amplitude scaling, lead dropout, Gaussian noise

    Runner-up: Transformer / xresnet1d50  (marginal improvement, 3× slower)
    """
    arch = f"""
    =============================================================
    RECOMMENDED ARCHITECTURE: 1D-ResNet
    =============================================================
    Input  : (batch={CFG['batch_size']}, leads={num_leads}, time=1000)
    Stem   : Conv1d(12→64, kernel=15, stride=2) → BN → ReLU → MaxPool
    Blocks : 4 × ResBlock groups with skip connections
             [64→64 (×2), 64→128 (×2), 128→256 (×2), 256→512 (×2)]
    Head   : GlobalAvgPool1d → Linear(512 → {num_classes}) → Sigmoid
    Loss   : BCEWithLogitsLoss(pos_weight=class_weights_tensor)
    Optim  : AdamW(lr=1e-3, weight_decay=1e-4)
    Sched  : CosineAnnealingLR
    Metric : macro AUROC (official PTB-XL benchmark metric)
    =============================================================
    """
    return arch


# =============================================================================
# MASTER PIPELINE  — runs all steps in order
# =============================================================================

def run_pipeline(cfg: dict = CFG):
    print("\n" + "=" * 65)
    print("  PTB-XL ECG PREPROCESSING PIPELINE  (v1.0.3)")
    print("=" * 65 + "\n")

    # ── Step 1 : Load ────────────────────────────────────────────────────────
    print("── STEP 1 : LOADING ──────────────────────────────────────────")
    meta    = load_metadata(cfg["data_dir"])
    scp_diag= load_scp_statements(cfg["data_dir"])
    X_raw   = load_signals(meta, cfg["data_dir"], fs=cfg["sampling_rate"])

    # ── Step 2 : Clean ───────────────────────────────────────────────────────
    print("\n── STEP 2 : CLEANING ─────────────────────────────────────────")
    inspect_missing_metadata(meta)
    X_clean, meta_clean, _ = clean_signals(X_raw, meta)

    # ── Step 3 : Signal preprocessing ───────────────────────────────────────
    print("\n── STEP 3 : SIGNAL PREPROCESSING ────────────────────────────")
    X_filt  = bandpass_filter(X_clean,
                               fs   = cfg["sampling_rate"],
                               low  = cfg["bandpass_low"],
                               high = cfg["bandpass_high"],
                               order= cfg["bandpass_ord"])
    X_res   = resample_signals(X_filt,
                                orig_fs   = cfg["sampling_rate"],
                                target_fs = cfg["target_fs"])
    X_norm  = normalize_signals(X_res, method="zscore")

    # ── Step 4 : Labels ──────────────────────────────────────────────────────
    print("\n── STEP 4 : LABEL PROCESSING ────────────────────────────────")
    Y, mlb, class_names, meta_labeled = build_label_matrix(
        meta_clean, scp_diag, threshold=cfg["label_thresh"]
    )
    class_weights = compute_class_weights(Y, class_names)

    # Align X with (possibly smaller) labeled meta
    labeled_iloc = meta_clean.index.get_indexer(meta_labeled.index)
    X_norm       = X_norm[labeled_iloc]

    # ── Step 5 : Feature engineering (deep learning path) ────────────────────
    print("\n── STEP 5 : FEATURE ENGINEERING ─────────────────────────────")
    X_dl = prepare_for_deep_learning(X_norm)   # (N, 12, T) for Conv1d

    # Optional: classical features (uncomment if using traditional ML)
    # X_feat = extract_time_domain_features(X_norm, fs=cfg["target_fs"])

    # ── Step 6 : Split ───────────────────────────────────────────────────────
    print("\n── STEP 6 : TRAIN / VAL / TEST SPLIT ────────────────────────")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
        patient_wise_split(X_dl, Y, meta_labeled,
                           val_fold  = cfg["val_fold"],
                           test_fold = cfg["test_fold"])

    # ── Step 7 : PyTorch tensors & DataLoaders ───────────────────────────────
    print("\n── STEP 7 : PYTORCH TENSORS & DATALOADERS ───────────────────")
    (train_t, val_t, test_t) = to_tensors(
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    )
    train_loader, val_loader, test_loader = make_dataloaders(
        train_t, val_t, test_t,
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"]
    )
    print(f"[7] DataLoaders ready — "
          f"train {len(train_loader)} | val {len(val_loader)} | "
          f"test {len(test_loader)} batches.")

    # ── Step 8 : Model recommendation ────────────────────────────────────────
    print("\n── STEP 8 : MODEL RECOMMENDATION ───────────────────────────")
    print(recommended_model_stub(num_leads=12, num_classes=len(class_names)))

    # ── Class weights tensor (for loss function) ──────────────────────────────
    pos_weight_tensor = torch.tensor(
        [class_weights[c] for c in class_names], dtype=torch.float32
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("=" * 65)
    print("  PIPELINE COMPLETE — OUTPUT SUMMARY")
    print("=" * 65)
    print(f"  X_train : {train_t[0].shape}  y_train : {train_t[1].shape}")
    print(f"  X_val   : {val_t[0].shape}    y_val   : {val_t[1].shape}")
    print(f"  X_test  : {test_t[0].shape}   y_test  : {test_t[1].shape}")
    print(f"  Classes : {class_names}")
    print(f"  pos_weight_tensor : {pos_weight_tensor}")
    print("=" * 65)

    return {
        # ── NumPy arrays ──────────────────────────────────────────────────
        "X_train"    : X_train,   "y_train"    : y_train,
        "X_val"      : X_val,     "y_val"      : y_val,
        "X_test"     : X_test,    "y_test"     : y_test,
        # ── PyTorch tensors ───────────────────────────────────────────────
        "train_t"    : train_t,   "val_t"      : val_t,
        "test_t"     : test_t,
        # ── DataLoaders ───────────────────────────────────────────────────
        "train_loader": train_loader,
        "val_loader"  : val_loader,
        "test_loader" : test_loader,
        # ── Metadata ──────────────────────────────────────────────────────
        "mlb"              : mlb,
        "class_names"      : class_names,
        "class_weights"    : class_weights,
        "pos_weight_tensor": pos_weight_tensor,
        "meta"             : meta_labeled,
    }


# =============================================================================
if __name__ == "__main__":
    results = run_pipeline(CFG)
