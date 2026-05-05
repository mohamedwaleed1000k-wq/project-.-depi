import os, ast, config, logging
import numpy as np
import pandas as pd

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly


# LOGGING SETUP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Data-Analyst")

def load_metadata(data_dir: str) -> pd.DataFrame:

    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    log.info(f"Loading metadata from {csv_path}...")
    
    meta = pd.read_csv(csv_path, index_col="ecg_id")
    
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval) # Convert stringified dict to actual dict
    
    before = len(meta)
    meta = meta.drop(index=[i for i in config.DROPPED_ECG_IDS if i in meta.index])
    log.info(f"Dropped {before - len(meta)} duplicate records. {len(meta)} records remain.")

    #Fix privacy-masked age
    masked = (meta["age"] == 300).sum()
    meta["age"] = meta["age"].replace(300, np.nan)
    log.info(f"Masked ages (300 -> NaN): {masked} patients aged >89.")

    return meta


def load_scp_statements(data_dir: str) -> pd.DataFrame:
    scp_path = os.path.join(data_dir, "scp_statements.csv")
    scp = pd.read_csv(scp_path, index_col=0)
    
    # Filter only diagnostic statements
    scp_diag = scp[scp["diagnostic"] == 1].copy()
    
    log.info(f"SCP diagnostic codes loaded: {len(scp_diag)} codes "
             f"across {scp_diag['diagnostic_class'].nunique()} superclasses.")
    
    return scp_diag

def inspect_missing_metadata(meta: pd.DataFrame) -> None:
    missing = meta.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    print("\n[2] Metadata missing values:")
    if missing.empty:
        print("None.")
    else:
        for col, cnt in missing.items():
            pct = 100 * cnt / len(meta)
            print(f"{col:<25} {cnt:>5} ({pct:.1f}%)")


def clean_single_signal(sig: np.ndarray) -> np.ndarray:

    """
    Clean a single ECG signal to save RAM.
    Detect and handle:
      • NaN / Inf values      → interpolate per lead
      • Completely flat leads → return None (mark as corrupted)
      • Amplitude clipping    → clip to ±5 mV (physiological range)
    """
    T, C = sig.shape
    
    for c in range(C):
        lead = sig[:, c]
        bad  = ~np.isfinite(lead)
        if bad.any():
            idx  = np.arange(T)
            good = np.where(~bad)[0]
            if len(good) < 2:               
                return None # الإشارة كلها تالفة
            sig[:, c] = np.interp(idx, good, lead[good]) #تقريب خطي

    if (sig.std(axis=0) < 1e-6).any():
        return None


    return sig

def bandpass_filter_single(sig, fs=500, low=0.05, high=40.0, order=4):
    nyq = fs / 2.0
    sos = butter(order, [low/nyq, high/nyq], btype="band", output="sos")
    sig_f = np.empty_like(sig)
    for c in range(sig.shape[1]):
        sig_f[:, c] = sosfiltfilt(sos, sig[:, c])
    return sig_f

def resample_single_signal(sig: np.ndarray,
                           orig_fs: int,
                           target_fs: int) -> np.ndarray:

    if orig_fs == target_fs:
        return sig
    
    sig_r = resample_poly(sig, target_fs, orig_fs, axis=0).astype(np.float32) #completing the axis points to 500
    sig_r = np.clip(sig_r, -5.0, 5.0)

    return sig_r

def normalize_single_signal(sig: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalize ECG signal to a standard range or distribution.
    
    Args:
        sig: Input signal array (samples x leads)
        method: 'zscore' or 'minmax'
    
    Returns:
        Normalized signal as float32
    """
    sig_n = sig.copy()
    
    if method == "zscore":
        # حساب المتوسط والانحراف المعياري لكل سلك
        mu  = sig_n.mean(axis=0, keepdims=True)
        std = sig_n.std(axis=0, keepdims=True) + 1e-8  # no division by zero
        sig_n = (sig_n - mu) / std
        sig_n = np.clip(sig_n, -5.0, 5.0)  # remove outliers after zscore
        
    elif method == "minmax":
        mn  = sig_n.min(axis=0, keepdims=True)
        mx  = sig_n.max(axis=0, keepdims=True)
        sig_n = (sig_n - mn) / (mx - mn + 1e-8)  # result is always 0 → 1
        
    return sig_n.astype(np.float32)