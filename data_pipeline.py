from scipy.signal import butter, sosfiltfilt, resample_poly
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
import logging

log = logging.getLogger("ECG-PTBXL")

def clean_single_signal(sig: np.ndarray) -> np.ndarray:

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

    sig = np.clip(sig, -5.0, 5.0)

    return sig


def bandpass_filter_single(sig: np.ndarray,
                           fs: int = 500, 
                           low: float = 0.5, high: float = 40.0, 
                           order: int = 4) -> np.ndarray:
    
    nyq = fs / 2.0  #(Nyquist) تردد نصف العينة ، كل تردد بنقطتين
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos") #(Butterworth) soft filter coefficients
    #Second-Order Sections 
    sig_f = np.empty_like(sig)
    
    C = sig.shape[1] # عدد المسارات (12)
    # تطبيق الفلتر على كل سلك لوحده
    for c in range(C):
        sig_f[:, c] = sosfiltfilt(sos, sig[:, c])  #(Forward and Backward)zero-phase shift
        
    return sig_f



def resample_single_signal(sig: np.ndarray,
                           orig_fs: int,
                           target_fs: int) -> np.ndarray:

    if orig_fs == target_fs:
        return sig
    
    sig_r = resample_poly(sig, target_fs, orig_fs, axis=0).astype(np.float32) #completing the axis points to 500

    return sig_r




def normalize_single_signal(sig: np.ndarray, method: str = "zscore") -> np.ndarray:
    
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




def aggregate_diagnostic(scp_codes: dict, scp_diag: pd.DataFrame, threshold: float = 0.0) -> list:

    classes = []
    for code, confidence in scp_codes.items():
        if code in scp_diag.index and confidence > threshold:
            cls = scp_diag.loc[code, "diagnostic_class"]
            if pd.notna(cls):
                classes.append(cls)
    return list(set(classes))




def build_label_matrix(meta: pd.DataFrame, scp_diag: pd.DataFrame, threshold: float = 0.0):
    
    meta = meta.copy()
    meta["diagnostic_superclass"] = meta["scp_codes"].apply(
        lambda d: aggregate_diagnostic(d, scp_diag, threshold)
    )

    # Drop records with no diagnostic label
    before = len(meta)
    meta = meta[meta["diagnostic_superclass"].map(len) > 0].copy()
    log.info(f"Records with >=1 diagnostic label: {len(meta)} ({before - len(meta)} dropped).")

    # Multi-label binarization
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(meta["diagnostic_superclass"]).astype(np.float32)

    log.info(f"Label matrix Y shape: {Y.shape} | Classes: {list(mlb.classes_)}")

    # Class distribution
    counts = Y.sum(axis=0)
    for name, cnt in zip(mlb.classes_, counts):
        log.info(f"Class {name:<6}: {int(cnt):>6} ({100*cnt/len(meta):.1f}%)")

    return Y, mlb, list(mlb.classes_), meta




def compute_class_weights(Y: np.ndarray, class_names: list) -> dict:

    N = Y.shape[0]
    weights = {}
    for i, name in enumerate(class_names):
        n_pos = Y[:, i].sum()
        n_neg = N - n_pos
        weights[name] = n_neg / (n_pos + 1e-8)
        
    log.info("Class positive weights (for Loss Function):")
    for name, w in weights.items():
        log.info(f"Weight {name:<6}: {w:.3f}")
        
    return weights