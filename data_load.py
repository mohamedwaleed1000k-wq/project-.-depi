import os, ast, logging
import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset, DataLoader

import config
import data_pipeline as dp

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



def filter_corrupted_records(meta: pd.DataFrame, data_dir: str, fs: int = 500) -> pd.DataFrame:

    log.info("Starting Quality Control Sweep... This might take a few minutes.")
    valid_indices = []
    path_col = "filename_hr" if fs == 500 else "filename_lr"

    for ecg_id, row in meta.iterrows():
        file_path = os.path.join(data_dir, row[path_col])
        try:
        
            sig, _ = wfdb.rdsamp(file_path)
            sig = dp.clean_single_signal(sig)
            
    
            if sig is not None:
                valid_indices.append(ecg_id)
        except Exception as e:
            pass  
            
    dropped_count = len(meta) - len(valid_indices)
    log.info(f"QC Sweep Finished! Dropped {dropped_count} corrupted records.")
    
    return meta.loc[valid_indices].copy()
    

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





#feature engineering functions

class PTBXLDataset(Dataset):
    def __init__(self, meta_df: pd.DataFrame,
                       labels: np.ndarray, 
                       data_dir: str, 
                       fs: int = 500):
        self.meta = meta_df
        self.labels = labels
        self.data_dir = data_dir
        self.fs = fs
        self.path_col = "filename_hr" if fs == 500 else "filename_lr"

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.data_dir, row[self.path_col])
        
        sig, _ = wfdb.rdsamp(file_path)
        
        sig = dp.clean_single_signal(sig)
        
        if sig is None:
            raise ValueError(f"Corrupted signal detected at index {idx}! Please run QC Sweep first.")
            

        sig = dp.bandpass_filter_single(sig, fs=self.fs, 
                                             low=config.CFG["bandpass_low"], 
                                             high=config.CFG["bandpass_high"])
        sig = dp.normalize_single_signal(sig, method="zscore")   

        sig = np.transpose(sig, (1, 0))  #(عدد القراءات الزمنية، عدد الأسلاك) X
        
        return torch.tensor(sig, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)



def create_dataloaders(meta: pd.DataFrame, Y: np.ndarray, data_dir: str):
    
    folds = meta["strat_fold"].values

    train_idx = np.where((folds != config.CFG["val_fold"]) & (folds != config.CFG["test_fold"]))[0]
    val_idx   = np.where(folds == config.CFG["val_fold"])[0]
    test_idx  = np.where(folds == config.CFG["test_fold"])[0]

    log.info("Split (patient-wise via strat_fold):")
    log.info(f"  Train : {len(train_idx)} patients")
    log.info(f"  Val   : {len(val_idx)} patients")
    log.info(f"  Test  : {len(test_idx)} patients")

    train_ds = PTBXLDataset(meta.iloc[train_idx], Y[train_idx], data_dir, config.CFG["sampling_rate"])
    val_ds   = PTBXLDataset(meta.iloc[val_idx],   Y[val_idx],   data_dir, config.CFG["sampling_rate"])
    test_ds  = PTBXLDataset(meta.iloc[test_idx],  Y[test_idx],  data_dir, config.CFG["sampling_rate"])

    train_dl = DataLoader(train_ds, batch_size=config.CFG["batch_size"], shuffle=True,  num_workers=config.CFG["num_workers"])
    val_dl   = DataLoader(val_ds,   batch_size=config.CFG["batch_size"], shuffle=False, num_workers=config.CFG["num_workers"])
    test_dl  = DataLoader(test_ds,  batch_size=config.CFG["batch_size"], shuffle=False, num_workers=config.CFG["num_workers"])

    return train_dl, val_dl, test_dl