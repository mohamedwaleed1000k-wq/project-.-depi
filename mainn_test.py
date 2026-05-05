import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # ✅ لازم يكون أول سطر

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import wfdb
import load, config

# ── Step 1: Config ──
print("Checking Config...")
try:
    print(f"   - Data Directory  : {config.CFG['data_dir']}")
    print(f"   - Sampling Rate   : {config.CFG['sampling_rate']} Hz")
    print(f"   - Batch Size      : {config.CFG['batch_size']}")
    print("✅ Config is OK.")
except Exception as e:
    print(f"❌ Config Error: {e}")

# ── Step 2: Load Metadata ──
print("\nChecking Metadata Loading...")
try:
    df_meta = load.load_metadata(config.CFG['data_dir'])  # ✅ هنا بنعرّفها
    load.inspect_missing_metadata(df_meta)
    print(f"   - Dataset contains {len(df_meta)} records.")
    print("✅ Metadata Load is OK.")
except Exception as e:
    print(f"❌ Metadata Test Failed: {e}")

# ── Step 3: Pipeline ──
print("\nChecking Signal Pipeline...")
try:
    sample_row = df_meta.iloc[0]
    file_path  = sample_row['filename_hr']
    
    print(f"   - Testing with Patient ID: {sample_row.name}")

    full_path = os.path.join(config.CFG['data_dir'], sample_row['filename_hr'])
    signal, _ = wfdb.rdsamp(full_path)
    print(f"   - Raw Signal Shape     : {signal.shape}")

    clean_sig    = load.clean_single_signal(signal)
    print(f"   - After Cleaning       : {clean_sig.shape}")

    filtered_sig = load.bandpass_filter_single(
        clean_sig,
        fs   = config.CFG['sampling_rate'],
        low  = config.CFG['bandpass_low'],
        high = config.CFG['bandpass_high']
    )
    filtered_sig = np.clip(filtered_sig, -5.0, 5.0) 
    print(f"   - After Filtering      : {filtered_sig.shape}")

    final_sig = load.normalize_single_signal(filtered_sig)
    print(f"   - After Normalization  : mean = {final_sig.mean():.4f}")

    print("\n✅ PIPELINE STEPS ARE SUCCESSFUL!")

    plt.figure(figsize=(15, 5))
    plt.plot(final_sig[:1000, 0])
    plt.title(f"Cleaned & Filtered ECG - Patient {sample_row.name} (Lead I)")
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"❌ Pipeline Test Failed: {e}")


fig, axes = plt.subplots(2, 1, figsize=(15, 8))

axes[0].plot(signal[:1000, 0])
axes[0].set_title("Before Processing (Raw)")
axes[0].grid(True)

axes[1].plot(final_sig[:1000, 0])
axes[1].set_title("After Processing (Cleaned)")
axes[1].grid(True)

plt.tight_layout()
plt.show()


