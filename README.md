ECG Diagnostic Classification — PTB-XL
A deep-learning-ready data pipeline for multi-label ECG classification using the PTB-XL dataset(https://physionet.org/content/ptb-xl/1.0.3/), a large publicly available 12-lead electrocardiography dataset.

Overview
This project preprocesses and prepares the PTB-XL ECG dataset for training a multi-label classifier that maps raw 12-lead ECG signals to diagnostic superclasses (e.g. NORM, MI, STTC, CD, HYP). The pipeline covers metadata loading, quality control, signal cleaning, bandpass filtering, resampling, normalization, label engineering, and PyTorch DataLoader creation.

Project Structure
.
├── config.py             # Central configuration (paths, hyperparams, filter settings)
├── data_load.py          # Metadata loading, QC sweep, Dataset & DataLoader construction
├── data_pipeline.py      # Signal processing functions & label matrix builder
├── main_test_cop.ipynb   # End-to-end pipeline test (Steps 1–12)
└── EDA_visuals_cop.ipynb # Exploratory Data Analysis & visualizations

Dataset
PTB-XL — ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
Place the dataset folder at the path specified in config.py under data_dir. The pipeline expects the following files to be present inside that folder:

ptbxl_database.csv — patient metadata and SCP codes
scp_statements.csv — SCP code definitions and diagnostic classes
Signal files at filename_hr (500 Hz) or filename_lr (100 Hz) paths


Configuration (config.py)
KeyDefaultDescriptiondata_dir(your local path)Root directory of the PTB-XL datasetsampling_rate500Source sampling rate to load (100 or 500 Hz)target_fs500Resample target frequencybandpass_low0.5 HzHigh-pass cutoff — removes baseline wanderbandpass_high40.0 HzLow-pass cutoff — removes high-frequency noisebandpass_ord4Butterworth filter orderval_fold9Stratified fold used as validation settest_fold10Stratified fold used as test set (official split)label_thresh0.0Minimum SCP code confidence to include a labelbatch_size64DataLoader batch sizenum_workers0DataLoader worker processes
LEAD_NAMES lists the standard 12-lead names: I, II, III, aVR, aVL, aVF, V1–V6.
DROPPED_ECG_IDS is a curated list of ECG IDs corresponding to known duplicate records that are removed during metadata loading.

Pipeline (data_load.py + data_pipeline.py)
1. Metadata Loading
load_metadata(data_dir) reads ptbxl_database.csv, parses SCP code dictionaries, removes duplicate records listed in DROPPED_ECG_IDS, and replaces privacy-masked ages (coded as 300) with NaN.
2. SCP Statements
load_scp_statements(data_dir) reads scp_statements.csv and filters it to diagnostic codes only (diagnostic == 1), giving the mapping from SCP codes to diagnostic superclasses.
3. Quality Control Sweep
filter_corrupted_records(meta, data_dir, fs) iterates over all records, reads each signal, and runs clean_single_signal on it. Records that fail (corrupted leads, flat signals, read errors) are silently dropped. This step can take several minutes on the full dataset.
4. Signal Cleaning (clean_single_signal)
Per-lead processing on each raw signal array (T, 12):

NaN/Inf values are repaired via linear interpolation across valid samples.
Signals with fewer than 2 valid samples in any lead are discarded (return None).
Flat leads (std < 1e-6) are discarded.
Values are clipped to [-5.0, 5.0] mV.

5. Bandpass Filtering (bandpass_filter_single)
Applies a zero-phase 4th-order Butterworth bandpass filter (0.5–40 Hz by default) independently to each of the 12 leads using sosfiltfilt (forward-backward pass, no phase shift).
6. Resampling (resample_single_signal)
Resamples signals from any source frequency to the target frequency using scipy.signal.resample_poly. If source and target frequencies match, the signal is returned unchanged.
7. Normalization (normalize_single_signal)
Two methods available, applied per-lead:

zscore (default): subtracts per-lead mean, divides by per-lead std, clips to [-5.0, 5.0].
minmax: scales each lead to [0, 1].

8. Label Engineering (build_label_matrix)

aggregate_diagnostic maps each record's SCP codes to diagnostic superclasses using the filtered SCP table and a confidence threshold.
build_label_matrix adds the superclass list to metadata, drops unlabeled records, and runs MultiLabelBinarizer to produce a binary label matrix Y of shape (N, n_classes).

9. Class Weights (compute_class_weights)
Computes per-class positive weights as n_negative / n_positive for use as pos_weight in BCEWithLogitsLoss, addressing class imbalance during training.
10. DataLoaders (create_dataloaders)
Splits the dataset using strat_fold (patient-disjoint), constructs PTBXLDataset instances for train/val/test, and wraps them in PyTorch DataLoaders. Each __getitem__ loads a signal, runs cleaning → bandpass filtering → z-score normalization, and transposes to shape (12, T) for convolutional model input.
Output tensor shapes per batch:

X_batch: (batch_size, 12, 5000) — 12 leads × 5000 time steps @ 500 Hz
Y_batch: (batch_size, n_classes) — multi-label binary targets


Exploratory Data Analysis (EDA_visuals_cop.ipynb)
The EDA notebook produces the following visualizations:

Diagnostic Superclass Distribution — bar chart of record counts per class (imbalance check)
Multi-Label Co-occurrence Heatmap — row-normalized co-occurrence matrix across all class pairs
Patient Age Distribution — histogram with median marker + per-class boxplots
Sex Distribution by Class — grouped bar chart (Male vs Female) per diagnostic class
Stratified Fold Distribution — record counts per fold with train/val/test color-coding
Full 12-Lead ECG Visualization — raw vs filtered signal plotted for all 12 leads of a sample patient


Pipeline Testing (main_test_cop.ipynb)
A step-by-step validation notebook that tests every component of the pipeline in sequence:
StepTest1Metadata loading and shape check2Duplicate ECG ID removal verification3SCP statements loading and diagnostic filtering4Label matrix construction and class distribution5Missing metadata inspection6Signal cleaning on a sample record7Bandpass filter application + visual validation8Resampling (100 Hz → 500 Hz) + no-op (500 → 500)9Z-score normalization (mean ≈ 0, std ≈ 1) + min-max (values in [0,1])10QC sweep on a 50-record sample11Class weight computation + tensor conversion for loss function12DataLoader batch shape and dtype assertions
All 12 steps must pass before proceeding to model training.

Dependencies
numpy
pandas
scipy
scikit-learn
torch
wfdb
matplotlib
seaborn
Install via:
bashpip install numpy pandas scipy scikit-learn torch wfdb matplotlib seaborn

Usage
pythonimport config
import data_load as dl
import data_pipeline as dp

# 1. Load and prepare metadata
meta = dl.load_metadata(config.CFG["data_dir"])
scp_diag = dl.load_scp_statements(config.CFG["data_dir"])

# 2. Build label matrix
Y, mlb, class_names, meta_labeled = dp.build_label_matrix(
    meta, scp_diag, threshold=config.CFG["label_thresh"]
)

# 3. (Optional) Run QC sweep to drop corrupted records
meta_labeled = dl.filter_corrupted_records(meta_labeled, config.CFG["data_dir"])

# 4. Compute class weights for imbalanced loss
weights = dp.compute_class_weights(Y, class_names)

# 5. Build DataLoaders
train_dl, val_dl, test_dl = dl.create_dataloaders(
    meta_labeled, Y, config.CFG["data_dir"]
)

# 6. Inspect a batch
X_batch, Y_batch = next(iter(train_dl))
print(X_batch.shape)  # (64, 12, 5000)
print(Y_batch.shape)  # (64, n_classes)

Notes

The dataset uses a patient-disjoint stratified split: folds 1–8 for training, fold 9 for validation, fold 10 for test. No patient appears in more than one split.
Age values of 300 in the original CSV are a privacy mask for patients over 89 and are converted to NaN automatically.
Signal processing (cleaning, filtering, normalization) is applied lazily inside PTBXLDataset.__getitem__ to keep memory usage low during training.
Run main_test_cop.ipynb in full before training to confirm your environment and dataset are correctly set up.
