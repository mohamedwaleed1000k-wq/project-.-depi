CFG = dict(
    data_dir  = r"D:\DEPI\ECG project\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    
    sampling_rate = 500,          # 100 or 500
    target_fs     = 500,          # resample target 
    
    bandpass_low  = 0.5,          # Hz  — removes baseline wander
    bandpass_high = 40.0,         # Hz  — removes high-freq noise
    bandpass_ord  = 4,   # filter order
    
    val_fold      = 9,            # strat_fold used as validation
    test_fold     = 10,           # strat_fold used as test (official)
    label_thresh  = 0.0,          # all predictions above this threshold are considered positive
    batch_size    = 64,           #protect
    num_workers   = 0,
)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


DROPPED_ECG_IDS = [
    137, 139, 140, 141, 142, 143, 145, 11817,
    456, 458, 459, 461, 462, 13796, 2506, 2511,
    3795, 3798, 3832, 5817, 7777, 7779, 7782,
    9821, 9825, 9888, 15742, 11810, 11838,
    13791, 13793, 13797, 13799, 18150,
    11814, 11815, 3800, 3801,            
]