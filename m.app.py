import os
import tempfile
import shutil
import numpy as np
import streamlit as st
import torch
import wfdb
import matplotlib.pyplot as plt

import data_pipeline as dp
from model import build_model

st.set_page_config(page_title="ECG Diagnosis", page_icon="❤️", layout="wide")

st.title("❤️ ECG Diagnosis using Deep Learning")

# ==========================
# Load Model
# ==========================

@st.cache_resource
def load_model():
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = build_model(
        n_classes=len(checkpoint["class_names"]),
        variant="resnet18"
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["class_names"]

model, class_names = load_model()

if model is None:
    st.error("Model not found.")
    st.stop()

st.success("✅ Model Loaded Successfully")

# ==========================
# Upload
# ==========================

uploaded_files = st.file_uploader(
    "Upload ECG Files (.hea + .dat)",
    type=["hea", "dat"],
    accept_multiple_files=True
)

if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    for file in uploaded_files:
        with open(os.path.join(temp_dir, file.name), "wb") as f:
            f.write(file.getbuffer())

    hea_file = None
    for file in uploaded_files:
        if file.name.endswith(".hea"):
            hea_file = file.name
            break

    if hea_file is None:
        st.error("Please upload .hea file")
        st.stop()

    record_name = os.path.splitext(hea_file)[0]
    record_path = os.path.join(temp_dir, record_name)
