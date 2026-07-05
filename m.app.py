import streamlit as st
import os
import tempfile
import shutil
import numpy as np
import torch
import wfdb
import matplotlib.pyplot as plt

# --- استيراد الملفات الخاصة بك ---
try:
    from model import build_model
    import data_pipeline as dp
except ImportError as e:
    st.error(f"خطأ في استيراد الملفات: {e}")
    st.stop()

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
    st.error("لم يتم العثور على ملف النموذج في checkpoints/best_model.pt")
    st.stop()

st.success("✅ Model Loaded Successfully")

# ==========================
# Upload & Processing
# ==========================
uploaded_files = st.file_uploader("Upload ECG Files (.hea + .dat)", type=["hea", "dat"], accept_multiple_files=True)

if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    for file in uploaded_files:
        with open(os.path.join(temp_dir, file.name), "wb") as f:
            f.write(file.getbuffer())

    hea_file = next((f.name for f in uploaded_files if f.name.endswith(".hea")), None)
    
    if hea_file:
        record_name = os.path.splitext(hea_file)[0]
        record_path = os.path.join(temp_dir, record_name)
        try:
            sig, fields = wfdb.rdsamp(record_path)
            
            # معالجة
            sig = dp.clean_single_signal(sig)
            sig = dp.bandpass_filter_single(sig)
            sig = dp.normalize_single_signal(sig)
            
            x = torch.tensor(np.transpose(sig, (1, 0)), dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                probs = torch.sigmoid(model(x))[0].numpy()
            
            pred = np.argmax(probs)
            st.write(f"### النتيجة: {class_names[pred]}")
            
        except Exception as e:
            st.error(f"حدث خطأ أثناء المعالجة: {e}")
    
    shutil.rmtree(temp_dir)
