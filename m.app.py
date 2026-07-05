import streamlit as st
import os
import tempfile
import shutil
import numpy as np
import torch
import wfdb
import matplotlib.pyplot as plt

# استيراد من ملفاتك (تأكد من وجود model.py و data_pipeline.py)
from model import build_model
import data_pipeline as dp

st.set_page_config(page_title="ECG Diagnosis", page_icon="❤️", layout="wide")
st.title("❤️ ECG Diagnosis using Deep Learning")

@st.cache_resource
def load_model():
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        return None, None, "ملف النموذج غير موجود"
    
    try:
        # تحميل النموذج
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = build_model(n_classes=len(checkpoint["class_names"]))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, checkpoint["class_names"], None
    except Exception as e:
        return None, None, str(e)

model, class_names, error = load_model()

if error:
    st.error(f"فشل تحميل النموذج: {error}")
    st.stop()

st.success("✅ Model Loaded Successfully")

uploaded_files = st.file_uploader("Upload ECG Files (.hea + .dat)", type=["hea", "dat"], accept_multiple_files=True)

if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    for file in uploaded_files:
        with open(os.path.join(temp_dir, file.name), "wb") as f:
            f.write(file.getbuffer())

    hea_file = next((f.name for f in uploaded_files if f.name.endswith(".hea")), None)
    
    if hea_file:
        try:
            record_path = os.path.join(temp_dir, os.path.splitext(hea_file)[0])
            sig, fields = wfdb.rdsamp(record_path)
            
            sig = dp.clean_single_signal(sig)
            sig = dp.bandpass_filter_single(sig)
            sig = dp.normalize_single_signal(sig)
            
            x = torch.tensor(np.transpose(sig, (1, 0)), dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                probs = torch.sigmoid(model(x))[0].numpy()
            
            pred = np.argmax(probs)
            st.write(f"### النتيجة: {class_names[pred]}")
            for c, p in zip(class_names, probs):
                st.write(f"{c}: {p*100:.2f}%")
                st.progress(float(p))
        except Exception as e:
            st.error(f"خطأ: {e}")
    shutil.rmtree(temp_dir)
