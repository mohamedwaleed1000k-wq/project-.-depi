import os
import streamlit as st
import tempfile
import shutil
import numpy as np
import torch
import wfdb
import matplotlib.pyplot as plt

# --- كود التحقق من وجود الملفات ---
# هذا الجزء سيظهر لك قائمة الملفات في واجهة الموقع للتأكد من وجود model.py و data_pipeline.py
st.write("### ملفات المجلد الحالي (لأغراض التصحيح):")
st.write(os.listdir('.')) 
# ----------------------------------

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
    st.error("Model not found. Ensure 'checkpoints/best_model.pt' exists.")
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

    hea_file = next((f.name for f in uploaded_files if f.name.endswith(".hea")), None)

    if hea_file is None:
        st.error("Please upload .hea file")
    else:
        record_name = os.path.splitext(hea_file)[0]
        record_path = os.path.join(temp_dir, record_name)

        try:
            sig, fields = wfdb.rdsamp(record_path)
            st.success("ECG Loaded Successfully")

            # ECG Plot
            fig, ax = plt.subplots(figsize=(12,3))
            ax.plot(sig[:,0])
            ax.set_title("Lead I")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)

            # Preprocessing
            sig = dp.clean_single_signal(sig)
            sig = dp.bandpass_filter_single(sig)
            sig = dp.normalize_single_signal(sig)

            x = np.transpose(sig, (1,0))
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

            # Prediction
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits)[0].numpy()

            st.header("Prediction")
            pred = np.argmax(probs)
            for c, p in zip(class_names, probs):
                st.write(f"### {c}")
                st.progress(float(p))
                st.write(f"{p*100:.2f}%")

            st.success(f"Predicted Class : *{class_names[pred]}*")
            confidence = probs[pred]*100
            if confidence > 80:
                st.success(f"High Confidence ({confidence:.1f}%)")
            elif confidence > 60:
                st.warning(f"Medium Confidence ({confidence:.1f}%)")
            else:
                st.error(f"Low Confidence ({confidence:.1f}%)")

        except Exception as e:
            st.error(f"Error processing file: {e}")
        
        shutil.rmtree(temp_dir)
