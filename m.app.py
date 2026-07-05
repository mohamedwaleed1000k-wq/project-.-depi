import streamlit as st
import os
import tempfile
import shutil
import numpy as np
import torch
import wfdb
import matplotlib.pyplot as plt
from model import build_model
import data_pipeline as dp

st.set_page_config(page_title="ECG Diagnosis", page_icon="❤️", layout="wide")

st.title("❤️ ECG Diagnosis using Deep Learning")

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        st.error(f"ملف النموذج غير موجود في: {checkpoint_path}")
        return None, None

    # تحميل النموذج
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # بناء النموذج باستخدام الدالة التي أضفناها
    model = build_model(n_classes=len(checkpoint["class_names"]))

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["class_names"]

model, class_names = load_model()

if model is None:
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
            
            # معالجة الإشارة
            sig = dp.clean_single_signal(sig)
            sig = dp.bandpass_filter_single(sig)
            sig = dp.normalize_single_signal(sig)
            
            # تحويل الإشارة لـ Tensor (تأكد من مطابقة الأبعاد لما يتوقعه ResNet1D)
            x = torch.tensor(np.transpose(sig, (1, 0)), dtype=torch.float32).unsqueeze(0)
            
            # التنبؤ
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits)[0].numpy()
            
            pred = np.argmax(probs)
            st.header("النتيجة:")
            st.write(f"### النوع المتوقع: {class_names[pred]}")
            
            # عرض الاحتمالات
            for c, p in zip(class_names, probs):
                st.write(f"{c}: {p*100:.2f}%")
                st.progress(float(p))

        except Exception as e:
            st.error(f"خطأ أثناء المعالجة: {e}")
    
    shutil.rmtree(temp_dir)
