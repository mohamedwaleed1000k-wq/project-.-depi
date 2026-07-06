import os
import tempfile
import streamlit as st
import torch
import wfdb

# حماية عملية الاستيراد لمنع توقف الموقع
try:
    import data_pipeline as dp
    from model import build_model
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    st.error(f"خطأ في استيراد الملفات: {e}")

st.set_page_config(page_title="ECG Diagnosis", page_icon="❤️", layout="wide")
st.title("❤️ ECG Diagnosis using Deep Learning")

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_model():
    checkpoint_path = "checkpoints/best_model.pt"
    if not MODEL_AVAILABLE or not os.path.exists(checkpoint_path):
        return None, None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = build_model(
            n_classes=len(checkpoint["class_names"]),
            variant="resnet18"
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, checkpoint["class_names"]
    except Exception:
        return None, None

model, class_names = load_model()

if model is None:
    st.warning("⚠️ الموديل غير محمل حالياً (الموقع يعمل كواجهة).")
else:
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
        st.write(f"تم استقبال الملف: {hea_file}")
        # هنا سيعمل الكود الأصلي الخاص بك لاحقاً بمجرد جاهزية الموديل
