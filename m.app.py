import streamlit as st
import torch
import os

# تعريف الموديل من الملف المجاور
from model import ResNet1D

st.set_page_config(page_title="ECG Analysis", layout="centered")
st.title("تحليل رسم القلب (ECG)")

# دالة لتحميل الموديل لما توفر ملف الأوزان الجديد
@st.cache_resource
def load_model():
    model_path = 'model_weights.pt' # سمي ملفك الجديد بهذا الاسم
    if not os.path.exists(model_path):
        return None
        
    model = ResNet1D(n_leads=12, n_classes=5)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

model = load_model()

if model:
    st.success("✅ الموديل جاهز للعمل!")
else:
    st.warning("⚠️ بانتظار رفع ملف الموديل (model_weights.pt).")
