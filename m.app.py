import streamlit as st
import torch
import os

# تعريف الموديل من الملف المجاور
from model import ResNet1D

st.set_page_config(page_title="ECG Analysis", layout="centered")
st.title("تحليل رسم القلب (ECG)")

# دالة لتحميل الموديل من المجلد المطلوب
@st.cache_resource
def load_model():
    # المسار المطلوب: checkpoints/best_model.pt
    model_path = os.path.join('checkpoints', 'best_model.pt')
    
    if not os.path.exists(model_path):
        return None
        
    model = ResNet1D(n_leads=12, n_classes=5)
    
    # تحميل الأوزان
    # استخدام weights_only=False هو الخيار الأفضل لتفادي مشاكل الإصدارات الجديدة
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    model.eval()
    return model

model = load_model()

if model:
    st.success("✅ الموديل جاهز للعمل!")
else:
    st.warning("⚠️ بانتظار رفع ملف الموديل في المسار: checkpoints/best_model.pt")
