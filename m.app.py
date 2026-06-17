import streamlit as st
import torch
import os
import sys

# ده عشان بايثون يلاقي ملف model.py
sys.path.append(os.getcwd())
from model import ResNet1D

st.title("تحليل رسم القلب (ECG)")

# الخطوة دي للتأكد إن الملفات موجودة
files = os.listdir('.')
if 'best_model.pt' not in files:
    st.error(f"ملف best_model.pt مش موجود! الملفات اللي السيرفر شايفها هي: {files}")
    st.stop()

@st.cache_resource
def load_model():
    model = ResNet1D(n_leads=12, n_classes=5)
    # تحميل آمن
    checkpoint = torch.load('best_model.pt', map_location='cpu', weights_only=False)
    
    # تنظيف الأسماء (لو الملف جواه 'module.')
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

try:
    model = load_model()
    st.success("✅ الموديل شغال وزي الفل!")
except Exception as e:
    st.error(f"❌ الموديل فيه مشكلة أثناء التحميل: {e}")
