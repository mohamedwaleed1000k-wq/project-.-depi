import streamlit as st
import torch
import os
import sys

# 1. إعداد المسارات عشان بايثون يشوف ملف model.py
sys.path.append(os.getcwd())
from model import ResNet1D 

# 2. إعداد صفحة الموقع
st.set_page_config(page_title="ECG Analysis", layout="centered")
st.title("تحليل رسم القلب (ECG)")

# 3. دالة تحميل الموديل الذكية
@st.cache_resource
def load_my_model():
    # تعريف هيكل الموديل
    model = ResNet1D(n_leads=12, n_classes=5)
    
    # التأكد من وجود ملف الأوزان
    model_path = 'best_model.pt'
    if not os.path.exists
