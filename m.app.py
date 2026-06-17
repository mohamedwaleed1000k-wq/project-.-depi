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

# 3. دالة تحميل الموديل (Cache عشان الموقع يفتح بسرعة)
@st.cache_resource
def load_my_model():
    # تعريف هيكل الموديل
    model = ResNet1D(n_leads=12, n_classes=5)
    
    # التأكد من وجود ملف الأوزان
    model_path = 'best_model.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ملف {model_path} مش موجود في نفس المجلد!")

    # تحميل الأوزان
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # استخراج الـ state_dict بذكاء
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
        
    # تنظيف الأسماء (إزالة 'module.' لو موجودة)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # تحميل الأوزان في الموديل
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# 4. تشغيل الموديل
try:
    with st.spinner("جاري تحميل الموديل..."):
        model = load_my_model()
    st.success("✅ الموديل اشتغل وزي الفل!")
except Exception as e:
    st.error(f"❌ الموديل فيه مشكلة: {e}")
    st.write("تأكد إن ملف 'best_model.pt' و 'model.py' موجودين في نفس مجلد 'm.app.py'")
