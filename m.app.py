import streamlit as st
import torch
import os
from model import ResNet1D  # تأكد أن ملف model.py موجود في نفس المجلد

@st.cache_resource
def load_my_model():
    # 1. التأكد من وجود الملف قبل المحاولة
    if not os.path.exists('best_model.pt'):
        st.error("❌ خطأ: ملف 'best_model.pt' غير موجود في المجلد الرئيسي.")
        st.stop()
    
    # 2. تحميل الموديل
    model = ResNet1D(n_leads=12, n_classes=5)
    
    # استخدام weights_only=False لتجنب UnpicklingError
    checkpoint = torch.load('best_model.pt', map_location=torch.device('cpu'), weights_only=False)
    
    # تنظيف الأسماء (إزالة 'module.' إذا كانت موجودة)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
    
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

st.title("تحليل رسم القلب (ECG)")

# تشغيل التحميل
model = load_my_model()
if model:
    st.success("✅ تم تحميل الموديل بنجاح!")
