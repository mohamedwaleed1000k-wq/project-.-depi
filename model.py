import streamlit as st
import torch
import os
import sys

# التأكد من أن المجلد الحالي موجود في مسار بايثون
sys.path.append(os.getcwd())

# استيراد كلاس الموديل من ملف model.py
try:
    from model import ResNet1D
except ImportError as e:
    st.error(f"خطأ في استيراد الموديل: {e}")
    st.stop()

# إعداد الموديل
@st.cache_resource
def load_model():
    model = ResNet1D(n_leads=12, n_classes=5)
    
    # تحميل الأوزان - تأكد من وجود الملف best_model.pt في نفس المجلد
    if not os.path.exists('best_model.pt'):
        st.error("ملف best_model.pt غير موجود في المجلد!")
        return None
        
    checkpoint = torch.load('best_model.pt', map_location=torch.device('cpu'), weights_only=False)
    
    # معالجة الأوزان (سواء كانت داخل قاموس أو مباشرة)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
        
    # تنظيف الأسماء (إزالة 'module.' إذا كانت موجودة)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

st.title("تحليل رسم القلب (ECG)")

# تحميل وتشغيل الموديل
model = load_model()

if model:
    st.success("✅ تم تحميل الموديل بنجاح!")
