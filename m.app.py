import streamlit as st
import torch
import os
import sys

# التأكد من أن المجلد الحالي مضاف لمسار بايثون
sys.path.append(os.getcwd())

# 1. استيراد كلاس الموديل من الملف الجديد (بعد تغيير الاسم)
try:
    from ecg_model import ResNet1D
except ImportError as e:
    st.error(f"خطأ: تأكد أن ملف 'ecg_model.py' موجود في نفس المجلد. التفاصيل: {e}")
    st.stop()

# 2. إعداد الموديل
@st.cache_resource
def load_model():
    # التأكد من وجود ملف الأوزان
    if not os.path.exists('best_model.pt'):
        st.error("❌ ملف 'best_model.pt' غير موجود في المجلد الرئيسي!")
        return None
        
    try:
        model = ResNet1D(n_leads=12, n_classes=5)
        # تحميل الأوزان بضبط يمنع الـ UnpicklingError
        checkpoint = torch.load('best_model.pt', map_location=torch.device('cpu'), weights_only=False)
        
        # استخراج الأوزان (سواء كانت داخل dict أو مباشرة)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
            
        # تنظيف الأسماء (إزالة 'module.' إذا كانت موجودة)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ خطأ أثناء تحميل الموديل: {e}")
        return None

st.title("تحليل رسم القلب (ECG)")

# تنفيذ التحميل
model = load_model()

if model:
    st.success("✅ الموديل اشتغل وزي الفل!")
