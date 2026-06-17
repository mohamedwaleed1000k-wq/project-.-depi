import streamlit as st
import torch
import os

@st.cache_resource
def load_model():
    # الاستيراد داخل الدالة يمنع مشكلة الـ Circular Import
    from model import ResNet1D
    
    model = ResNet1D(n_leads=12, n_classes=5)
    
    if not os.path.exists('best_model.pt'):
        st.error("ملف الأوزان (best_model.pt) غير موجود في المجلد الرئيسي!")
        return None
        
    checkpoint = torch.load('best_model.pt', map_location='cpu', weights_only=False)
    
    # استخراج الأوزان
    state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

st.title("تحليل رسم القلب (ECG)")

# تحميل وتشغيل
model = load_model()

if model:
    st.success("✅ تم تحميل الموديل بنجاح!")
