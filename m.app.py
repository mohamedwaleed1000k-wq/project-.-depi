import streamlit as st
import torch
import os
from model import ResNet1D  # تأكد أن model.py موجود بجانب m.app.py

@st.cache_resource
def load_my_model():
    # 1. تعريف الموديل
    model = ResNet1D(n_leads=12, n_classes=5)
    
    # 2. تحميل الموديل بضبط خاص لتجنب UnpicklingError
    # weights_only=False تسمح بتحميل الملفات التي تم حفظها بـ pickle
    checkpoint = torch.load('best_model.pt', map_location=torch.device('cpu'), weights_only=False)
    
    # 3. معالجة الـ state_dict إذا كان الملف يحتوي عليه
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
        
    # 4. تنظيف الأسماء (إزالة module. إذا كانت موجودة)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# تنفيذ التحميل
try:
    model = load_my_model()
    st.success("✅ الموديل اشتغل وزي الفل!")
except Exception as e:
    st.error(f"❌ حدث خطأ أثناء التحميل: {e}")
