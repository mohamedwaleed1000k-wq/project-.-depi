import streamlit as st
import torch
from model import ResNet1D  # لازم ملف model.py يكون موجود في نفس المجلد

@st.cache_resource
def load_my_model():
    # 1. تعريف الموديل
    model = ResNet1D(n_leads=12, n_classes=5)
    
    # 2. تحميل الملف
    checkpoint = torch.load('best_model.pt', map_location=torch.device('cpu'))
    
    # 3. استخراج الأوزان (بناءً على اللي نجح معانا في Colab)
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # 4. تنظيف الأسماء (إزالة كلمة module)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 5. تركيب الأوزان
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# تشغيل الموديل
model = load_my_model()
st.success("الموديل شغال وزي الفل!")
