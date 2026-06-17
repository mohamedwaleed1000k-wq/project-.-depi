import streamlit as st
import torch
import os
import requests
from model import ResNet1D

# رابط التحميل المباشر للملف (تم استخراجه من رابط الدرايف بتاعك)
MODEL_URL = "https://github.com/mohamed-projects/ecg-system/raw/main/best_model.pt" 
MODEL_PATH = 'best_model.pt'

@st.cache_resource
def load_model():
    # تحميل الموديل إذا لم يكن موجوداً
    if not os.path.exists(MODEL_PATH):
        with st.spinner('جاري تحميل الموديل لأول مرة...'):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            
    # تحميل الموديل إلى الذاكرة
    model = ResNet1D(n_leads=12, n_classes=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# واجهة المستخدم
st.title("⚡ ECG Diagnostic System")
patient_id = st.text_input("Patient Medical ID:", "MRN-001")
uploaded_file = st.file_uploader("Upload ECG Image(s)", type=['jpg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded ECG', use_column_width=True)
    if st.button("Diagnose"):
        model = load_model()
        if model:
            st.success("تم تحميل الموديل بنجاح وجاهز للتحليل!")
