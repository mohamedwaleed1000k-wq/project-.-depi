import streamlit as st
import torch
import os
import requests
from model import ResNet1D

# رابط الموديل (تأكد من رفعه على الـ Repo)
MODEL_URL = "https://github.com/mohamed-projects/ecg-system/raw/main/best_model.pt" 
MODEL_PATH = 'best_model.pt'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        try:
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            st.error(f"فشل تحميل الموديل: {e}")
            return None
            
    model = ResNet1D(n_leads=12, n_classes=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# إعداد واجهة النظام
st.set_page_config(page_title="ECG Diagnostic System", layout="wide")
st.title("⚡ ECG Diagnostic System")

# الجزء الجانبي (Side Bar)
with st.sidebar:
    st.header("Patient Information")
    patient_id = st.text_input("Patient Medical ID:", "MRN-001")
    patient_name = st.text_input("Patient Name:", "Mohamed")
    age = st.number_input("Age:", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender:", ["Male", "Female"])

# الجزء الأساسي لرفع الصورة
st.subheader("Upload ECG Image(s)")
uploaded_file = st.file_uploader("200MB per file • JPG, PNG", type=['jpg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded ECG', use_column_width=True)
    
    if st.button("Diagnose"):
        with st.spinner('جاري التحليل...'):
            model = load_model()
            if model:
                # هنا سيقوم الموديل بالتحليل (Placeholder)
                st.success(f"تم التحليل بنجاح للمريض: {patient_name}")
                st.write("النتيجة: النظام جاهز لربط مخرجات الموديل.")

# سجل التاريخ
st.divider()
st.subheader("Recent History")
st.info("لم يتم العثور على سجلات سابقة.")
