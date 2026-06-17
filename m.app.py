import streamlit as st
import torch
import os
from model import YourModelClass  # تأكد أن اسم الكلاس داخل model.py هو YourModelClass

# 1. إعداد الصفحة
st.set_page_config(page_title="ECG Diagnostic System", layout="wide")
st.title("⚡ ECG Diagnostic System")

# 2. وظيفة تحميل الموديل
@st.cache_resource
def load_model():
    model_path = 'best_model.pt'
    if not os.path.exists(model_path):
        st.error(f"ملف الموديل {model_path} غير موجود!")
        return None
    
    try:
        # تحميل الموديل
        model = YourModelClass()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل الموديل: {e}")
        return None

# 3. واجهة المستخدم
patient_id = st.text_input("Patient Medical ID:", "MRN-001")
patient_name = st.text_input("Patient Name:", "Mohamed")
age = st.number_input("Age:", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender:", ["Male", "Female"])

uploaded_file = st.file_uploader("Upload ECG Image(s)", type=['jpg', 'png'])

# 4. زر التشخيص
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded ECG', use_column_width=True)
    if st.button("Diagnose"):
        model = load_model()
        if model:
            st.write("جاري المعالجة...")
            # هنا يمكنك إضافة كود المعالجة الخاص بك
            st.success("تم تشخيص الحالة بنجاح!")
