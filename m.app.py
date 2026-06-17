import streamlit as st
import torch
import os
from model import ResNet1D  # الكلاس بتاعك

# 1. إعداد الصفحة
st.set_page_config(page_title="ECG Diagnostic System", layout="wide")
st.title("⚡ ECG Diagnostic System")

# 2. وظيفة تحميل الموديل (مضمونة ومحسنة)
@st.cache_resource
def load_model():
    # المسار الكامل للملف لضمان إن السيستم يلاقيه
    model_path = os.path.join(os.getcwd(), 'best_model.pt')
    
    if not os.path.exists(model_path):
        st.error(f"الموقع مش لاقي ملف الموديل! تأكد إن الملف موجود باسم: {model_path}")
        return None
    
    try:
        # إنشاء الموديل
        model = ResNet1D(n_leads=12, n_classes=5)
        # تحميل الأوزان
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"خطأ تقني أثناء تحميل الموديل: {e}")
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
        with st.spinner('جاري التحليل...'):
            model = load_model()
            if model:
                # هنا الموديل جاهز، المفروض تضيف كود التنبؤ هنا
                st.success("تم تشخيص الحالة بنجاح!")
