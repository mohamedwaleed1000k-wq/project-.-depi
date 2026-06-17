import streamlit as st
import torch
from model import ResNet1D  # استدعينا الكلاس الصح

# 1. إعداد الصفحة
st.set_page_config(page_title="ECG Diagnostic System", layout="wide")
st.title("⚡ ECG Diagnostic System")

# 2. وظيفة تحميل الموديل
@st.cache_resource
def load_model():
    # إنشاء نسخة من الموديل (تأكد من نفس المعطيات اللي استخدمتها في التدريب)
    model = ResNet1D(n_leads=12, n_classes=5) 
    
    # تحميل الأوزان من ملف best_model.pt
    try:
        model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"خطأ في تحميل الأوزان: {e}")
        return None

# 3. واجهة المستخدم
patient_id = st.text_input("Patient Medical ID:", "MRN-001")
patient_name = st.text_input("Patient Name:", "Mohamed")
# ... (باقي كود الواجهة) ...

uploaded_file = st.file_uploader("Upload ECG Image(s)", type=['jpg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded ECG', use_column_width=True)
    if st.button("Diagnose"):
        model = load_model()
        if model:
            st.write("جاري التحليل باستخدام ResNet1D...")
            # هنا ستحتاج لإضافة كود تحويل الصورة إلى Tensor (Preprocessing)
            st.success("تم التشخيص بنجاح!")
