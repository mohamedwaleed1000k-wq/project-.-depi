import streamlit as st
import torch
import torch.nn as nn
import os
from datetime import datetime

# --- تعريف الموديل ---
class ResNet1D(nn.Module):
    def __init__(self, n_leads, n_classes):
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(n_leads, 64, kernel_size=3)
        self.fc = nn.Linear(64, n_classes)
        
    def forward(self, x):
        return self.fc(x)

# --- إعداد الصفحة ---
st.set_page_config(page_title="ECG Analysis", layout="wide")

# --- الشريط الجانبي (التاريخ والوقت) ---
with st.sidebar:
    st.header("🕒 معلومات النظام")
    now = datetime.now()
    st.metric("التاريخ", now.strftime("%Y-%m-%d"))
    st.metric("الوقت", now.strftime("%H:%M:%S"))
    st.divider()
    st.info("نظام تحليل الإشارات القلبية")

# --- المحتوى الرئيسي ---
st.title("تحليل رسم القلب (ECG)")

# --- حقول إدخال بيانات المريض ---
st.subheader("📋 بيانات المريض")
col1, col2, col3 = st.columns(3)
with col1:
    patient_name = st.text_input("اسم المريض")
with col2:
    patient_age = st.number_input("السن (العمر)", min_value=0, max_value=120)
with col3:
    patient_gender = st.selectbox("الجنس", ["ذكر", "أنثى"])

st.divider()

# --- تشغيل الموديل ---
@st.cache_resource
def load_model():
    model_path = os.path.join('checkpoints', 'best_model.pt')
    if not os.path.exists(model_path):
        return None
    try:
        model = ResNet1D(n_leads=12, n_classes=5)
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
        model.eval()
        return model
    except:
        return "Error"

model = load_model()

# --- واجهة الرفع ---
uploaded_file = st.file_uploader("ارفع ملف الإشارة", type=['csv', 'png', 'jpg'])

if uploaded_file is not None:
    if patient_name:
        st.write(f"✅ تم استقبال ملف المريض: **{patient_name}**")
        st.write("جاري التحليل...")
    else:
        st.warning("⚠️ يرجى إدخال اسم المريض قبل البدء.")
else:
    st.info("ℹ️ يرجى إدخال بيانات المريض ورفع ملف الإشارة للبدء.")
