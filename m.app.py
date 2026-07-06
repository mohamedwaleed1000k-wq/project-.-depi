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
    
    # الحصول على التاريخ والوقت الحالي
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    st.metric("التاريخ", date_str)
    st.metric("الوقت", time_str)
    
    st.divider()
    st.info("نظام تحليل الإشارات القلبية")
    st.caption("جامعة حورس - هندسة الميكاترونيات")

st.title("تحليل رسم القلب (ECG)")

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
    st.write("✅ تم رفع الملف بنجاح.")
else:
    st.info("ℹ️ يرجى رفع ملف ECG للبدء.")
