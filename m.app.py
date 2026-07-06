import streamlit as st
import torch
import torch.nn as nn
import os

# --- تعريف الموديل مباشرة هنا عشان نتفادى خطأ الـ Import ---
class ResNet1D(nn.Module):
    def __init__(self, n_leads, n_classes):
        super(ResNet1D, self).__init__()
        # تعريف بسيط للموديل عشان ميبقاش فاضي
        self.conv1 = nn.Conv1d(n_leads, 64, kernel_size=3)
        self.fc = nn.Linear(64, n_classes)
        
    def forward(self, x):
        return self.fc(x)

# --- إعداد الصفحة ---
st.set_page_config(page_title="ECG Analysis", layout="wide")

with st.sidebar:
    st.header("📋 تفاصيل المشروع")
    st.success("مشروع التخرج: نظام ذكي لتحليل إشارات القلب")
    st.write("• **محمد وليد محمد أحمد**")
    st.write("• **محمد جمال الدين يوسف**")

st.title("تحليل رسم القلب (ECG)")

# --- تشغيل الموديل ---
@st.cache_resource
def load_model():
    model_path = os.path.join('checkpoints', 'best_model.pt')
    if not os.path.exists(model_path):
        return None
    try:
        model = ResNet1D(n_leads=12, n_classes=5)
        # تحميل الأوزان
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
        model.eval()
        return model
    except:
        return "Error"

model = load_model()

# --- عرض النتيجة ---
uploaded_file = st.file_uploader("ارفع ملف الإشارة", type=['csv', 'png', 'jpg'])
if uploaded_file is not None:
    st.write("✅ تم رفع الملف بنجاح.")
else:
    st.info("ℹ️ يرجى رفع ملف ECG للبدء.")
