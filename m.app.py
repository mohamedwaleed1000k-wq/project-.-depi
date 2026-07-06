import streamlit as st
import torch
import os
from model import ResNet1D

# إعداد الصفحة
st.set_page_config(page_title="ECG Analysis", layout="wide")

# --- الشريط الجانبي (التدليع والبيانات) ---
with st.sidebar:
    st.header("📋 تفاصيل المشروع")
    st.success("مشروع التخرج: نظام ذكي لتحليل إشارات القلب")
    
    st.divider()
    
    st.subheader("👨‍💻 فريق العمل")
    st.write("• **محمد وليد محمد أحمد**")
    st.write("• **محمد جمال الدين يوسف**")
    
    st.divider()
    
    st.subheader("🎓 الإشراف والجامعة")
    st.write("جامعة حورس (HUE)")
    st.write("هندسة الميكاترونيات")
    
    st.divider()
    
    st.subheader("📅 تاريخ المشروع")
    st.write("يونيو - يوليو 2026")

# --- المحتوى الرئيسي ---
st.title("تحليل رسم القلب (ECG)")
st.write("مرحباً بك في نظام تحليل إشارات القلب الذكي.")

# دالة تحميل الموديل
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
    except Exception:
        return "Error"

model = load_model()

# واجهة الرفع
uploaded_file = st.file_uploader("ارفع ملف الإشارة (CSV/Image)", type=['csv', 'png', 'jpg'])

if uploaded_file is not None:
    if model is None:
        st.warning("⚠️ الموديل غير موجود حالياً في المسار: checkpoints/best_model.pt")
    elif model == "Error":
        st.error("❌ حدث خطأ في تحميل ملف الموديل. تأكد من سلامة الملف.")
    else:
        st.write("✅ تم تحميل الموديل بنجاح، جاري معالجة الإشارة...")
        # كود التحليل الخاص بك يوضع هنا لاحقاً
else:
    st.info("ℹ️ يرجى رفع ملف ECG للبدء في التحليل.")
