import streamlit as st
import pandas as pd

# إعدادات شكل الصفحة
st.set_page_config(page_title="مشروع تخرج DEPI", page_icon="🧠", layout="centered")

# عنوان الموقع
st.title("🧠 نظام تحليل إشارات الدماغ EEG")
st.subheader("مبادرة بناة مصر الرقمية - DEPI")
st.markdown("---")

st.write("مرحباً بك! يرجى رفع ملف البيانات الوصفية (Metadata) لبدء التحليل.")

# زرار لرفع الملفات
uploaded_file = st.file_uploader("اختر ملف CSV", type=["csv"])

if uploaded_file is not None:
    st.success("✅ تم رفع الملف بنجاح!")
    
    df = pd.read_csv(uploaded_file)
    st.write("📋 **معاينة البيانات المرفوعة:**")
    st.dataframe(df.head(5))
    
    if st.button("🚀 بدء التحليل والتنبؤ"):
        st.info("جاري تشغيل الـ Pipeline وفحص البيانات... (سيتم ربط الموديل الفعلي في الخطوة القادمة)")
