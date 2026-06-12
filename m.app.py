import streamlit as st
from PIL import Image
import time
# استيراد دالة التنبؤ من الـ Pipeline بتاعكم
from data_pipeline import predict_ecg

# إعدادات الصفحة الأساسية
st.set_page_config(page_title="ECG Diagnostic Classification", page_icon="❤️", layout="centered")

# تصميم مخصص متناسق مع الخلفية الداكنة
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #ff4b4b; text-align: center; font-family: 'Cairo', sans-serif; }
    h3 { color: #ffffff; text-align: center; font-family: 'Cairo', sans-serif; font-weight: normal; }
    p, div, label { font-family: 'Cairo', sans-serif; }
    .result-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border-right: 5px solid #10b981;
        margin-top: 20px;
        text-align: right;
    }
    /* تعديل اتجاه النصوص في القائمة الجانبية والخانات */
    [data-testid="stSidebar"] { text-align: right; }
    .stTextInput input, .stNumberInput input, .stSelectbox div { text-align: right; direction: rtl; }
    </style>
""", unsafe_allow_html=True)

# 1️⃣ إضافة لوحة البيانات الإحصائية في القائمة الجانبية (Sidebar)
with st.sidebar:
    st.markdown("<h2 style='text-align: right; color: #ff4b4b;'>📊 إحصائيات المشروع</h2>", unsafe_allow_html=True)
    st.write("---")
    st.markdown("<p><b>الداتا سيت المستخدمة:</b> PTB-XL Dataset</p>", unsafe_allow_html=True)
    st.markdown("<p><b>حجم بيانات التدريب:</b> 21,841 إشارة رسم قلب</p>", unsafe_allow_html=True)
    st.markdown("<p><b>عدد القنوات الطبية:</b> 12-Lead ECG</p>", unsafe_allow_html=True)
    st.write("---")
    st.markdown("<p style='color: #94a3b8;'>مبادرة بناة مصر الرقمية<br>DEPI - 2026</p>", unsafe_allow_html=True)

# العنوان الرئيسي للموقع
st.markdown("<h1>❤️ نظام تشخيص وتحليل رسم القلب (ECG)</h1>", unsafe_allow_html=True)
st.markdown("<h3>مبادرة بناة مصر الرقمية - DEPI</h3>", unsafe_allow_html=True)
st.write("---")

# 2️⃣ خانة بيانات المريض (Patient Information Form)
st.markdown("<h4 style='text-align: right; color: #ff4b4b;'>📋 بيانات المريض:</h4>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col3:
    patient_name = st.text_input("اسم المريض:", placeholder="محمد أحمد...")
with col2:
    patient_age = st.number_input("السن:", min_value=1, max_value=120, value=30)
with col1:
    patient_gender = st.selectbox("الجنس:", ["ذكر", "أنثى"])

st.write("---")
st.markdown("<p style='text-align: right; color: #cbd5e1; font-size: 18px;'>يرجى رفع صورة تخطيط رسم القلب (ECG Strip) لبدء التحليل.</p>", unsafe_allow_html=True)

# أداة رفع الملفات
uploaded_file = st.file_uploader("اختر صورة رسم القلب", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("---")
    image = Image.open(uploaded_file)
    st.image(image, caption="صورة رسم القلب المرفوعة", use_container_width=True)
    st.markdown("<div style='text-align: right; color: #10b981; font-size: 16px; font-weight: bold;'>✅ تم رفع الصورة بنجاح وجاهزة للموديل</div>", unsafe_allow_html=True)
    
    # زر بدء التشخيص
    if st.button("بدء تشخيص رسم القلب الفعلي 🚀"):
        with st.spinner("جاري تشغيل شبكة الـ CNN واستخراج تفاصيل الـ Waves..."):
            time.sleep(1.5) # وقت المعالجة البرستيج
            
            # تشغيل الموديل الحقيقي
            result, success = predict_ecg(uploaded_file)
            
            if success:
                # عرض النتيجة مدمج معها بيانات المريض اللي كتبتها فوق
                st.markdown(f"""
                    <div class="result-box">
                        <h4 style="color: #10b981; margin-top:0; text-align: right;">📊 التقرير الطبي المعتمد:</h4>
                        <p style="color: #ffffff; text-align: right;"><b>اسم المريض:</b> {patient_name if patient_name else 'غير مسجل'}</p>
                        <p style="color: #ffffff; text-align: right;"><b>السن والجنس:</b> {patient_age} سنة | {patient_gender}</p>
                        <hr style="border-color: #334155;">
                        <p style="color: #ffffff; text-align: right; font-size: 18px;"><b>التصنيف المكتشف بواسطة الموديل:</b> <span style="color: #ff4b4b;">{result}</span></p>
                        <p style="color: #94a3b8; font-size: 13px; margin-bottom: 0; text-align: right;">💡 تم ربط البيانات والمخرج بناءً على نموذج الذكاء الاصطناعي بنجاح.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.error(result)
