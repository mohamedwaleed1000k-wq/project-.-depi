import streamlit as st
from PIL import Image

# إعدادات الصفحة الأساسية
st.set_page_config(
    page_title="ECG Diagnostic Classification",
    page_icon="❤️",
    layout="centered"
)

# تصميم مخصص متناسق مع الخلفية الداكنة
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-family: 'Cairo', sans-serif;
        margin-bottom: 5px;
    }
    h3 {
        color: #ffffff;
        text-align: center;
        font-family: 'Cairo', sans-serif;
        font-weight: normal;
    }
    p {
        font-family: 'Cairo', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# العنوان الجديد المتوافق مع مشروع الـ ECG
st.markdown("<h1>❤️ نظام تشخيص وتحليل رسم القلب (ECG)</h1>", unsafe_allow_html=True)
st.markdown("<h3>مبادرة بناة مصر الرقمية - DEPI</h3>", unsafe_allow_html=True)
st.write("---")

# توجيهات للمستخدم
st.markdown("<p style='text-align: right; color: #cbd5e1; font-size: 18px;'>مرحباً بك! لبدء الفحص والتشخيص، يرجى رفع صورة تخطيط رسم القلب (ECG Strip).</p>", unsafe_allow_html=True)

# أداة رفع الملفات مصممة للصور فقط (PNG, JPG, JPEG)
uploaded_file = st.file_uploader(
    "اختر صورة رسم القلب", 
    type=["jpg", "jpeg", "png"],
    help="ارفع صورة واضحة لتخطيط القلب ليقوم النموذج بتحليلها"
)

# إذا قام المستخدم برفع صورة
if uploaded_file is not None:
    st.write("---")
    # عرض الصورة في الواجهة للتأكيد
    image = Image.open(uploaded_file)
    st.image(image, caption="صورة رسم القلب المرفوعة", use_container_width=True)
    
    # تنسيق رسالة النجاح في جهة اليمين
    st.markdown("<div style='text-align: right; color: #10b981; font-size: 16px; font-weight: bold; margin-bottom: 15px;'>✅ تم رفع الصورة بنجاح وتجهيزها للتحليل</div>", unsafe_allow_html=True)
    
    # زر بدء التشخيص والموديل
    if st.button("بدء تشخيص رسم القلب 🚀"):
        with st.spinner("جاري معالجة الصورة واستخراج الإشارات الرقمية..."):
            # هنا سيتم لاحقاً استدعاء الموديل الخاص بكم من دالة في data_pipeline.py
            st.info("💡 سيتم ربط الموديل الفعلي لعرض نتائج التشخيص وتصنيف الحالات بناءً على نموذج PTB-XL قريباً!")
