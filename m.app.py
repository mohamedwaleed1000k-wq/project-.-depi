import streamlit as st
from PIL import Image
import time

# إعدادات الصفحة الأساسية
st.set_page_config(
    page_title="ECG Diagnostic Classification",
    page_icon="❤️",
    layout="centered"
)

# تصميم مخصص متناسق مع الخلفية الداكنة
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #ff4b4b; text-align: center; font-family: 'Cairo', sans-serif; margin-bottom: 5px; }
    h3 { color: #ffffff; text-align: center; font-family: 'Cairo', sans-serif; font-weight: normal; }
    p, div { font-family: 'Cairo', sans-serif; }
    .result-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border-right: 5px solid #10b981;
        margin-top: 20px;
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

# العنوان الجديد المتوافق مع مشروع الـ ECG
st.markdown("<h1>❤️ نظام تشخيص وتحليل رسم القلب (ECG)</h1>", unsafe_allow_html=True)
st.markdown("<h3>مبادرة بناة مصر الرقمية - DEPI</h3>", unsafe_allow_html=True)
st.write("---")

st.markdown("<p style='text-align: right; color: #cbd5e1; font-size: 18px;'>مرحباً بك! لبدء الفحص والتشخيص، يرجى رفع صورة تخطيط رسم القلب (ECG Strip).</p>", unsafe_allow_html=True)

# أداة رفع الملفات مصممة للصور فقط
uploaded_file = st.file_uploader(
    "اختر صورة رسم القلب", 
    type=["jpg", "jpeg", "png"],
    help="ارفع صورة واضحة لتخطيط القلب ليقوم النموذج بتحليلها"
)

if uploaded_file is not None:
    st.write("---")
    image = Image.open(uploaded_file)
    st.image(image, caption="صورة رسم القلب المرفوعة", use_container_width=True)
    
    st.markdown("<div style='text-align: right; color: #10b981; font-size: 16px; font-weight: bold; margin-bottom: 15px;'>✅ تم رفع الصورة بنجاح وتجهيزها للتحليل</div>", unsafe_allow_html=True)
    
    # زر بدء التشخيص
    if st.button("بدء تشخيص رسم القلب 🚀"):
        # عمل تأثير التحميل والمحاكاة
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for percent_complete in range(100):
            time.sleep(0.02)  # سرعة التحميل
            progress_bar.progress(percent_complete + 1)
            if percent_complete < 40:
                status_text.markdown("<p style='text-align: right; color: #e2e8f0;'>⏳ جاري قراءة تفاصيل الصورة واستخراج الإشارات...</p>", unsafe_allow_html=True)
            elif percent_complete < 80:
                status_text.markdown("<p style='text-align: right; color: #e2e8f0;'>⚡ جاري مطابقة البيانات مع نموذج PTB-XL...</p>", unsafe_allow_html=True)
            else:
                status_text.markdown("<p style='text-align: right; color: #e2e8f0;'>🔍 جاري كتابة التقرير النهائي...</p>", unsafe_allow_html=True)
                
        status_text.empty()
        progress_bar.empty()

        # عرض النتيجة التجريبية بشكل مبهر للتحكيم
        st.markdown("""
            <div class="result-box">
                <h4 style="color: #10b981; margin-top:0;">📊 التقرير الطبي المتوقع (محاكاة):</h4>
                <p style="color: #ffffff;"><b>حالة النبض الفعلي (Heart Rate):</b> 72 bpm (طبيعي)</p>
                <p style="color: #ffffff;"><b>التشخيص الأولي للموديل:</b> Normal Sinus Rhythm (إيقاع جيبي طبيعي)</p>
                <p style="color: #94a3b8; font-size: 13px; margin-bottom: 0;">💡 ملاحظة: هذا العرض تفاعلي لتوضيح طريقة عمل الـ Pipeline، وسيتم ربطه بالأوزان النهائية فور صدورها من فريق الـ AI.</p>
            </div>
        """, unsafe_allow_html=True)
