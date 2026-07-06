import streamlit as st
import tempfile
import os

# إعداد الصفحة
st.set_page_config(page_title="ECG Diagnosis", page_icon="❤️", layout="wide")

st.title("❤️ ECG Diagnosis System")
st.info("ℹ️ النظام يعمل حالياً كواجهة استقبال ملفات (التحليل قيد الصيانة).")

# ==========================
# Upload Section
# ==========================
uploaded_files = st.file_uploader(
    "ارفع ملفات رسم القلب (.hea + .dat)",
    type=["hea", "dat"],
    accept_multiple_files=True
)

if uploaded_files:
    # إنشاء مجلد مؤقت لحفظ الملفات
    temp_dir = tempfile.mkdtemp()
    
    for file in uploaded_files:
        with open(os.path.join(temp_dir, file.name), "wb") as f:
            f.write(file.getbuffer())
    
    st.success(f"✅ تم استقبال {len(uploaded_files)} ملف بنجاح في المسار المؤقت.")
    
    # التأكد من وجود ملف .hea
    hea_files = [f.name for f in uploaded_files if f.name.endswith(".hea")]
    
    if hea_files:
        st.write("ملف الإشارة المكتشف:", hea_files[0])
    else:
        st.warning("⚠️ يرجى التأكد من رفع ملف .hea بجانب ملف .dat")

else:
    st.write("يرجى اختيار الملفات للبدء.")
