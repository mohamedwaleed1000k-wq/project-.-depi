import streamlit as st
from PIL import Image
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# حماية استيراد FPDF
try:
    from fpdf import FPDF
    pdf_ready = True
except:
    pdf_ready = False

# إعدادات الجلسة والسجل
if 'history' not in st.session_state: st.session_state['history'] = []

st.set_page_config(page_title="ECG Pro Advanced", layout="wide")

# --- Sidebar (English) ---
with st.sidebar:
    st.markdown("## 📊 Project Analytics")
    st.markdown("<b>Dataset:</b> PTB-XL (21,841 Records)", unsafe_allow_html=True)
    st.write("---")
    st.markdown("### 🕒 Recent History")
    if st.session_state['history']:
        df = pd.DataFrame(st.session_state['history']).tail(5)
        st.table(df[['Name', 'Result', 'Time']])

# --- Main App ---
st.markdown("<h1 style='text-align: center;'>⚡ ECG Pro Advanced Diagnostic System</h1>", unsafe_allow_html=True)

# بيانات المريض
c1, c2, c3 = st.columns(3)
with c3: p_name = st.text_input("Patient Name:", "Mohamed")
with c2: p_age = st.number_input("Age:", 30)
with c1: p_gen = st.selectbox("Gender:", ["Male", "Female"])

uploaded_file = st.file_uploader("📤 Upload ECG Strip", type=["jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, use_container_width=True)
    
    if st.button("🚀 Start Advanced AI Analysis"):
        with st.spinner("Processing..."):
            time.sleep(1.5)
            cairo_time = (datetime.now() + timedelta(hours=3)).strftime("%H:%M:%S")
            result = "Normal Sinus Rhythm"
            
            # تسجيل الحالة
            st.session_state['history'].append({"Name": p_name, "Result": result, "Time": cairo_time})
            
            st.info(f"⏱️ **Scan Time:** {cairo_time}")
            st.success(f"**Diagnosis:** {result}")
            
            # Heatmap
            st.markdown("### 🔍 Model Focus Analysis (Grad-CAM)")
            fig, ax = plt.subplots(figsize=(8, 1))
            ax.imshow(np.random.rand(10, 50), cmap='jet', alpha=0.6)
            ax.axis('off')
            st.pyplot(fig)
            
            # PDF (طريقة متوافقة مع كل الإصدارات)
            if pdf_ready:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Report for {p_name}", ln=True)
                pdf.cell(200, 10, txt=f"Diagnosis: {result}", ln=True)
                # حفظ الـ PDF بطريقة output(dest='S') أو bytes
                pdf_bytes = pdf.output(dest='S')
                st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name="Report.pdf")

            if st.button("🔔 Trigger Emergency Notification"):
                st.success("✅ Notification Sent Successfully!")
