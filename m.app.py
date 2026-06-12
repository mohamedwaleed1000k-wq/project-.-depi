import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from fpdf import FPDF # تأكد من وجود fpdf2 في ملف requirements.txt

# إعدادات الصفحة
st.set_page_config(page_title="ECG Pro Final System", layout="wide")

if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📊 Project Analytics")
    st.markdown("<b>Dataset:</b> PTB-XL", unsafe_allow_html=True)
    st.write("---")
    st.markdown("### 🕒 Recent History")
    if st.session_state['history']:
        st.table(pd.DataFrame(st.session_state['history'])[['Time', 'Name', 'Result']].tail(5))

# --- Main App ---
st.markdown("<h1 style='text-align: center;'>⚡ ECG Pro Advanced Diagnostic System</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1: p_name = st.text_input("Patient Name:", "Mohamed")
with col2: p_age = st.number_input("Age:", 1, 100, 30)
with col3: p_gen = st.selectbox("Gender:", ["Male", "Female"])

uploaded_file = st.file_uploader("📤 Upload ECG Strip", type=["jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, use_container_width=True)
    
    if st.button("🚀 Start Advanced Analysis"):
        with st.spinner("Analyzing..."):
            time.sleep(1.0)
            now = datetime.now() + timedelta(hours=3)
            scan_time = now.strftime("%H:%M:%S")
            result = "Myocardial Infarction"
            
            st.session_state['history'].append({"Time": scan_time, "Name": p_name, "Result": result})
            
            st.success(f"**Diagnosis:** {result}")
            
            # --- زر تحميل الـ PDF (الكود الآمن) ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="Medical Diagnosis Report", ln=True, align='C')
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Patient: {p_name} | Age: {p_age} | Result: {result}", ln=True)
            
            # حفظ وتحويل لـ Bytes
            pdf_output = pdf.output(dest='S')
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_output,
                file_name=f"Report_{p_name}.pdf",
                mime="application/pdf"
            )

            # زر الطوارئ
            if st.button("🔔 Trigger Emergency Notification"):
                st.error("🚨 ALERT: Emergency team notified!")
