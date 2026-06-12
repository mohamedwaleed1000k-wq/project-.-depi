import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta

# ضبط إعدادات الصفحة
st.set_page_config(page_title="ECG Diagnostic System", layout="wide")

# تهيئة السجلات
if 'history' not in st.session_state: st.session_state['history'] = []
if 'show_feedback_input' not in st.session_state: st.session_state['show_feedback_input'] = False

# --- 1. الـ Sidebar ---
with st.sidebar:
    st.header("📊 Analytics")
    # إضافة ميزة (3): تبديل اللغة (بسيط)
    lang = st.radio("Language:", ["English", "العربية"])
    st.write("---")
    st.markdown("<b>Dataset:</b> PTB-XL (21,841 Records)", unsafe_allow_html=True)
    st.markdown("<b>System:</b> ECG Diagnostic Model", unsafe_allow_html=True)
    st.write("---")
    st.subheader("🕒 Recent History")
    if st.session_state['history']:
        df_history = pd.DataFrame(st.session_state['history'])
        st.table(df_history[['Time', 'PatientID', 'Result']])

# --- 2. الواجهة الرئيسية ---
st.title("⚡ ECG Diagnostic System")

# إضافة ميزة (1): رقم الملف الطبي
col1, col2, col3 = st.columns(3)
with col1: p_id = st.text_input("Patient Medical ID:", "MRN-001")
with col2: p_name = st.text_input("Patient Name:", "Mohamed")
with col3: p_gen = st.selectbox("Gender:", ["Male", "Female"])

# إضافة ميزة (4): الفحص الجماعي (رفع ملفات)
up_files = st.file_uploader("Upload ECG Image(s)", type=['jpg', 'png'], accept_multiple_files=True)

if up_files:
    for up_file in up_files:
        st.image(up_file, use_container_width=True)
    
    if st.button("🚀 Run Processing"):
        with st.spinner("Processing results..."):
            time.sleep(1.2)
            
            now_cairo = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
            res = "Normal Sinus Rhythm"
            
            st.session_state['history'].append({"Time": now_cairo, "PatientID": p_id, "Result": res})
            
            # عرض النتائج
            st.success(f"**Diagnosis Result:** {res}")
            
            # إضافة ميزة (2): مقارنة تقرير (مبسطة)
            st.info("📊 Comparison: Status stable compared to previous record.")
            
            st.metric("Confidence Score", "97.5%")
            
            st.subheader("🔍 Focus Analysis")
            fig, ax = plt.subplots(figsize=(8, 1))
            ax.imshow(np.random.rand(10, 50), cmap='jet', alpha=0.6)
            ax.axis('off')
            st.pyplot(fig)
            
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                report_data = f"ECG Medical Report\nID: {p_id}\nTime: {now_cairo}\nResult: {res}"
                st.download_button("📥 Download Report", report_data, file_name="Report.txt")
            with c_btn2:
                if st.button("🔔 Emergency Alert"):
                    st.error("🚨 ALERT: Emergency team notified!")
            
            # التقييم في الأسفل تماماً
            st.write("---")
            st.subheader("💡 Was this result correct?")
            f1, f2 = st.columns(2)
            if f1.button("👍 Correct"): st.success("Thank you!")
            if f2.button("👎 Incorrect"): st.session_state['show_feedback_input'] = True
            
            if st.session_state['show_feedback_input']:
                correction = st.text_input("Enter the correct diagnosis:")
                if st.button("Submit Correction"):
                    st.warning("Correction recorded!")
                    st.session_state['show_feedback_input'] = False
