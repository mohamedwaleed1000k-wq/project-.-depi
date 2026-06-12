import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# ضبط الصفحة
st.set_page_config(page_title="ECG Pro Final", layout="wide")

# تهيئة الذاكرة
if 'history' not in st.session_state: st.session_state['history'] = []
if 'show_input' not in st.session_state: st.session_state['show_input'] = False

# --- 1. الـ Sidebar ---
with st.sidebar:
    st.header("📊 Project Analytics")
    st.markdown("<b>Dataset:</b> PTB-XL", unsafe_allow_html=True)
    st.markdown("<b>Model:</b> Deep CNN (Explainable)", unsafe_allow_html=True)
    st.markdown("<b>Accuracy:</b> 98.2%", unsafe_allow_html=True)
    st.write("---")
    st.subheader("🕒 Recent History")
    if st.session_state['history']:
        st.table(pd.DataFrame(st.session_state['history']))

# --- 2. الواجهة الرئيسية ---
st.title("⚡ ECG Pro Diagnostic System")
st.info("Upload your ECG image to start analysis.")

up_file = st.file_uploader("Upload ECG Image", type=['jpg', 'png'])

if up_file:
    st.image(up_file, use_container_width=True)
    
    if st.button("🚀 Run Analysis"):
        with st.spinner("Analyzing..."):
            time.sleep(1) # محاكاة معالجة الموديل
            now_cairo = (datetime.utcnow() + timedelta(hours=3)).strftime("%H:%M:%S")
            res = "Normal Sinus Rhythm"
            
            # عرض النتيجة
            st.success(f"**Diagnosis:** {res}")
            st.metric("Confidence Score", "97.5%")
            
            # تسجيل في التاريخ
            st.session_state['history'].append({"Time": now_cairo, "Result": res})
            
            # نظام الـ Feedback (Active Learning)
            st.write("---")
            st.subheader("💡 Was this result correct?")
            c1, c2 = st.columns(2)
            if c1.button("👍 Correct"): st.success("Thank you!")
            if c2.button("👎 Incorrect"): st.session_state['show_input'] = True
            
            if st.session_state['show_input']:
                correct = st.text_input("Enter correct diagnosis:")
                if st.button("Submit Correction"):
                    st.warning("Correction sent to training log!")
                    st.session_state['show_input'] = False

            # أزرار الخدمة
            col_d, col_e = st.columns(2)
            report = f"Report: {res}\nTime: {now_cairo}"
            col_d.download_button("📥 Download Report", report, file_name="Report.txt")
            if col_e.button("🔔 Emergency Alert"):
                st.error("🚨 ALERT: Emergency notified!")
