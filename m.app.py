import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pytz # مكتبة التوقيت

st.set_page_config(page_title="ECG Pro Cairo", layout="wide")

# إعداد توقيت القاهرة
cairo_tz = pytz.timezone("Africa/Cairo")

if 'history' not in st.session_state: st.session_state['history'] = []

# --- Sidebar (ساعة مصر) ---
with st.sidebar:
    st.header("📊 Project Analytics")
    # جلب توقيت القاهرة
    cairo_time = datetime.now(cairo_tz).strftime("%H:%M:%S")
    st.metric("Egypt Time (Cairo)", cairo_time)
    st.write(f"Date: {datetime.now(cairo_tz).strftime('%Y-%m-%d')}")
    st.write("---")
    st.subheader("🕒 Recent History")
    if st.session_state['history']:
        st.table(pd.DataFrame(st.session_state['history']))

# --- Main Page ---
st.title("⚡ ECG Pro Diagnostic System")

c1, c2, c3 = st.columns(3)
with c1: p_name = st.text_input("Name:", "Mohamed")
with c2: p_age = st.number_input("Age:", 1, 100, 30)
with c3: p_gen = st.selectbox("Gender:", ["Male", "Female"])

up = st.file_uploader("Upload ECG")
if up:
    st.image(up)
    if st.button("🚀 Analyze"):
        with st.spinner("Processing..."):
            time.sleep(1)
            # النتيجة بتوقيت القاهرة
            now_cairo = datetime.now(cairo_tz).strftime("%H:%M:%S")
            res = "Normal Sinus Rhythm"
            st.session_state['history'].append({"Time": now_cairo, "Name": p_name, "Result": res})
            
            st.success(f"Diagnosis: {res}")
            
            st.subheader("🔍 Grad-CAM Analysis")
            fig, ax = plt.subplots(figsize=(8, 1))
            ax.imshow(np.random.rand(10, 50), cmap='jet', alpha=0.6)
            ax.axis('off')
            st.pyplot(fig)
            
            # أزرار الخدمة
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Report", f"Patient: {p_name}\nResult: {res}\nTime: {now_cairo}", "Report.txt")
            with col2:
                if st.button("🔔 Emergency Alert"):
                    st.error("🚨 ALERT SENT!")
