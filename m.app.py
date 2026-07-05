import streamlit as st
import torch
import os
import sys

# --- حل مشكلة الـ Import (إجبار النظام على رؤية المجلد الحالي) ---
# هذا الجزء يضمن أن بايثون ستجد ملف model.py حتى لو كانت هناك مشاكل في المسارات
sys.path.append(os.getcwd())

try:
    from model import build_model
    # استيراد ناجح
except ImportError as e:
    st.error(f"خطأ: تعذر العثور على 'model.py'. تأكد أنه في نفس مجلد 'm.app.py'. التفاصيل: {e}")
    st.stop() # إيقاف التطبيق إذا فشل الاستيراد

# --- إعدادات الصفحة ---
st.set_page_config(page_title="ECG Diagnosis App", layout="wide")
st.title("🩺 ECG Analysis Application")

# --- تحميل الموديل ---
@st.cache_resource
def load_trained_model():
    # تأكد من وجود ملف الموديل في مسار checkpoints
    model_path = "checkpoints/best_model.pt"
    
    if not os.path.exists(model_path):
        st.error(f"ملف الموديل غير موجود في المسار: {model_path}")
        return None
    
    try:
        # تحميل الموديل باستخدام دالة build_model من ملف model.py
        model = build_model(n_classes=5)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model
    except Exception as e:
        st.error(f"خطأ أثناء تحميل أوزان الموديل: {e}")
        return None

# تنفيذ عملية التحميل
model = load_trained_model()

if model:
    st.success("تم تحميل الموديل بنجاح!")
    
    # هنا يمكنك إضافة كود رفع ملفات الـ ECG الخاصة بك
    uploaded_file = st.file_uploader("يرجى رفع ملف ECG (مثلاً .npy أو .csv)", type=['npy', 'csv'])
    
    if uploaded_file is not None:
        st.write("تم رفع الملف بنجاح، جاري التحليل...")
        # هنا ستضيف كود التنبؤ (Inference) الخاص بك
else:
    st.warning("الموديل غير جاهز للعمل.")
