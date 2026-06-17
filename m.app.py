import streamlit as st
import torch
from model import ResNet1D

# 1. إعداد الموديل
model = ResNet1D(n_leads=12, n_classes=5)

# 2. تحميل الأوزان
# تأكد إن ملف 'best_model.pt' موجود فعلاً في نفس المجلد
checkpoint = torch.load('best_model.pt', map_location='cpu')

# 3. إسناد الأوزان للموديل
# (إذا كان الملف عبارة عن dict، بنستخرج الـ state_dict منه)
if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
    model.load_state_dict(checkpoint['model_state'])
else:
    model.load_state_dict(checkpoint)

model.eval()

st.success("الموديل تم تحميله!")
