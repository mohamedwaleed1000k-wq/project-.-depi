import streamlit as st
import torch
import torch.nn as nn
import os

# --- تعريف الموديل هنا مباشرة (عشان نلغي الاعتماد على ملف model.py الخارجي) ---
class ResNet1D(nn.Module):
    def __init__(self, n_leads=12, n_classes=5):
        super(ResNet1D, self).__init__()
        self.conv1 = nn.Conv1d(n_leads, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.mean(x, dim=2)
        return self.fc(x)

st.set_page_config(page_title="ECG Analysis", layout="centered")
st.title("تحليل رسم القلب (ECG)")

# --- دالة التحميل ---
@st.cache_resource
def load_model():
    model_path = os.path.join('checkpoints', 'best_model.pt')
    if not os.path.exists(model_path):
        return None
    try:
        model = ResNet1D(n_leads=12, n_classes=5)
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model:
    st.success("✅ الموديل جاهز للعمل!")
else:
    st.warning("⚠️ بانتظار رفع ملف الموديل في المسار: checkpoints/best_model.pt")
