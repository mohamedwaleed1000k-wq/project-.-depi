import torch
from model import ResNet1D

# 1. تعريف الموديل
model = ResNet1D(n_leads=12, n_classes=5)

# 2. تحميل آمن للموديل
try:
    # نحمل الملف ونفحص إذا كان يحتوي على أوزان مباشرة أو "Checkpoint"
    checkpoint = torch.load('best_model.pt', map_location=torch.device('cpu'), weights_only=False)
    
    # إذا كان الملف يحتوي على 'model_state' أو 'state_dict' (الشائع في التدريب)
    if isinstance(checkpoint, dict):
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # تنظيف أسماء الأوزان (إزالة 'module.' إذا كانت موجودة)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.eval()
    print("✅ الموديل يعمل بنجاح في Colab!")
except Exception as e:
    print(f"❌ حدث خطأ أثناء التحميل: {e}")
