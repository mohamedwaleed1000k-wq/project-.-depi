@st.cache_resource
def load_model():
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        return None, None, "ملف النموذج غير موجود"
    
    try:
        # التعديل هنا: إضافة weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        model = build_model(n_classes=len(checkpoint["class_names"]))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, checkpoint["class_names"], None
    except Exception as e:
        return None, None, str(e)
