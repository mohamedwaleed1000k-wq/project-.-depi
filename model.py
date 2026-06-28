import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=pad, bias=False)
        self.drop = nn.Dropout(p=dropout)
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))
            
    def forward(self, x):
        identity = self.skip(x)
        out = self.bn1(x); out = self.relu(out); out = self.conv1(out)
        out = self.bn2(out); out = self.relu(out); out = self.drop(out); out = self.conv2(out)
        return out + identity

class ResNet1D(nn.Module):
    def __init__(self, n_leads=12, n_classes=5, base_ch=64, layers=[2,2,2,2], kernel_size=7, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_leads, base_ch, kernel_size=15, stride=2, padding=7, bias=False), nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        ch = base_ch
        self.stage1 = self._make_stage(ch, ch, layers[0], 1, kernel_size, dropout)
        self.stage2 = self._make_stage(ch, ch*2, layers[1], 2, kernel_size, dropout)
        self.stage3 = self._make_stage(ch*2, ch*4, layers[2], 2, kernel_size, dropout)
        self.stage4 = self._make_stage(ch*4, ch*8, layers[3], 2, kernel_size, dropout)
        
        # FIXED: تم تعديل طبقة الـ head لتعمل بشكل سليم
        # الـ ch*8 يمثل حجم المعالم بعد الـ stage4
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(), 
            nn.Dropout(p=dropout),
            nn.Linear(ch*8, 256), # طبقة وسيطة لزيادة التعقيد
            nn.ReLU(),
            nn.Linear(256, n_classes) # طبقة الإخراج النهائية
        )
        
    def _make_stage(self, in_ch, out_ch, n_blocks, stride, ks, dp):
        blocks = [ResBlock1D(in_ch, out_ch, ks, stride, dp)]
        for _ in range(1, n_blocks): blocks.append(ResBlock1D(out_ch, out_ch, ks, 1, dp))
        return nn.Sequential(*blocks)
        
    def forward(self, x):
        # المرور عبر الشبكة الأساسية (backbone)
        features = self.stem(x)
        features = self.stage1(features)
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.stage4(features)
        
        # FIXED: المرور الصحيح عبر الـ head الجديد
        output = self.head(features)
        return output
