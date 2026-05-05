
RECOMMENDED ARCHITECTURE: 1D-ResNet

This is the recommended baseline model for the PTB-XL dataset 
based on official benchmarks.

• Input  : (batch_size, leads=12, time_steps=5000)  [For 500Hz]
• Stem   : Conv1d(12 -> 64, kernel=15, stride=2) -> BatchNorm -> ReLU -> MaxPool
• Blocks : 4 × ResBlock groups with skip connections
           [64->64 (x2), 64->128 (x2), 128->256 (x2), 256->512 (x2)]
• Head   : GlobalAvgPool1d -> Linear(512 -> 5) -> Sigmoid
• Loss   : BCEWithLogitsLoss(pos_weight=class_weights_tensor)
• Optim  : AdamW(lr=1e-3, weight_decay=1e-4)
• Sched  : CosineAnnealingLR
• Metric : macro AUROC (official PTB-XL benchmark metric)

*** Transformers / xresnet1d50 show marginal improvement but are 3x slower.
