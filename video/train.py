import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from dataset import FNIRSVideos
from model import build_model
from utils import evaluate, EarlyStopper

# 配置
DATA_PATH    = '/data0/zxj_data/total/feature/concatenated_data.npy'
LABEL_PATH   = '/data0/zxj_data/total/label/label_squeezed.npy'
BATCH_SIZE   = 8
LR           = 1e-3
EPOCHS       = 50
VAL_SPLIT    = 0.2
WEIGHT_DECAY = 1e-4
RANDOM_SEED  = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载完整标签，并转为整数
labels_all = np.load(LABEL_PATH).astype(np.int64)
dataset_size = len(labels_all)

# 2. 准备完整数据集
full_ds = FNIRSVideos(DATA_PATH, LABEL_PATH)

# 3. Stratified 划分索引
sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
train_idx, val_idx = next(sss.split(np.zeros(dataset_size), labels_all))

# 4. 用 Subset 构造训练/验证集
train_ds = Subset(full_ds, train_idx)
val_ds   = Subset(full_ds, val_idx)

# 5. 构造训练集的 WeightedRandomSampler（可选）
train_labels = labels_all[train_idx]
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# 6. 创建 DataLoader
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 7. 构建模型、损失、优化器、调度器、早停
model = build_model(num_classes=2).to(device)
weight_tensor = torch.from_numpy(class_weights.astype(np.float32)).to(device)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
stopper   = EarlyStopper(patience=7)

# 8. 训练循环
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for videos, labels in train_loader:
        # videos: (B,1,T,H,W) --> repeat to 3 channels
        videos = videos.to(device).repeat(1, 3, 1, 1, 1)
        labels = labels.to(device)

        outputs = model(videos)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * videos.size(0)

    scheduler.step()
    train_loss = running_loss / len(train_ds)

    val_acc, val_cm = evaluate(model, val_loader, device)

    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
    print("Confusion Matrix:\n", val_cm)

    stopper(val_acc)
    if stopper.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# 9. 保存最终模型
torch.save(model.state_dict(), 'r3d18_fnirs_stratified.pth')
print("Model saved to r3d18_fnirs_stratified.pth")
