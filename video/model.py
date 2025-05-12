import torch.nn as nn
from torchvision.models.video import r3d_18

def build_model(num_classes=2):
    """
    使用 torchvision 内置的 r3d_18 模型（3D ResNet-18），pretrained=False 表示从头训练。
    """
    model = r3d_18(pretrained=False)          # 5-layer 3D CNN, 不联网
    in_feats = model.fc.in_features           # 默认分类头是 model.fc
    model.fc = nn.Linear(in_feats, num_classes)  # 替换成你的二分类头
    return model
