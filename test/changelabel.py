import numpy as np

# 加载原始文件
label_path = '/data0/zxj_data/total/label/label.npy'
labels = np.load(label_path)

# 打印原始形状
print("Original shape:", labels.shape)

# 调整形状
labels_squeezed = labels.squeeze()

# 打印调整后的形状
print("Squeezed shape:", labels_squeezed.shape)

# 保存修改后的文件
np.save('/data0/zxj_data/total/label/label_squeezed.npy', labels_squeezed)

print("Modified label saved to /data0/zxj_data/total/label/label_squeezed.npy")