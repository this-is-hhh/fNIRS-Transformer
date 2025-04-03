import os
import numpy as np

# 加载两个 npy 文件
n_data_path = "/data0/zxj_data/total/N/N_data.npy"
p_data_path = "/data0/zxj_data/total/P/P_data.npy"

n_data = np.load(n_data_path)
p_data = np.load(p_data_path)

# 拼接数据
concatenated_data = np.concatenate((n_data, p_data), axis=0)

# 确保保存路径存在
save_feature_path = "/data0/zxj_data/total/feature"
save_label_path = "/data0/zxj_data/total/label"

os.makedirs(save_feature_path, exist_ok=True)
os.makedirs(save_label_path, exist_ok=True)

# 保存拼接后的数据
feature_file_path = os.path.join(save_feature_path, "concatenated_data.npy")
np.save(feature_file_path, concatenated_data)

# 创建标签
n_samples = n_data.shape[0]
p_samples = p_data.shape[0]
labels = np.concatenate((np.zeros((n_samples, 1)), np.ones((p_samples, 1))), axis=0)

# 保存标签
label_file_path = os.path.join(save_label_path, "label.npy")
np.save(label_file_path, labels)

print(f"拼接后的数据已保存到 {feature_file_path}")
print(f"标签文件已保存到 {label_file_path}")
    