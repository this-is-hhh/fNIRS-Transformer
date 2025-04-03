import os
import numpy as np
from scipy.io import loadmat

# 根文件夹路径
root_folder = "/mnt/fNIRS/VFT/P"
# 保存 npy 文件的路径
save_path = "/data0/zxj_data/total/P"

# 确保保存路径存在
os.makedirs(save_path, exist_ok=True)

# 处理 dxy 文件夹
dxy_folder = os.path.join(root_folder, "dxy")
dxy_files = [f for f in os.listdir(dxy_folder) if f.endswith('.mat')]
num_files_dxy = len(dxy_files)
dxy_data = np.zeros((num_files_dxy, 53, 2400))

for i, file in enumerate(dxy_files):
    file_path = os.path.join(dxy_folder, file)
    try:
        data = loadmat(file_path)
        if 'dxydata' in data:
            dxydata = data['dxydata']
            # 截取数据
            dxy_data[i] = dxydata[500:2900].T
        else:
            print(f"{file} 中未找到 dxydata 变量。")
    except Exception as e:
        print(f"读取 {file} 时出错: {e}")

# 处理 oxy 文件夹
oxy_folder = os.path.join(root_folder, "oxy")
oxy_files = [f for f in os.listdir(oxy_folder) if f.endswith('.mat')]
num_files_oxy = len(oxy_files)
oxy_data = np.zeros((num_files_oxy, 53, 2400))

for i, file in enumerate(oxy_files):
    file_path = os.path.join(oxy_folder, file)
    try:
        data = loadmat(file_path)
        if 'oxydata' in data:
            oxydata = data['oxydata']
            # 截取数据
            oxy_data[i] = oxydata[500:2900].T
        else:
            print(f"{file} 中未找到 oxydata 变量。")
    except Exception as e:
        print(f"读取 {file} 时出错: {e}")

# 检查两个文件夹下文件数量是否一致
if num_files_dxy != num_files_oxy:
    print("dxy 和 oxy 文件夹下的 .mat 文件数量不一致，无法进行堆叠操作。")
else:
    # 堆叠数据
    stacked_data = np.stack([dxy_data, oxy_data], axis=1)

    # 保存为 npy 文件
    npy_file_path = os.path.join(save_path, "stacked_data.npy")
    np.save(npy_file_path, stacked_data)

    print(f"数据已保存到 {npy_file_path}")
    