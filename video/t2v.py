import numpy as np

# 假设你的npy文件名为data.npy，根据实际情况修改文件名
file_path = '/data0/zxj_data/total/feature/concatenated_data.npy'
data = np.load(file_path)
print(data.shape)