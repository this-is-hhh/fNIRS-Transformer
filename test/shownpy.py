import numpy as np

file_path = '/data0/zxj_data/total/P/P_data.npy'

# 加载.npy文件
data = np.load(file_path,allow_pickle=True)

# 打印数据的形状
print("数据形状:", data.shape)
print(data[0].dtype)