import numpy as np
import matplotlib.pyplot as plt

file_path = '/data0/zxj_data/total/P/P_data.npy'

# 加载.npy文件
data = np.load(file_path, allow_pickle=True)

data = data[0:1, 0:1, :, :]
# 提取出(53, 2400)形状的数据
data_2d = data.squeeze()

offset_step = 0.000001

# 绘制数据
plt.figure(figsize=(50, 30))
for i in range(data_2d.shape[0]):
    # 为每个通道的数据添加偏移量
    offset = i * 5 * offset_step
    plt.plot(data_2d[i] + offset, label=f'Channel {i + 1}')


plt.title('FNIRS Data Visualization')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 保存为jpg格式的图片
plt.savefig('fnirs_data_visualization.jpg')

plt.show()