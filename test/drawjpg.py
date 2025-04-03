import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load('/data0/zxj_data/total/feature/concatenated_data.npy',allow_pickle=True)

# 选择一个样本
sample = data[0]  # 选择第一个样本

# 绘制氧合数据（Oxygenated）的频域图
fig1, axes1 = plt.subplots(1, 10, figsize=(20, 4))
for j in range(10):
    fft_result = np.fft.fft(sample[0, j, :])
    frequencies = np.fft.fftfreq(len(sample[0, j, :]))
    magnitudes = np.abs(fft_result)
    axes1[j].plot(frequencies, magnitudes)
    axes1[j].set_xlabel('Frequency', fontsize=10)
    axes1[j].set_ylabel('Magnitude', fontsize=10)
    axes1[j].set_title(f'Channel {j + 1}, Oxygenated', fontsize=10)
    axes1[j].tick_params(axis='both', labelsize=8)
plt.tight_layout()
plt.savefig('oxygenated_frequency_domain.jpg', dpi=300)

# 绘制脱氧数据（Deoxygenated）的频域图
fig2, axes2 = plt.subplots(1, 10, figsize=(20, 4))
for j in range(10):
    fft_result = np.fft.fft(sample[1, j, :])
    frequencies = np.fft.fftfreq(len(sample[1, j, :]))
    magnitudes = np.abs(fft_result)
    axes2[j].plot(frequencies, magnitudes)
    axes2[j].set_xlabel('Frequency', fontsize=10)
    axes2[j].set_ylabel('Magnitude', fontsize=10)
    axes2[j].set_title(f'Channel {j + 1}, Deoxygenated', fontsize=10)
    axes2[j].tick_params(axis='both', labelsize=8)
plt.tight_layout()
plt.savefig('deoxygenated_frequency_domain.jpg', dpi=300)