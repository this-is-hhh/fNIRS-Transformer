import numpy as np
import cv2

# 加载数据
DATA_PATH = '/data0/zxj_data/total/feature/concatenated_data.npy'
data = np.load(DATA_PATH)          # shape (N, 2, 53, 2400)
sample = data[0, 1]                # shape (53, 2400)

# 原始通道坐标（x: 列，y: 行）
coords = {
    0:(2,2), 1:(1,4), 2:(2,4), 3:(3,2), 4:(3,4), 5:(4,3), 6:(2,6),
    7:(3,6), 8:(4,7), 9:(4,1),10:(5,2),11:(6,1),12:(4,5),13:(5,4),
    14:(5,6),15:(6,5),16:(6,3),17:(7,2),18:(7,4),19:(8,3),20:(6,7),
    21:(7,6),22:(8,7),23:(8,1),24:(9,2),25:(10,1),26:(8,5),27:(9,4),
    28:(9,6),29:(10,5),30:(10,3),31:(11,2),32:(11,4),33:(12,3),34:(10,7),
    35:(11,6),36:(12,7),37:(12,1),38:(13,2),39:(14,1),40:(12,5),41:(13,4),
    42:(13,6),43:(14,5),44:(14,3),45:(15,4),46:(15,2),47:(14,7),48:(15,6),
    49:(16,6),50:(16,4),51:(16,2),52:(17,4)
}

# 坐标整体平移，使最小 x, y 变为 0
xs = [x for (x, y) in coords.values()]
ys = [y for (x, y) in coords.values()]
min_x, min_y = min(xs), min(ys)

coords_shifted = {ch: (x - min_x, y - min_y) for ch, (x, y) in coords.items()}

# 网格尺寸（最大坐标+1）
W = max(x for x, y in coords_shifted.values()) + 1  # 宽：x方向
H = max(y for x, y in coords_shifted.values()) + 1  # 高：y方向

# 视频尺寸
scale = 20
out_width = W * scale
out_height = H * scale

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('sample_clean_gray.mp4', fourcc, 10, (out_width, out_height))

# 全局归一化
global_min = sample.min()
global_max = sample.max()

# 写入每帧
T = sample.shape[1]
for t in range(T):
    grid = np.zeros((H, W), dtype=np.float32)
    for ch, (x, y) in coords_shifted.items():
        grid[y, x] = sample[ch, t]  # y 是行，x 是列

    norm = ((grid - global_min) / (global_max - global_min) * 255).astype(np.uint8)
    resized = cv2.resize(norm, (out_width, out_height), interpolation=cv2.INTER_NEAREST)
    frame = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    out.write(frame)

out.release()
print(f"✅ 干净灰度视频已保存，尺寸: {out_width}x{out_height}")
