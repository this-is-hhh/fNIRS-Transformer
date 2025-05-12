import numpy as np
import torch
from torch.utils.data import Dataset

channel_coordinates = {
    0:(2,2), 1:(1,4), 2:(2,4), 3:(3,2), 4:(3,4), 5:(4,3), 6:(2,6),
    7:(3,6), 8:(4,7), 9:(4,1),10:(5,2),11:(6,1),12:(4,5),13:(5,4),
    14:(5,6),15:(6,5),16:(6,3),17:(7,2),18:(7,4),19:(8,3),20:(6,7),
    21:(7,6),22:(8,7),23:(8,1),24:(9,2),25:(10,1),26:(8,5),27:(9,4),
    28:(9,6),29:(10,5),30:(10,3),31:(11,2),32:(11,4),33:(12,3),34:(10,7),
    35:(11,6),36:(12,7),37:(12,1),38:(13,2),39:(14,1),40:(12,5),41:(13,4),
    42:(13,6),43:(14,5),44:(14,3),45:(15,4),46:(15,2),47:(14,7),48:(15,6),
    49:(16,6),50:(16,4),51:(16,2),52:(17,4)
}

class FNIRSVideos(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)        # (N, 2, 53, 2400)
        self.labels = np.load(label_path)     # (N,)
        assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx, 1]            # (53, 2400)
        label  = int(self.labels[idx])

        # 构建 (T=2400, H=18, W=8)
        video = np.zeros((2400, 18, 8), dtype=np.float32)
        for ch in range(53):
            x, y = channel_coordinates[ch]
            video[:, x, y] = sample[ch]

        # 转成 (1, T, H, W)
        video = torch.from_numpy(video).unsqueeze(0)
        return video, torch.tensor(label, dtype=torch.long)
