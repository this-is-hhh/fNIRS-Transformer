import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange
import math
from mamba_ssm import Mamba


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            # nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,num_channels=53, num_regions=7):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # 输入维度是dim（可能是原始dim的2倍）
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.region_to_channels = [
            [channel + 6 for channel in region]
            for region in [
                [1, 4, 10],
                [40, 47, 52],
                [2, 3, 5, 7, 8, 13],
                [44, 46, 49, 50, 51, 53],
                [6, 11, 14, 17, 18, 20, 25, 31, 32, 34, 39, 42, 45],
                [12, 24, 26, 38],
                [9, 15, 16, 19, 21, 22, 23, 27, 28, 29, 30, 33, 35, 36, 37, 41, 43, 48],
            ]
        ]

        self.num_regions = num_regions
        self.num_channels = num_channels


    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        # if mask is not None:
        #     mask = F.pad(mask.flatten(1), (1, 0), value=True)
        #     assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
        #     mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
        #     dots.masked_fill_(~mask, mask_value)
        #     del mask

        
         # 生成交互掩码
        interaction_mask = torch.zeros((1, n, n), dtype=torch.bool, device=x.device)
        
        # Region tokens 只能与指定的通道交互
        for region_idx, channel_indices in enumerate(self.region_to_channels):
            interaction_mask[:, region_idx, channel_indices] = True
            interaction_mask[:, channel_indices, region_idx] = True
        
        # 添加区域令牌之间的交互
        interaction_mask[:, :self.num_regions, :self.num_regions] = True

        # 其他通道可以相互交互
        interaction_mask[:, self.num_regions:, self.num_regions:] = True

        dots.masked_fill_(~interaction_mask, mask_value)


        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out
        # 只返回 7 个 region tokens
        # return out[:, :self.num_regions, :]

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class BidirectionalMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba_forward = Mamba(dim)
        self.mamba_reverse = Mamba(dim)

    def forward(self, x):
        z1 = self.mamba_forward(x)
        z2 = torch.flip(x, dims=[1])  # Reverse the sequence
        z2 = self.mamba_reverse(z2)
        z2 = torch.flip(z2, dims=[1])  # Flip back after processing
        
        z_prime = z1 + z2
        z_double_prime = z_prime + x
        return z_double_prime

class FrequencyDomainEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, sampling_point, dim, n_freq_bands=10):
        super().__init__()
        self.n_freq_bands = n_freq_bands
        self.sampling_point = sampling_point
        
        # 频域特征提取卷积层
        self.freq_conv = nn.Conv1d(in_channels=n_freq_bands, out_channels=out_channels, 
                                  kernel_size=3, padding=1)
        
        # 投影层将频域特征映射到指定维度
        self.proj = nn.Linear(out_channels * in_channels, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x shape: [batch, 2, channels, sampling_points]
        batch, types, channels, _ = x.shape
        
        # 对每个通道进行FFT变换
        freq_features = []
        for t in range(types):  # 对脱氧和有氧分别处理
            for c in range(channels):  # 对每个通道处理
                # 提取当前通道的时域信号
                signal = x[:, t, c, :]
                
                # 应用FFT
                fft_result = torch.fft.fft(signal, dim=1)
                magnitudes = torch.abs(fft_result[:, :self.n_freq_bands])  # 只取前n_freq_bands个频带
                
                # 归一化
                if torch.max(magnitudes) > 0:
                    magnitudes = magnitudes / torch.max(magnitudes)
                
                freq_features.append(magnitudes)
        
        # 将所有频域特征堆叠 [batch, channels*types, n_freq_bands]
        freq_features = torch.stack(freq_features, dim=1)
        
        # 应用卷积提取频域特征 [batch, channels*types, out_channels]
        freq_features = self.freq_conv(freq_features.transpose(1, 2)).transpose(1, 2)
        
        # 展平并投影到指定维度
        freq_features = freq_features.reshape(batch, channels, -1)
        freq_features = self.proj(freq_features)
        
        return self.norm(freq_features)

class MultiScaleEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, sampling_point, dim, kernel_lengths, stride):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=(1, k), stride=(1, stride),
                      padding=(0, (k - 1) // 2))  # 添加 padding
            for k in kernel_lengths
        ])
        out_w = math.floor((sampling_point - max(kernel_lengths) + (max(kernel_lengths) - 1) // 2 * 2) / stride) + 1
        self.proj = nn.Linear(len(kernel_lengths) * out_channels * out_w, dim)
        self.norm = nn.LayerNorm(dim)
        
        # 添加频域特征提取
        self.freq_embedding = FrequencyDomainEmbedding(
            in_channels=in_channels, 
            out_channels=out_channels, 
            sampling_point=sampling_point, 
            dim=dim, 
            n_freq_bands=20
        )

    def forward(self, x):
        # 时域特征提取
        features = [conv(x) for conv in self.conv_layers]
        features = torch.cat(features, dim=1)  # 现在所有 feature 维度匹配了
        features = rearrange(features, 'b c h w -> b h (c w)')
        features = self.proj(features)
        time_features = self.norm(features)
        
        # 频域特征提取
        freq_features = self.freq_embedding(x)
        
        # 拼接时域和频域特征
        # 将同一通道的频域和时域特征在维度上拼接，而不是在序列长度上拼接
        # 时域特征形状: [batch, num_channels, dim]
        # 频域特征形状: [batch, num_channels, dim]
        # 拼接后形状: [batch, num_channels, 2*dim]
        combined_features = torch.cat([freq_features, time_features], dim=2)
        
        return combined_features
        


class fNIRS_T(nn.Module):
    """
    fNIRS-T model

    Args:
        n_class: number of classes.
        sampling_point: sampling points of input fNIRS signals. Input shape is [B, 2, fNIRS channels, sampling points].
        dim: last dimension of output tensor after linear transformation.
        depth: number of Transformer blocks.
        heads: number of the multi-head self-attention.
        mlp_dim: dimension of the MLP layer.
        pool: MLP layer classification mode, 'cls' is [CLS] token pooling, 'mean' is  average pooling, default='cls'.
        dim_head: dimension of the multi-head self-attention, default=64.
        dropout: dropout rate, default=0.
        emb_dropout: dropout for patch embeddings, default=0.
    """
    def __init__(self, n_class, sampling_point, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        # num_patches = 100
        num_channels = 53  # 实际的fNIRS通道数
        # 由于MultiScaleEmbedding返回的特征维度是2*dim，我们需要调整模型的其他部分
        feature_dim = 2 * dim  # 特征维度为原始dim的两倍

        # 多尺度时域特征提取
        self.to_channel_embedding = MultiScaleEmbedding(
            in_channels=2, out_channels=8, sampling_point=sampling_point, dim=dim, kernel_lengths=[50, 25, 12], stride=4
        )

        # 计算位置编码的大小：7个区域令牌 + 通道特征数量
        # 频域和时域特征在维度上拼接，序列长度为num_channels
        total_tokens = 7 + num_channels  # token数量为区域令牌 + 通道特征（每个通道包含频域和时域信息）
        
        # 位置编码 - 注意：维度需要调整为feature_dim以匹配拼接后的特征维度
        self.pos_embedding_channel = nn.Parameter(torch.randn(1, total_tokens, feature_dim))
        
        # 区域令牌 - 维度需要调整为feature_dim以匹配拼接后的特征维度
        self.region_token_channel = nn.Parameter(torch.randn(1, 7, feature_dim))
        
        self.dropout_channel = nn.Dropout(emb_dropout)

        # Transformer和Mamba层 - 维度需要调整为feature_dim
        self.transformer_channel = Transformer(feature_dim, depth, heads, dim_head, mlp_dim, dropout)
        # self.mamba_layers = nn.ModuleList([BidirectionalMambaBlock(feature_dim) for _ in range(depth)])

        self.pool = pool
        self.to_latent = nn.Identity()
        # 增强的MLP头部结构，包含两个全连接层和ReLU激活函数
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, n_class))


    def forward(self, img, mask=None):
        # 获取时域和频域特征，现在形状为 [batch, num_channels, 2*dim]
        x2 = self.to_channel_embedding(img)    
        
        b, n, d = x2.shape
        
        # 不再分割频域和时域特征，而是直接使用它们的组合
        # x2的形状为 [batch, num_channels, 2*dim]，其中包含了频域和时域信息
        
        # 添加区域令牌到特征中
        region_tokens = repeat(self.region_token_channel, '() n d -> b n d', b=b)
        
        # 将区域令牌与通道特征组合（通道特征已经包含了频域和时域信息）
        # 在序列维度上拼接区域令牌和通道特征
        combined_features = torch.cat([region_tokens, x2], dim=1)
        
        # 添加位置编码
        # 注意：位置编码的大小需要匹配 7(区域令牌) + n(通道特征)
        combined_features += self.pos_embedding_channel[:, :combined_features.shape[1]]
        combined_features = self.dropout_channel(combined_features)
        # print("combined_features: ", combined_features.shape)
        # print("region_tokens: ", region_tokens.shape)
        # print("x2: ", x2.shape)
        # print("pos_embedding_channel: ", self.pos_embedding_channel.shape)
        # print("pos_embedding_channel[:, :combined_features.shape[1]]: ", self.pos_embedding_channel[:, :combined_features.shape[1]].shape)
        # print("combined_features + pos_embedding_channel[:, :combined_features.shape[1]]: ", combined_features.shape)
        # 通过transformer处理
        x2 = self.transformer_channel(combined_features, mask)
        # print("x2 after transformer: ", x2.shape)
        
        # 只取区域令牌部分进行分类（前7个token）
        x2 = x2[:, :7, :]
        x2 = x2.mean(dim=1) if self.pool == 'mean' else x2[:, 0]
        
        return self.mlp_head(x2)