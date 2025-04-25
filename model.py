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

class FeatureDiscretizer(nn.Module):
    """特征离散化模块，将连续特征转换为离散特征表示"""
    def __init__(self, dim, num_features=5, dropout=0.):
        super().__init__()
        self.num_features = num_features
        
        # 特征提取器 - 将输入特征映射到不同的离散特征空间
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 2, 1)
            ) for _ in range(num_features)
        ])
        
        # 特征激活函数 - 使用sigmoid将特征值映射到0-1之间
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: [batch, seq_len, dim]
        batch_size, seq_len, _ = x.shape
        
        # 提取离散特征
        features = []
        feature_scores = []
        
        for extractor in self.feature_extractors:
            # 提取特征得分 [batch, seq_len, 1]
            score = extractor(x)
            # 应用sigmoid将得分映射到0-1之间
            score = self.sigmoid(score)
            feature_scores.append(score)
            
            # 根据得分生成特征表示
            # 如果得分大于0.5，则认为该特征存在
            feature = (score > 0.5).float()
            features.append(feature)
        
        # 将所有特征拼接 [batch, seq_len, num_features]
        features = torch.cat(features, dim=-1)
        feature_scores = torch.cat(feature_scores, dim=-1)
        
        return features, feature_scores

class InterpretableAttention(nn.Module):
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
        
        # 特征离散化模块
        self.feature_discretizer = FeatureDiscretizer(dim, num_features=5, dropout=dropout)
        
        # 为每个离散特征创建专门的注意力头
        self.feature_attention_weights = nn.Parameter(torch.ones(heads, 5))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        
        # 特征离散化
        discrete_features, feature_scores = self.feature_discretizer(x)
        
        # 计算QKV
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # 计算注意力得分
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        
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
        
        # 根据离散特征调整注意力权重
        # 计算每个头对每个特征的权重
        feature_weights = self.softmax(self.feature_attention_weights)  # [heads, num_features]
        
        # 将特征得分扩展为与注意力矩阵相同的形状
        # [batch, seq_len, num_features] -> [batch, 1, seq_len, 1, num_features]
        expanded_scores = feature_scores.unsqueeze(1).unsqueeze(3)
        
        # 将特征权重扩展为与注意力矩阵相同的形状
        # [heads, num_features] -> [1, heads, 1, 1, num_features]
        expanded_weights = feature_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        # 计算特征加权得分 [batch, heads, seq_len, 1, num_features]
        weighted_scores = expanded_scores * expanded_weights
        
        # 沿特征维度求和，得到每个位置的特征加权得分 [batch, heads, seq_len, 1]
        feature_attention = weighted_scores.sum(dim=-1)
        
        # 将特征加权得分添加到原始注意力得分中
        # 扩展feature_attention以匹配dots的形状 [batch, heads, seq_len, seq_len]
        feature_attention = feature_attention.expand(-1, -1, -1, dots.size(-1))
        
        # 将特征注意力添加到原始注意力中
        dots = dots + feature_attention

        # 计算softmax得到最终注意力权重
        attn = dots.softmax(dim=-1)

        # 应用注意力权重到值向量
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        # 保存特征得分和注意力权重，用于可视化和解释
        self.last_feature_scores = feature_scores
        self.last_attn_weights = attn
        self.last_discrete_features = discrete_features

        return out

class Attention(InterpretableAttention):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,num_channels=53, num_regions=7):
        super().__init__(dim, heads, dim_head, dropout, num_channels, num_regions)

    def forward(self, x, mask=None):
        # 调用父类的forward方法获取注意力输出
        return super().forward(x, mask)

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


# class BidirectionalMambaBlock(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.mamba_forward = Mamba(dim)
#         self.mamba_reverse = Mamba(dim)

#     def forward(self, x):
#         z1 = self.mamba_forward(x)
#         z2 = torch.flip(x, dims=[1])  # Reverse the sequence
#         z2 = self.mamba_reverse(z2)
#         z2 = torch.flip(z2, dims=[1])  # Flip back after processing
        
#         z_prime = z1 + z2
#         z_double_prime = z_prime + x
#         return z_double_prime


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
            
        # 特征名称列表，用于可解释性分析
        self.feature_names = [
            "高频活动",  # 高频脑电活动
            "低频活动",  # 低频脑电活动
            "血氧变化",  # 血氧浓度变化
            "区域连接",  # 脑区连接强度
            "时间变化"   # 时间序列变化特征
        ]


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
        
        # 通过transformer处理
        x2 = self.transformer_channel(combined_features, mask)
        
        # 只取区域令牌部分进行分类（前7个token）
        x2 = x2[:, :7, :]
        x2 = x2.mean(dim=1) if self.pool == 'mean' else x2[:, 0]
        
        # 获取模型的可解释性信息
        self.feature_importance = self._get_feature_importance()
        self.attention_patterns = self._get_attention_patterns()
        
        return self.mlp_head(x2)
    
    def _get_feature_importance(self):
        """获取每个离散特征的重要性"""
        feature_importance = {}
        
        # 遍历Transformer层中的所有注意力模块
        for i, (attn, _) in enumerate(self.transformer_channel.layers):
            # 获取注意力模块
            attention_module = attn.fn.fn
            
            # 检查是否是InterpretableAttention类型
            if hasattr(attention_module, 'feature_attention_weights'):
                # 获取特征权重
                weights = attention_module.feature_attention_weights.softmax(dim=-1)
                
                # 计算每个特征的平均重要性
                avg_weights = weights.mean(dim=0)  # 平均所有头的权重
                
                # 将特征名称与重要性关联
                for j, name in enumerate(self.feature_names):
                    if j < avg_weights.size(0):
                        feature_importance[f"layer_{i}_{name}"] = avg_weights[j].item()
                
                # 如果有最后一次计算的特征得分，也保存下来
                if hasattr(attention_module, 'last_feature_scores'):
                    feature_scores = attention_module.last_feature_scores
                    for j, name in enumerate(self.feature_names):
                        if j < feature_scores.size(-1):
                            feature_importance[f"score_{i}_{name}"] = feature_scores[0, 0, j].item()
        
        return feature_importance
    
    def _get_attention_patterns(self):
        """获取注意力模式，用于可视化"""
        attention_patterns = {}
        
        # 遍历Transformer层中的所有注意力模块
        for i, (attn, _) in enumerate(self.transformer_channel.layers):
            # 获取注意力模块
            attention_module = attn.fn.fn
            
            # 检查是否有保存的注意力权重
            if hasattr(attention_module, 'last_attn_weights'):
                # 获取注意力权重并确保形状正确
                attn_weights = attention_module.last_attn_weights.detach()
                
                # 如果形状是[batch, heads, seq_len, seq_len]，取第一个batch和第一个head
                if len(attn_weights.shape) == 4:
                    attn_weights = attn_weights[0, 0]
                # 如果形状是[heads, seq_len, seq_len]，取第一个head
                elif len(attn_weights.shape) == 3:
                    attn_weights = attn_weights[0]
                
                # 确保注意力权重是二维矩阵
                if len(attn_weights.shape) == 1:
                    seq_len = int(math.sqrt(attn_weights.shape[0]))
                    attn_weights = attn_weights.view(seq_len, seq_len)
                
                # 保存处理后的注意力权重
                attention_patterns[f"layer_{i}_attention"] = attn_weights
                
                # 如果有离散特征，也保存下来
                if hasattr(attention_module, 'last_discrete_features'):
                    attention_patterns[f"layer_{i}_discrete_features"] = attention_module.last_discrete_features.detach()
        
        return attention_patterns
    
    def get_interpretable_features(self):
        """返回模型的可解释性特征，用于分析和可视化"""
        if not hasattr(self, 'feature_importance') or not hasattr(self, 'attention_patterns'):
            raise RuntimeError("必须先运行forward方法才能获取可解释性特征")
        
        return {
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "attention_patterns": self.attention_patterns
        }