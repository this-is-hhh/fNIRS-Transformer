import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import fNIRS_T, AttentionCodebook
from dataloader import Dataset, Load_Dataset_A, Load_Dataset_B, Load_Dataset_C


def visualize_attention_matrices(model, data_loader, device, save_dir, num_samples=5):
    """可视化模型中的注意力矩阵
    
    Args:
        model: 训练好的fNIRS-T模型
        data_loader: 数据加载器
        device: 设备
        save_dir: 保存可视化结果的目录
        num_samples: 每个类别要可视化的样本数量
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 确保模型处于评估模式
    model.eval()
    
    # 获取codebook
    healthy_codebook = model.attention_codebook.codebook_healthy.detach().cpu().numpy()
    patient_codebook = model.attention_codebook.codebook_patient.detach().cpu().numpy()
    
    # 可视化codebook中的注意力矩阵
    visualize_codebook(healthy_codebook, os.path.join(save_dir, 'healthy_codebook.png'), title='健康人Codebook')
    visualize_codebook(patient_codebook, os.path.join(save_dir, 'patient_codebook.png'), title='病人Codebook')
    
    # 收集每个类别的样本
    samples_by_class = {}
    attention_by_class = {}
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 获取最后一层的注意力矩阵
            # 修改模型的forward方法以返回注意力矩阵
            combined_features = model.to_channel_embedding(inputs)
            b, n, d = combined_features.shape
            region_tokens = torch.repeat_interleave(model.region_token_channel, b, dim=0)
            combined_features = torch.cat([region_tokens, combined_features], dim=1)
            combined_features += model.pos_embedding_channel[:, :combined_features.shape[1]]
            combined_features = model.dropout_channel(combined_features)
            
            # 获取最后一层的注意力矩阵
            _, attention_matrices = model.transformer_channel(combined_features, return_attention=True)
            
            # 按类别收集样本和注意力矩阵
            for i, label in enumerate(labels):
                label_int = label.item()
                if label_int not in samples_by_class:
                    samples_by_class[label_int] = []
                    attention_by_class[label_int] = []
                
                if len(samples_by_class[label_int]) < num_samples:
                    samples_by_class[label_int].append(inputs[i].cpu().numpy())
                    attention_by_class[label_int].append(attention_matrices[i].cpu().numpy())
            
            # 检查是否已收集足够的样本
            all_collected = True
            for label in samples_by_class:
                if len(samples_by_class[label]) < num_samples:
                    all_collected = False
                    break
            
            if all_collected:
                break
    
    # 可视化每个类别的注意力矩阵
    for label in attention_by_class:
        class_name = '健康人' if label == 0 else '病人'
        for i, attn_matrices in enumerate(attention_by_class[label]):
            # 对每个头的注意力矩阵进行可视化
            for h in range(attn_matrices.shape[0]):
                attn_matrix = attn_matrices[h]
                plt.figure(figsize=(10, 8))
                sns.heatmap(attn_matrix, cmap='viridis')
                plt.title(f'{class_name} - 样本 {i+1} - 注意力头 {h+1}')
                plt.xlabel('通道')
                plt.ylabel('通道')
                plt.savefig(os.path.join(save_dir, f'{class_name}_sample{i+1}_head{h+1}.png'))
                plt.close()


def visualize_codebook(codebook, save_path, title='Codebook'):
    """可视化codebook中的注意力矩阵
    
    Args:
        codebook: codebook张量，形状为 [K, C, C]
        save_path: 保存路径
        title: 图表标题
    """
    K, C, C = codebook.shape
    
    # 计算网格大小
    grid_size = int(np.ceil(np.sqrt(K)))
    
    plt.figure(figsize=(20, 20))
    for k in range(K):
        if k >= K:
            break
        
        plt.subplot(grid_size, grid_size, k + 1)
        sns.heatmap(codebook[k], cmap='viridis', xticklabels=False, yticklabels=False)
        plt.title(f'Prototype {k+1}')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def analyze_codebook_similarity(model, save_dir):
    """分析codebook中原型之间的相似度
    
    Args:
        model: 训练好的fNIRS-T模型
        save_dir: 保存可视化结果的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取codebook
    healthy_codebook = model.attention_codebook.codebook_healthy.detach().cpu()
    patient_codebook = model.attention_codebook.codebook_patient.detach().cpu()
    
    # 将codebook展平为向量
    K, C, C = healthy_codebook.shape
    healthy_vecs = healthy_codebook.view(K, -1)
    patient_vecs = patient_codebook.view(K, -1)
    
    # 计算健康人codebook内部相似度
    healthy_sim = torch.matmul(torch.nn.functional.normalize(healthy_vecs, dim=1), 
                              torch.nn.functional.normalize(healthy_vecs, dim=1).T)
    
    # 计算病人codebook内部相似度
    patient_sim = torch.matmul(torch.nn.functional.normalize(patient_vecs, dim=1), 
                              torch.nn.functional.normalize(patient_vecs, dim=1).T)
    
    # 计算健康人和病人codebook之间的相似度
    cross_sim = torch.matmul(torch.nn.functional.normalize(healthy_vecs, dim=1), 
                            torch.nn.functional.normalize(patient_vecs, dim=1).T)
    
    # 可视化相似度矩阵
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(healthy_sim.numpy(), cmap='viridis')
    plt.title('健康人Codebook内部相似度')
    
    plt.subplot(1, 3, 2)
    sns.heatmap(patient_sim.numpy(), cmap='viridis')
    plt.title('病人Codebook内部相似度')
    
    plt.subplot(1, 3, 3)
    sns.heatmap(cross_sim.numpy(), cmap='viridis')
    plt.title('健康人-病人Codebook相似度')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'codebook_similarity.png'))
    plt.close()


def main():
    # 设置参数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = 'path/to/your/trained/model.pt'  # 修改为你的模型路径
    save_dir = 'codebook_visualization'
    data_path = 'data'  # 修改为你的数据路径
    
    # 加载数据
    feature, label = Load_Dataset_A(data_path, model='fNIRS-T')  # 根据你的数据集修改
    _, _, channels, sampling_points = feature.shape
    
    # 创建数据加载器
    dataset = Dataset(feature, label, transform=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 创建模型
    model = fNIRS_T(
        n_class=2,  # 根据你的任务修改
        sampling_point=sampling_points,
        dim=64,  # 根据你的模型修改
        depth=6,
        heads=8,
        mlp_dim=64,
        use_codebook=True,
        codebook_size=32
    ).to(device)
    
    # 加载训练好的模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 可视化注意力矩阵
    visualize_attention_matrices(model, data_loader, device, save_dir)
    
    # 分析codebook相似度
    analyze_codebook_similarity(model, save_dir)


if __name__ == '__main__':
    main()