import torch
from sklearn.model_selection import RepeatedKFold, StratifiedKFold
import numpy as np
# from model import fNIRS_T, fNIRS_PreT
from model import fNIRS_T
from dataloader import Dataset, Load_Dataset_A, Load_Dataset_B, Load_Dataset_C
import os
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_curve, auc, confusion_matrix

class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing."""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha == None:
            self.alpha = torch.tensor([1.0])
        else:
            self.alpha = torch.tensor(alpha)
            
        # 将 alpha 移动到指定设备
        self.alpha = self.alpha.to("cuda:2")
 
    def forward(self, predict, label):
        bce_loss = torch.nn.functional.cross_entropy(predict, label, reduction="none")
        pt = torch.exp(-bce_loss)
        alpha = self.alpha.gather(0, label.data.view(-1))
        loss = alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()

        
class MultiClassFocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (list, optional): Class-specific weights to handle imbalance. Default is None (equal weights).
            gamma (float, optional): Focusing parameter to down-weight well-classified examples. Default is 2.0.
            reduction (str, optional): Specifies the reduction to apply to the output ('mean', 'sum', or 'none').
        """
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model outputs (shape: [N, C]).
            targets (torch.Tensor): Ground truth class indices (shape: [N]).
        Returns:
            torch.Tensor: Computed focal loss.
        """
        # Apply softmax to get class probabilities
        probs = torch.nn.functional.softmax(logits, dim=1)
        # Gather probabilities of the true class for each sample
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=probs.size(1)).float()
        probs_true = (probs * targets_one_hot).sum(dim=1)

        # Compute the focal weight
        focal_weight = (1 - probs_true) ** self.gamma

        # Compute the log-probabilities
        log_probs = torch.log(probs_true)

        # Apply class-specific weights
        if self.alpha is not None:
            alpha_tensor = torch.tensor(self.alpha, device=logits.device)
            class_weights = (targets_one_hot * alpha_tensor).sum(dim=1)
            loss = -class_weights * focal_weight * log_probs
        else:
            loss = -focal_weight * log_probs

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CombinedLoss(torch.nn.Module):
    def __init__(self, focal_alpha=0.8, focal_gamma=2, smoothing=0.1, reduction='mean'):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)
        self.label_smoothing = LabelSmoothing(smoothing=smoothing)

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        label_smooth_loss = self.label_smoothing(inputs, targets)
        
        # 可以调节这两部分损失的权重
        combined_loss = 0.8 * focal_loss + 0.2 * label_smooth_loss
        return combined_loss


class CombinedMultifocalLoss(torch.nn.Module):
    def __init__(self,  focal_gamma=2, smoothing=0.05, reduction='mean'):
        super(CombinedMultifocalLoss, self).__init__()
        self.focal_loss = MultiClassFocalLoss(alpha=[2,2,2], gamma=focal_gamma, reduction=reduction)
        self.label_smoothing = LabelSmoothing(smoothing=smoothing)

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        label_smooth_loss = self.label_smoothing(inputs, targets)
        
        # 可以调节这两部分损失的权重
        combined_loss =  2 * focal_loss + 0.1 * label_smooth_loss
        return combined_loss


def set_seed(seed=2025):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    set_seed(2025)

    # Training epochs
    EPOCH = 120

    # Select dataset
    dataset = ['A', 'B', 'C', 'D']
    dataset_id = 3
    print(dataset[dataset_id])

    # Select model
    models = ['fNIRS-T', 'fNIRS-PreT']
    models_id = 0
    print(models[models_id])

    # Select the specified path
    data_path = '/data0/zxj_data/predata'

    # Generate unique save path with timestamp and experiment config
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_config = f"{models[models_id]}_dim{64 if dataset[dataset_id] in ['A', 'B'] else 128}"
    save_path = '/data1/zxj_log/save/' + dataset[dataset_id] + f'/KFold/{timestamp}_{exp_config}'
    os.makedirs(save_path, exist_ok=True)

    # Load dataset and set flooding levels. Different models may have different flooding levels.
    if dataset[dataset_id] == 'A':
        flooding_level = [0, 0, 0]
        if models[models_id] == 'fNIRS-T':
            feature, label = Load_Dataset_A(data_path, model='fNIRS-T')
        elif models[models_id] == 'fNIRS-PreT':
            feature, label = Load_Dataset_A(data_path, model='fNIRS-PreT')
    elif dataset[dataset_id] == 'B':
        if models[models_id] == 'fNIRS-T':
            flooding_level = [0.45, 0.40, 0.35]
        else:
            flooding_level = [0.40, 0.38, 0.35]
        feature, label = Load_Dataset_B(data_path)
    elif dataset[dataset_id] == 'C':
        flooding_level = [0.45, 0.40, 0.35]
        feature, label = Load_Dataset_C(data_path)
    elif dataset[dataset_id] == 'D':
        flooding_level = [0.45, 0.40, 0.35]
        # feature, label = Load_Dataset_D()
        
        feature = np.load('/data0/zxj_data/total/feature/concatenated_data.npy',allow_pickle=True)
        # feature[:, 0, :, :] = feature[:, 1, :, :]
        print('feature shape:',feature.shape)
        # feature = np.load('/data0/zxj_data/fNIRS/VFT/feature/features_augmented.npy')
        # feature = feature[:, :, :20, 500:756]
        # print('feature.shape:',feature.shape)
        
        label = np.load('/data0/zxj_data/total/label/label_squeezed.npy',allow_pickle=True)
        print('label :',label)

        # print('label.shape:',label.shape)
        # 假设原始 feature 和 label 的形状如下
        # feature = np.load('/data0/zxj_data/fNIRS/VFT/feature/features_augmented.npy')  # (999, 2, 53, 2048)
        # label = np.load('/data0/zxj_data/fNIRS/VFT/feature/labels_combined_shuffled.npy')  # (999,)

        # # 截取特征维度 2048 为 256 的块大小
        # block_size = 256
        # num_blocks = feature.shape[-1] // block_size  # 2048 // 256 = 8

        # # 验证可以整除
        # assert feature.shape[-1] % block_size == 0, "2048 不能被 block_size 整除！"

        # # 重新生成 feature
        # new_features = feature.reshape(feature.shape[0], feature.shape[1], feature.shape[2], num_blocks, block_size)
        # new_features = new_features.transpose(0, 3, 1, 2, 4)  # 调整维度顺序：把 num_blocks 提到样本维度
        # new_features = new_features.reshape(-1, feature.shape[1], feature.shape[2], block_size)  # 展平样本维度

        # # 重新生成 label
        # new_labels = np.repeat(label, num_blocks)

        # # 打印新特征和标签的形状
        # print("Original feature shape:", feature.shape)  # (999, 2, 53, 2048)
        # print("Original label shape:", label.shape)      # (999,)
        # print("New feature shape:", new_features.shape)  # (7992, 2, 53, 256)
        # print("New label shape:", new_labels.shape)      # (7992,)

    # feature = new_features
    # label = new_labels
    _, _, channels, sampling_points = feature.shape

    feature = feature.reshape((label.shape[0], -1))
    # 5 × 5-fold-CV

    indices = np.arange(len(feature))
    rng = np.random.default_rng(2025)  # 固定 NumPy 生成器
    rng.shuffle(indices)

    # indices = np.arange(len(feature))
    # np.random.shuffle(indices)
    feature = feature[indices]
    label = label[indices]

    # rkf = RepeatedKFold(n_splits=4, n_repeats=4, random_state=2025)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=2025)

    # 记录每个 epoch 的训练和测试信息
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # 创建一个统一的文件来保存所有折的预测结果
    all_folds_predictions_file = os.path.join(save_path, 'all_folds_predictions.txt')
    with open(all_folds_predictions_file, 'w') as all_pred_file:
        all_pred_file.write(f"所有折的预测结果详情\n\n")
        all_pred_file.write(f"实验配置: {exp_config}\n")
        all_pred_file.write(f"时间戳: {timestamp}\n\n")
    
    # 创建一个统一的文件来保存所有折的预测结果
    all_folds_predictions_file = os.path.join(save_path, 'all_folds_predictions.txt')
    with open(all_folds_predictions_file, 'w') as all_pred_file:
        all_pred_file.write(f"所有折的预测结果详情\n\n")
        all_pred_file.write(f"实验配置: {exp_config}\n")
        all_pred_file.write(f"时间戳: {timestamp}\n\n")

    n_runs = 0
    # for train_index, test_index in skf.split(feature):
    for fold, (train_index, test_index) in enumerate(skf.split(feature, label)):
        n_runs += 1
        print('======================================\n', n_runs)
        path = save_path + '/' + str(n_runs)
        assert os.path.exists(path) is False, 'sub-path is exist'
        os.makedirs(path)

        X_train = feature[train_index]
        print('X_train.shape:',X_train.shape)
        y_train = label[train_index]
        print('y_train.shape:',y_train.shape)
        X_test = feature[test_index]
        print('X_test.shape:',X_test.shape)
        y_test = label[test_index]
        print('y_test.shape:',y_test.shape)

        X_train = X_train.reshape((X_train.shape[0], 2, channels, -1))
        print('X_train.shape:',X_train.shape)
        X_test = X_test.reshape((X_test.shape[0], 2, channels, -1))
        print('X_test.shape:',X_test.shape)

        # 论文数据
        # tq = 256
        # gamma = 20

        # tq = 2400  # Total time duration for each question
        # gamma = 200  # Upper bound for masking/warping duration
        #不增强
        # X_train_augmented = X_trainß

        # X_train_augmented = apply_time_augmentations(X_train, tq, gamma)
        X_train_augmented = X_train

        train_set = Dataset(X_train_augmented, y_train, transform=True)
        test_set = Dataset(X_test, y_test, transform=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(2025))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(2025))

        # -------------------------------------------------------------------------------------------------------------------- #
        device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
        if dataset[dataset_id] == 'A':
            if models[models_id] == 'fNIRS-T':
                net = fNIRS_T(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
            elif models[models_id] == 'fNIRS-PreT':
                net = fNIRS_PreT(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
        elif dataset[dataset_id] == 'B':
            if models[models_id] == 'fNIRS-T':
                net = fNIRS_T(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
            elif models[models_id] == 'fNIRS-PreT':
                net = fNIRS_PreT(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
        elif dataset[dataset_id] == 'C':
            if models[models_id] == 'fNIRS-T':
                net = fNIRS_T(n_class=3, sampling_point=sampling_points, dim=128, depth=6, heads=8, mlp_dim=64).to(device)
            elif models[models_id] == 'fNIRS-PreT':
                net = fNIRS_PreT(n_class=3, sampling_point=sampling_points, dim=128, depth=6, heads=8, mlp_dim=64).to(device)
        elif dataset[dataset_id] == 'D':
            if models[models_id] == 'fNIRS-T':
                net = fNIRS_T(n_class=2, sampling_point=sampling_points, dim = 128, depth=6, heads=8, mlp_dim=64).to(device)
        

        criterion = LabelSmoothing(0.1)  # 使用标签平滑，平滑系数为0.1
        # criterion = torch.nn.CrossEntropyLoss()  # 数据平衡时
        
        # 基于样本数量的比例
        # total_samples = 79 + 17
        # alpha_0 = 1 - (79 / total_samples)
        # alpha_1 = 1 - (17 / total_samples)
        # total = alpha_0 + alpha_1
        # alpha_0 = alpha_0 / total
        # alpha_1 = alpha_1 / total

        
        # criterion = FocalLoss(alpha=[alpha_0, alpha_1])
        # 使用 CombinedLoss
        # criterion = CombinedLoss(focal_alpha=1, focal_gamma=2, smoothing=0.1, reduction='mean')
        #多分类的focal loss
        # 重新定义 alpha 权重
        # class_ratios = [1, 3, 3, 3, 4]
        # total = sum(class_ratios)
        # alpha = [1 / ratio for ratio in class_ratios]  # 倒数计算
        # alpha = [a / sum(alpha) for a in alpha]       # 归一化
        # criterion = MultiClassFocalLoss(alpha=alpha, gamma=2.0, reduction='mean')
        
        
        # criterion = MultiClassFocalLoss(alpha=[0.35, 0.3, 0.3, 0.3, 0.25], gamma=2.0, reduction='mean')
        # criterion = CombinedMultifocalLoss()

        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.1)

        lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        

        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []


        # -------------------------------------------------------------------------------------------------------------------- #
        log_file_path = os.path.join(path, f'training_log_run_{n_runs}.txt')
        with open(log_file_path, 'w') as log_file:
        
            test_max_acc = 0
            best_epoch = -1
            best_correct_indices = []
            best_incorrect_indices = []
            best_true_labels = []
            best_pred_labels = []
            best_sample_indices = []
            for epoch in range(EPOCH):
                net.train()
                train_running_acc = 0
                total = 0
                loss_steps = []

                num_classes = 2
                # 初始化每个类别的统计变量
                train_class_correct = [0] * num_classes  # 每个类别预测正确的样本数
                train_class_total = [0] * num_classes    # 每个类别的总样本数

                for i, data in enumerate(train_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())

                    # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
                    if epoch < 30:
                        b = flooding_level[0]
                    elif epoch < 50:
                        b = flooding_level[1]
                    else:
                        b = flooding_level[2]

                    # flooding
                    loss = (loss - b).abs() + b

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_steps.append(loss.item())
                    total += labels.shape[0]
                    pred = outputs.argmax(dim=1, keepdim=True)
                    train_running_acc += pred.eq(labels.view_as(pred)).sum().item()

                    # 按类别统计
                    for i in range(labels.size(0)):
                        label_int = int(labels[i].item())  # 获取原始标签整数值
                        predicted_label = int(pred[i].item())  # 获取预测标签整数值
                        train_class_correct[label_int] += (predicted_label == label_int)
                        train_class_total[label_int] += 1

                train_running_loss = float(np.mean(loss_steps))
                train_running_acc = 100 * train_running_acc / total
                print('[%d, %d] Train loss: %0.4f' % (n_runs, epoch, train_running_loss))
                print('[%d, %d] Train acc: %0.3f%%' % (n_runs, epoch, train_running_acc))
                log_file.write(f'[{n_runs}, {epoch}] Train loss: {train_running_loss:.4f}\n')
                log_file.write(f'[{n_runs}, {epoch}] Train acc: {train_running_acc:.3f}%\n')


                # 打印每个类别的训练准确率
                print('     Training Class-wise accuracy:')
                for i in range(num_classes):
                    if train_class_total[i] > 0:
                        acc = 100.0 * train_class_correct[i] / train_class_total[i]
                        print(f'     Class {i}: {acc:.2f}% ({train_class_correct[i]}/{train_class_total[i]})')
                        log_file.write(f'     Class {i}: {acc:.2f}% ({train_class_correct[i]}/{train_class_total[i]})\n')
                    else:
                        print(f'     Class {i}: No samples')
                        log_file.write(f'     Class {i}: No samples\n')


                train_losses.append(train_running_loss)
                train_accuracies.append(train_running_acc)

                # -------------------------------------------------------------------------------------------------------------------- #
                # 初始化每个类别的统计变量和混淆矩阵指标
                class_correct = [0] * num_classes  # 每个类别预测正确的样本数
                class_total = [0] * num_classes    # 每个类别的总样本数
                tp = fp = fn = tn = 0  # 混淆矩阵指标
                all_labels = []  # 用于ROC曲线
                all_probs = []   # 用于ROC曲线
                
                # 用于记录预测结果的列表
                correct_indices = []  # 预测正确的样本索引
                incorrect_indices = []  # 预测错误的样本索引
                true_labels = []  # 真实标签
                pred_labels = []  # 预测标签
                sample_indices = []  # 原始样本索引（在test_index中的位置）
                
                net.eval()
                test_running_acc = 0
                total = 0
                loss_steps = []
                with torch.no_grad():
                    batch_idx = 0
                    for data in test_loader:
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())

                        loss_steps.append(loss.item())
                        total += labels.shape[0]
                        pred = outputs.argmax(dim=1, keepdim=True)
                        test_running_acc += pred.eq(labels.view_as(pred)).sum().item()

                        # 保存标签和预测概率用于ROC曲线（二分类问题）
                        if num_classes == 2:
                            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
                            all_probs.extend(probs)
                            all_labels.extend(labels.cpu().numpy())
                        
                        # 按类别统计和累积混淆矩阵
                        for i in range(labels.size(0)):
                            # 计算当前样本在测试集中的索引
                            sample_idx = batch_idx * test_loader.batch_size + i
                            if sample_idx < len(test_index):  # 确保不超出测试集索引范围
                                original_idx = test_index[sample_idx]  # 获取原始数据集中的索引
                                
                                label_int = int(labels[i].item())  # 获取原始标签整数值
                                predicted_label = int(pred[i].item())  # 获取预测标签整数值
                                
                                # 记录预测结果
                                sample_indices.append(original_idx)
                                true_labels.append(label_int)
                                pred_labels.append(predicted_label)
                                
                                if predicted_label == label_int:
                                    correct_indices.append(original_idx)
                                else:
                                    incorrect_indices.append(original_idx)
                                
                                class_correct[label_int] += (predicted_label == label_int)
                                class_total[label_int] += 1
                                
                                # 累积统计混淆矩阵（二分类问题）
                                if num_classes == 2:
                                    if predicted_label == 1 and label_int == 1:
                                        tp += 1
                                    elif predicted_label == 1 and label_int == 0:
                                        fp += 1
                                    elif predicted_label == 0 and label_int == 1:
                                        fn += 1
                                    else:  # predicted_label == 0 and label_int == 0
                                        tn += 1
                        
                        batch_idx += 1
                
                # 计算评估指标
                test_running_acc = 100 * test_running_acc / total
                test_running_loss = float(np.mean(loss_steps))
                
                # 计算精确率和召回率
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # 计算敏感性(Sensitivity)和特异性(Specificity)
                sensitivity = recall  # 敏感性就是召回率
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性
                
                # 计算F1分数
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f'     Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}')
                print(f'     Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}')
                print(f'     Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}')
                log_file.write(f'     Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}\n')
                log_file.write(f'     Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}\n')
                
                # 绘制ROC曲线（仅适用于二分类问题）
                if num_classes == 2 and len(all_labels) > 0:
                    fpr, tpr, _ = roc_curve(all_labels, all_probs)
                    roc_auc = auc(fpr, tpr)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - Run {n_runs}, Epoch {epoch}')
                    plt.legend(loc="lower right")
                    plt.savefig(os.path.join(path, f'run_{n_runs}_epoch_{epoch}_roc_curve.png'))
                    plt.close()

                # 打印每个类别的准确率
                log_file.write('     test Class-wise accuracy:\n')
                print('     test Class-wise accuracy:')
                for i in range(num_classes):
                    if class_total[i] > 0:
                        acc = 100.0 * class_correct[i] / class_total[i]
                        print(f'     Class {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})')
                    else:
                        print(f'     Class {i}: No samples')

                # 保存到文件（可选）
                with open(os.path.join(path, f'classwise_accuracy_run_{n_runs}.txt'), 'w') as f:
                    f.write(f'Test Accuracy: {test_running_acc:.3f}%\n')
                    f.write('Class-wise accuracy:\n')
                    for i in range(num_classes):
                        if class_total[i] > 0:
                            acc = 100.0 * class_correct[i] / class_total[i]
                            log_file.write(f'     Class {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})\n')
                        else:
                            log_file.write(f'     Class {i}: No samples\n')


                print('     [%d, %d] Test loss: %0.4f' % (n_runs, epoch, test_running_loss))
                print('     [%d, %d] Test acc: %0.3f%%' % (n_runs, epoch, test_running_acc))
                print('     [%d, %d] Test F1: %0.3f (P: %0.3f, R: %0.3f)' % (n_runs, epoch, f1_score, precision, recall))
                log_file.write(f'     [{n_runs}, {epoch}] Test loss: {test_running_loss:.4f}\n')
                log_file.write(f'     [{n_runs}, {epoch}] Test acc: {test_running_acc:.3f}%\n')
                log_file.write(f'     [{n_runs}, {epoch}] Test F1: {f1_score:.3f} (P: {precision:.3f}, R: {recall:.3f})\n')
                test_losses.append(test_running_loss)
                test_accuracies.append(test_running_acc)

                if f1_score > test_max_acc:  # 使用F1分数作为保存模型的指标
                    test_max_acc = f1_score
                    best_epoch = epoch
                    # 保存当前最佳epoch的预测结果
                    best_correct_indices = correct_indices.copy()
                    best_incorrect_indices = incorrect_indices.copy()
                    best_true_labels = true_labels.copy()
                    best_pred_labels = pred_labels.copy()
                    best_sample_indices = sample_indices.copy()
                    
                    torch.save(net.state_dict(), path + '/model.pt')
                    with open(path + '/test_metrics.txt', "w") as test_save:
                        test_save.write(f"F1 Score: {f1_score:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nAccuracy: {test_running_acc:.3f}%\nSensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}")
                    
                    # 如果是二分类问题，保存ROC曲线的AUC值
                    if num_classes == 2 and len(all_labels) > 0:
                        with open(path + '/roc_auc.txt', "w") as roc_file:
                            roc_file.write(f"ROC AUC: {roc_auc:.3f}\n")
                    
                    # 保存模型结构信息
                    with open(path + '/model_structure.txt', "w") as model_structure_file:
                        model_structure_file.write(f"7 regions, multiscale 50\n")
                    
                    # 保存当前最佳epoch的预测结果
                    best_correct_indices = correct_indices.copy()
                    best_incorrect_indices = incorrect_indices.copy()
                    best_true_labels = true_labels.copy()
                    best_pred_labels = pred_labels.copy()
                    best_sample_indices = sample_indices.copy()
                    best_epoch = epoch
                    
                    # 保存预测结果详情到当前折的文件
                    with open(path + '/prediction_details.txt', "w") as pred_file:
                        pred_file.write(f"折数: {n_runs}, 最佳Epoch: {epoch}\n\n")
                        
                        # 保存正确预测的样本信息
                        pred_file.write(f"正确预测的样本数: {len(correct_indices)}\n")
                        pred_file.write("样本序号\t真实标签\t预测标签\n")
                        for idx, true_label, pred_label in zip(sample_indices, true_labels, pred_labels):
                            if idx in correct_indices:
                                pred_file.write(f"{idx}\t{true_label}\t{pred_label}\n")
                        
                        # 保存错误预测的样本信息
                        pred_file.write(f"\n错误预测的样本数: {len(incorrect_indices)}\n")
                        pred_file.write("样本序号\t真实标签\t预测标签\n")
                        for idx, true_label, pred_label in zip(sample_indices, true_labels, pred_labels):
                            if idx in incorrect_indices:
                                pred_file.write(f"{idx}\t{true_label}\t{pred_label}\n")
                        
                        # 保存所有样本的预测结果
                        pred_file.write(f"\n所有样本的预测结果:\n")
                        pred_file.write("样本序号\t真实标签\t预测标签\t预测结果\n")
                        for idx, true_label, pred_label in zip(sample_indices, true_labels, pred_labels):
                            result = "正确" if true_label == pred_label else "错误"
                            pred_file.write(f"{idx}\t{true_label}\t{pred_label}\t{result}\n")
                    
                    # 同时将结果追加到统一的预测结果文件中
                    with open(all_folds_predictions_file, 'a') as all_pred_file:
                        all_pred_file.write(f"============ 折数: {n_runs}, 最佳Epoch: {epoch} ============\n\n")
                        
                        # 追加正确预测的样本信息
                        all_pred_file.write(f"正确预测的样本数: {len(correct_indices)}\n")
                        all_pred_file.write("样本序号\t真实标签\t预测标签\n")
                        for idx, true_label, pred_label in zip(sample_indices, true_labels, pred_labels):
                            if idx in correct_indices:
                                all_pred_file.write(f"{idx}\t{true_label}\t{pred_label}\n")
                        
                        # 追加错误预测的样本信息
                        all_pred_file.write(f"\n错误预测的样本数: {len(incorrect_indices)}\n")
                        all_pred_file.write("样本序号\t真实标签\t预测标签\n")
                        for idx, true_label, pred_label in zip(sample_indices, true_labels, pred_labels):
                            if idx in incorrect_indices:
                                all_pred_file.write(f"{idx}\t{true_label}\t{pred_label}\n")
                        
                        # 追加所有样本的预测结果
                        all_pred_file.write(f"\n所有样本的预测结果:\n")
                        all_pred_file.write("样本序号\t真实标签\t预测标签\t预测结果\n")
                        for idx, true_label, pred_label in zip(sample_indices, true_labels, pred_labels):
                            result = "正确" if true_label == pred_label else "错误"
                            all_pred_file.write(f"{idx}\t{true_label}\t{pred_label}\t{result}\n")
                        
                        all_pred_file.write("\n\n")

                lrStep.step()
            
            # 在每一折训练结束后，将该折最佳epoch的预测结果追加到统一文件中
            with open(all_folds_predictions_file, 'a') as all_pred_file:
                all_pred_file.write(f"============ 折数: {n_runs}, 最佳Epoch: {best_epoch} ============\n\n")
                
                # 追加正确预测的样本信息
                all_pred_file.write(f"正确预测的样本数: {len(best_correct_indices)}\n")
                all_pred_file.write("样本序号\t真实标签\t预测标签\n")
                for idx, true_label, pred_label in zip(best_sample_indices, best_true_labels, best_pred_labels):
                    if idx in best_correct_indices:
                        all_pred_file.write(f"{idx}\t{true_label}\t{pred_label}\n")
                
                # 追加错误预测的样本信息
                all_pred_file.write(f"\n错误预测的样本数: {len(best_incorrect_indices)}\n")
                all_pred_file.write("样本序号\t真实标签\t预测标签\n")
                for idx, true_label, pred_label in zip(best_sample_indices, best_true_labels, best_pred_labels):
                    if idx in best_incorrect_indices:
                        all_pred_file.write(f"{idx}\t{true_label}\t{pred_label}\n")
                
                # 追加所有样本的预测结果
                all_pred_file.write(f"\n所有样本的预测结果:\n")
                all_pred_file.write("样本序号\t真实标签\t预测标签\t预测结果\n")
                for idx, true_label, pred_label in zip(best_sample_indices, best_true_labels, best_pred_labels):
                    result = "正确" if true_label == pred_label else "错误"
                    all_pred_file.write(f"{idx}\t{true_label}\t{pred_label}\t{result}\n")
                
                all_pred_file.write("\n\n")

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(test_losses, label='Test Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve for Run {n_runs}')
        plt.legend()
        plt.savefig(os.path.join(path, f'run_{n_runs}_loss_curve.png'))  # 保存 Loss 图表
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracies, label='Train Accuracy', color='blue')
        plt.plot(test_accuracies, label='Test Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy Curve for Run {n_runs}')
        plt.legend()
        plt.savefig(os.path.join(path, f'run_{n_runs}_accuracy_curve.png'))  # 保存 Accuracy 图表
        plt.close()
        
        # 生成可解释性图片
        try:
            # 使用测试集的一个批次来获取可解释性特征
            net.eval()
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    _ = net(inputs)  # 运行前向传播以获取可解释性特征
                    break
            
            # 获取可解释性特征
            if hasattr(net, 'get_interpretable_features'):
                interpretable_features = net.get_interpretable_features()
                
                # 绘制特征重要性图
                plt.figure(figsize=(10, 6))
                # 提取特征名称和对应的重要性值
                feature_names = interpretable_features['feature_names']
                # 只选择与feature_names长度匹配的特征重要性值
                importance_values = []
                for name in feature_names:
                    # 查找包含该特征名称的键的平均值
                    matching_values = [v for k, v in interpretable_features['feature_importance'].items() if name in k]
                    if matching_values:
                        importance_values.append(sum(matching_values) / len(matching_values))
                    else:
                        importance_values.append(0)  # 如果没有匹配的值，则设为0
                
                # 确保两个数组长度相同
                assert len(feature_names) == len(importance_values), f"形状不匹配: feature_names长度为{len(feature_names)}，importance_values长度为{len(importance_values)}"
                
                plt.bar(feature_names, importance_values)
                plt.title('Feature Importance')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.xticks(rotation=45)
                plt.tight_layout() 
                plt.savefig(os.path.join(save_path, f'feature_importance_fold_{fold}.jpg'))
                plt.close()
                
                # 绘制注意力模式图
                for layer, attn_weights in interpretable_features['attention_patterns'].items():
                    plt.figure(figsize=(10, 6))
                    plt.imshow(attn_weights[0].cpu().numpy(), cmap='viridis')
                    plt.title(f'Attention Pattern - {layer}')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, f'attention_pattern_{layer}_fold_{fold}.jpg'))
                    plt.close()
                    
                    # 绘制区域连接热图 - 只关注前7个token（区域令牌）
                    plt.figure(figsize=(8, 6))
                    region_attn = attn_weights[:7, :7].cpu().numpy()
                    plt.imshow(region_attn, cmap='hot')
                    plt.colorbar()
                    plt.title(f'Layer {layer} Region Connectivity')
                    plt.savefig(os.path.join(path, f'run_{n_runs}_layer_{layer}_region_connectivity.png'))
                    plt.close()
                    
                    # 绘制通道重要性热图 - 区域令牌对各通道的注意力
                    plt.figure(figsize=(12, 6))
                    channel_importance = attn_weights[:7, 7:].cpu().numpy()
                    plt.imshow(channel_importance, cmap='coolwarm')
                    plt.colorbar()
                    plt.title(f'Layer {layer} Channel Importance')
                    plt.xlabel('Channels')
                    plt.ylabel('Brain Regions')
                    plt.savefig(os.path.join(path, f'run_{n_runs}_layer_{layer}_channel_importance.png'))
                    plt.close()

        except Exception as e:
            print(f"生成可解释性图片时出错: {e}")
            with open(os.path.join(path, 'error_log.txt'), 'a') as error_log:
                error_log.write(f"生成可解释性图片时出错: {e}\n")

try:
    interpretable_features = net.get_interpretable_features()
    # 创建一个可视化摘要图，而不是直接将字典传递给imshow
    plt.figure(figsize=(10, 6))
    
    # 从字典中提取一个注意力权重矩阵用于可视化
    # 选择第一个注意力模式作为示例
    if interpretable_features['attention_patterns']:
        # 获取第一个注意力层的权重
        first_layer_key = list(interpretable_features['attention_patterns'].keys())[0]
        attn_weights = interpretable_features['attention_patterns'][first_layer_key]
        
        # 确保我们只使用第一个批次的第一个头的注意力权重
        # 转换为2D numpy数组以便可视化
        attn_matrix = attn_weights[0].cpu().numpy()
        
        # 绘制注意力矩阵
        plt.imshow(attn_matrix, cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Weights - {first_layer_key}')
        plt.savefig(os.path.join(path, 'interpretable_features.jpg'))
    else:
        print("没有可用的注意力模式进行可视化")
    plt.close()
except Exception as e:
    print(f"生成解释性特征图片时出错: {e}")
    with open(os.path.join(path, 'error_log.txt'), 'a') as error_log:
        error_log.write(f"生成解释性特征图片时出错: {e}\n")
plt.close()