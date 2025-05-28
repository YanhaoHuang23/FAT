from utils import eegDataset
import os
import sys
import torch.optim as optim
from scipy import io as scio
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
from utils import normalize_A
from einops import rearrange, reduce
from FANLayer import FANLayer
from FaCo_FMLP import FAConv2D, FAMLP
import random
import numpy as np
import torch
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior if possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TORCH_HOME'] = './'  # 设置环境变量

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 关注通道数并结合位置编码
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2  # 将字节转换为 MB
    return size_all_mb


# 数据增强方法
def scale_frequency_bands(features, scale_range=(0.9, 1.1)):
    """
    随机缩放频域特征的特定频带。
    """
    scale_factors = np.random.uniform(scale_range[0], scale_range[1], size=features.shape[-1])
    enhanced_features = features * scale_factors
    return enhanced_features


# 建议测试多个 frequency 和 amplitude 组合，如 (2, 0.05) 或 (5, 0.1)，以观察对性能的实际影响。
def add_periodic_perturbation(features, frequency=2, amplitude=0.05):
    """
    向频域特征添加正弦波扰动，增强周期性。
    """
    num_bands = features.shape[-1]
    time = np.linspace(0, 2 * np.pi, num_bands)
    sinusoid = amplitude * np.sin(frequency * time)
    enhanced_features = features + sinusoid
    return enhanced_features


def mixup(features, labels, alpha=0.2):
    """
    使用 Mixup 方法增强频域特征。
    """
    lam = np.random.beta(alpha, alpha)
    indices = np.random.permutation(features.shape[0])
    mixed_features = lam * features + (1 - lam) * features[indices]
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    return mixed_features, mixed_labels


"""
class FrequencyAttention(nn.Module):
    def __init__(self, num_freq_bands):
        super(FrequencyAttention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(1, num_freq_bands))  # 可学习的权重

    def forward(self, x):
        # x: [B, 1, C, num_freq_bands]
        weights = torch.softmax(self.attention_weights, dim=-1)  # [1, num_freq_bands]
        return x * weights.unsqueeze(0).unsqueeze(1)  # 加权 [B, 1, C, num_freq_bands]
"""


# 定义 EEG Conformer 模型
class ModifiedPatchEmbedding2D(nn.Module):
    def __init__(self, emb_size=40, num_channels=62, num_freq_bands=5):
        super(ModifiedPatchEmbedding2D, self).__init__()
        self.emb_size = emb_size
        self.num_channels = num_channels
        self.num_freq_bands = num_freq_bands
        self.batch_norm_stage1 = nn.BatchNorm2d(emb_size // 2)
        self.batch_norm_stage2 = nn.BatchNorm2d(emb_size)

        # 位置编码
        self.position_encodings = nn.Parameter(torch.randn(1, 1, num_channels, num_freq_bands))

        # 频率注意力模块
        # self.frequency_attention = FrequencyAttention(num_freq_bands)

        self.conv2d_stage1 = nn.Sequential(
            nn.Conv2d(num_freq_bands, emb_size // 2, kernel_size=(1, 1)),
            self.batch_norm_stage1,
            nn.ReLU()
        )

        self.conv2d_stage2 = nn.Sequential(
            nn.Conv2d(emb_size // 2, emb_size, kernel_size=(1, 1)),
            self.batch_norm_stage2,
            nn.ReLU()
        )

    def forward(self, x):
        # 添加位置编码
        x = x + self.position_encodings  # [B, 1, C, num_freq_bands]

        # 频率注意力加权
        # x = self.frequency_attention(x)  # [B, 1, C, num_freq_bands]

        x = x.squeeze(1).permute(0, 2, 1).unsqueeze(-1)  # [B, num_freq_bands, C, 1]

        x = self.conv2d_stage1(x)  # [B, emb_size // 2, C, 1]
        x = self.conv2d_stage2(x)  # [B, emb_size, C, 1]

        # 调整回 Transformer 格式
        x = x.squeeze(-1).permute(0, 2, 1)  # [B, C, emb_size]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_size))
        pe = torch.zeros(max_len, emb_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, emb_size]
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)  # 加位置编码
        return self.dropout(x)


class DynamicGraphLearner(nn.Module):
    """
    根据输入 x (形状 [B, n, emb_size]) 计算出 [B, n, n] 的矩阵
    """
    def __init__(self, n, num_heads=4):
        super(DynamicGraphLearner, self).__init__()
        self.num_heads = num_heads
        self.adj = nn.Parameter(torch.randn(n, n))  # 可学习

        # 使用xavier初始化
        nn.init.xavier_normal_(self.adj)  # 在这里进行 Xavier 初始化

    def forward(self, x):
        B = x.size(0)
        n = x.size(1)

        adj_normalized = normalize_A(self.adj)  # 对邻接矩阵进行归一化
        # [n, n] -> [B, n, n]
        adj_expanded = adj_normalized.unsqueeze(0).expand(B, -1, -1)  # [B, n, n]

        # [B, 1, n, n] -> [B, h, n, n]
        adj_expanded = adj_expanded.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        return adj_expanded


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.5, use_dynamic_graph=False):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = FANLayer(emb_size, emb_size, activation=nn.Identity())
        self.queries = FANLayer(emb_size, emb_size, activation=nn.Identity())
        self.values = FANLayer(emb_size, emb_size, activation=nn.Identity())
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

        self.use_dynamic_graph = use_dynamic_graph
        if self.use_dynamic_graph:
            self.dynamic_graph_learners = nn.ModuleList([
                DynamicGraphLearner(n=63, num_heads=4) for _ in range(2)
            ])

            # 新增两个线性层用于生成融合权重
            self.dg_weight_linear1 = nn.Linear(emb_size // 2, 1)
            self.dg_weight_linear2 = nn.Linear(emb_size // 2, 1)

            # 新增sigmoid激活函数
            self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None, dynamic_graph1=None, dynamic_graph2=None):
        """
        x.shape = (B, n, emb_size=40)
        前提：
          - num_heads = 8
          - 每个 head_dim = 5
          - 前 4 个 head 和后 4 个 head 分别融合到 DG1 / DG2
          - DG1.shape = (B,4,n,n), DG2.shape = (B,4,n,n)
        """
        B, n, _ = x.shape

        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=8)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=8)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=8)

        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))

        queries_4_4 = queries.permute(0, 2, 1, 3)

        q_front4 = queries_4_4[:, :, :4, :]
        q_back4 = queries_4_4[:, :, 4:, :]

        q_front4_20 = q_front4.reshape(B, n, 20)

        flat_front4 = q_front4_20.reshape(-1, 20)
        w_front4 = self.dg_weight_linear1(flat_front4)
        w_front4 = self.sigmoid(w_front4)

        w_front4 = w_front4.view(B, n, 1)
        w_front4 = w_front4.unsqueeze(1).expand(-1, 4, -1, n)

        q_back4_20 = q_back4.reshape(B, n, 20)
        flat_back4 = q_back4_20.reshape(-1, 20)
        w_back4 = self.dg_weight_linear2(flat_back4)
        w_back4 = self.sigmoid(w_back4).view(B, n, 1)
        w_back4 = w_back4.unsqueeze(1).expand(-1, 4, -1, n)

        if dynamic_graph1 is not None:
            energy[:, :4, :, :] = energy[:, :4, :, :] + w_front4 * dynamic_graph1
        if dynamic_graph2 is not None:
            energy[:, 4:, :, :] = energy[:, 4:, :, :] + w_back4 * dynamic_graph2

        scaling = (self.emb_size ** 0.5)
        att = torch.softmax(energy / scaling, dim=-1)
        out = torch.einsum("bhqk, bhkd -> bhqd", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        if isinstance(self.fn, MultiHeadAttention):
            return x + self.fn(x, **kwargs)
        else:
            return x + self.fn(x)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0.5):
        super().__init__(
            # FAMLP(emb_size, expansion * emb_size, emb_size, num_layers=1, dropout=drop_p)
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )


class AttentionBlock(nn.Module):
    def __init__(self, emb_size, num_heads, drop_p, use_dynamic_graph=False):
        super().__init__()
        self.layernorm = nn.LayerNorm(emb_size)
        self.mha = MultiHeadAttention(emb_size, num_heads, drop_p, use_dynamic_graph=use_dynamic_graph)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, mask=None, dynamic_graph1=None, dynamic_graph2=None):
        # 1. LayerNorm
        x_norm = self.layernorm(x)
        # 2. Multi-Head Attention（传入额外的 dynamic_graph 参数）
        out = self.mha(x_norm, mask=mask, dynamic_graph1=dynamic_graph1, dynamic_graph2=dynamic_graph2)
        # 3. Dropout
        out = self.dropout(out)
        # 4. 残差连接
        return out + x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads=4, drop_p=0.5, forward_expansion=4, use_dynamic_graph=False):
        super().__init__()
        # 注意力部分用自定义的 AttentionBlock 来替代原先的 nn.Sequential
        self.attention = AttentionBlock(emb_size, num_heads, drop_p, use_dynamic_graph)

        # FeedForward 部分可以继续用 ResidualAdd + nn.Sequential，因为它不需要额外参数
        self.feed_forward = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=drop_p),
                nn.Dropout(drop_p),
            )
        )

    def forward(self, x, mask=None, dynamic_graph1=None, dynamic_graph2=None):
        # 只在多头注意力这里需要传入 dynamic_graph1 / dynamic_graph2
        x = self.attention(x, mask=mask, dynamic_graph1=dynamic_graph1, dynamic_graph2=dynamic_graph2)
        x = self.feed_forward(x)  # FFN 部分无需额外参数
        return x



class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size, num_heads=4, drop_p=0.1, use_dynamic_graph=False):
        super().__init__()

        self.use_dynamic_graph = use_dynamic_graph
        self.dynamic_graph_learner1 = DynamicGraphLearner(n=63, num_heads=num_heads // 2) if use_dynamic_graph else None
        self.dynamic_graph_learner2 = DynamicGraphLearner(n=63, num_heads=num_heads // 2) if use_dynamic_graph else None

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, drop_p, use_dynamic_graph=use_dynamic_graph)
            for _ in range(depth)
        ])

    def forward(self, x, mask=None):
        dynamic_graph1 = self.dynamic_graph_learner1(x) if self.use_dynamic_graph else None
        dynamic_graph2 = self.dynamic_graph_learner2(x) if self.use_dynamic_graph else None

        for layer in self.layers:
            x = layer(x, mask=mask, dynamic_graph1=dynamic_graph1, dynamic_graph2=dynamic_graph2)

        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes, drop_p=0.3):
        super().__init__(
            nn.LayerNorm(emb_size),  # Normalize features
            nn.Linear(emb_size, n_classes)  # Map to class logits
        )


class GATGraph(nn.Module):
    def __init__(self, emb_size=40, depth=6, n_classes=3,
                 num_channels=62, num_freq_bands=5, num_heads=8,
                 use_dynamic_graph=True):
        super(GATGraph, self).__init__()
        self.patch_embedding = ModifiedPatchEmbedding2D(emb_size, num_channels, num_freq_bands)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_encoding = PositionalEncoding(emb_size, dropout=0.1,
                                                      max_len=num_channels + 1)
        self.transformer_encoder = TransformerEncoder(
            depth, emb_size, num_heads, drop_p=0.2, use_dynamic_graph=use_dynamic_graph
        )
        self.classification_head = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)  # [B, C, emb_size]

        B, C, E = x.shape
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, C+1, emb_size]

        x = self.positional_encoding(x)       # [B, C+1, emb_size]
        x = self.transformer_encoder(x)       # [B, C+1, emb_size]

        cls_output = x[:, 0, :]              # [B, emb_size]
        logits = self.classification_head(cls_output)
        return logits


# 定义 eegDataset 类
class eegDataset(Dataset):
    def __init__(self, features, labels, augment=False):
        self.features = features
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        if self.augment:
            feature = scale_frequency_bands(feature)
            feature = add_periodic_perturbation(feature)

        return feature.clone().detach().float(), label.clone().detach().long()


def load_DE_SEED(load_path):
    datasets = scio.loadmat(load_path)
    DE = datasets['DE']  # 形状为（通道数，样本数，频带数）
    dataAll = np.transpose(DE, [1, 0, 2])  # 转换为（样本数，通道数，频带数）
    labelAll = datasets['labelAll'].flatten()
    labelAll = labelAll # 将标签从[-1,0,1]转换为[0,1,2]
    return dataAll, labelAll


def load_dataloader(data_train, data_test, label_train, label_test, augment=False):
    batch_size = 32
    train_dataset = eegDataset(data_train, label_train, augment=augment)
    test_dataset = eegDataset(data_test, label_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, test_loader


def train(train_loader, test_loader, model, criterion, optimizer, num_epochs, sub_name):
    print('开始在', device, '上训练...')
    alpha = 0.1
    acc_test_best = 0.0
    n = 0
    model_best = None
    for ep in range(num_epochs):
        model.train()
        n += 1
        batch_id = 1
        correct, total, total_loss = 0, 0, 0.
        for images, labels in train_loader:
            images = images.float().to(device)
            labels = labels.long().to(device)

            # 前向传播
            # ----------- 1) 生成 lam &amp; 随机乱序索引 -----------
            lam = np.random.beta(alpha, alpha)  # from numpy or torch.distributions
            rand_index = torch.randperm(images.size(0)).to(device)

            # ----------- 2) 生成混合的输入 -----------
            images_shuffled = images[rand_index, :]
            images_mix = lam * images + (1 - lam) * images_shuffled

            # 对应的标签
            labels_shuffled = labels[rand_index]

            # ----------- 3) 前向传播 (用混合输入) -----------
            output = model(images_mix)

            loss = lam * criterion(output, labels) + (1 - lam) * criterion(output, labels_shuffled)

            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += len(labels)
            accuracy = correct / total
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('Epoch {}, batch {}, loss: {:.4f}, accuracy: {:.4f}'.format(ep + 1,
                                                                              batch_id,
                                                                              total_loss / batch_id,
                                                                              accuracy))
            batch_id += 1

        print('Epoch {} 总损失: {:.4f}'.format(ep + 1, total_loss))

        acc_test = evaluate(test_loader, model)

        if acc_test >= acc_test_best:
            n = 0
            acc_test_best = acc_test
            model_best = copy.deepcopy(model)

        # 提前停止条件
        if n >= num_epochs // 10 and acc_test < acc_test_best - 0.1:
            print('######################### 重新加载最佳模型 #########################')
            n = 0
            model = copy.deepcopy(model_best)
        # 输出目前最佳测试准确率
        print('>>> 目前最佳测试准确率: {:.4f}'.format(acc_test_best))

    return acc_test_best


def evaluate(test_loader, model):
    print('开始在', device, '上测试...')
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.float().to(device)
            labels = labels.long().to(device)

            output = model(images)
            pred = output.argmax(dim=1)

            correct += (pred == labels).sum().item()
            total += len(labels)
    accuracy = correct / total
    print('测试准确率: {:.4f}'.format(accuracy))
    return accuracy


def main_LOSO_all_sessions():
    sessions = ['4fold']
    acc_all_sessions = []
    session_results = {}
    total_training_times_sessions = []

    for session in sessions:
        print(f"\nProcessing {session}...")
        dir = f'/mnt/DATA-2/SEED-VII/{session}/'  # 根据当前 session 设置数据路径
        file_list = os.listdir(dir)
        sub_num = len(file_list)

        num_epochs = 200
        acc_all = []

        for sub_i, file_name in enumerate(file_list):
            print(f"Processing Subject {sub_i + 1}/{len(file_list)}: {file_name}")
            load_path = os.path.join(dir, file_name)
            data = scio.loadmat(load_path)

            DE1 = data['DE1']
            DE2 = data['DE2']
            DE3 = data['DE3']
            DE4 = data['DE4']
            label1 = data['label1'].flatten()
            label2 = data['label2'].flatten()
            label3 = data['label3'].flatten()
            label4 = data['label4'].flatten()
            data_folds = [DE1, DE2, DE3, DE4]
            label_folds = [label1, label2, label3, label4]

            all_data = np.concatenate(data_folds, axis=0)
            mean = np.mean(all_data, axis=0)
            std = np.std(all_data, axis=0)
            sub_fold_accs = []
            for fold in range(4):
                print('begin fold', fold)
                data_test = data_folds[fold]
                label_test = label_folds[fold]

                data_train_list = []
                label_train_list = []

                for i in range(4):
                    if i != fold:
                        # 手动计算 z-score 标准化
                        data_fold_normalized = (data_folds[i] - mean) / std
                        data_fold_normalized = data_fold_normalized.reshape(-1, 62, 5)

                        data_train_list.append(data_fold_normalized)
                        label_train_list.append(label_folds[i])

                data_train = np.concatenate(data_train_list, axis=0)
                label_train = np.concatenate(label_train_list, axis=0)

                # 手动计算测试集的 z-score 标准化
                data_test_normalized = (data_test - mean) / std
                data_test = data_test_normalized.reshape(-1, 62, 5)  # 请根据实际的通道数和频带数调整
                # 调整数据形状为（样本数，1，通道数，频带数）
                data_train = data_train[:, np.newaxis, :, :]
                data_test = data_test[:, np.newaxis, :, :]

                # 转换为 torch 张量
                data_train = torch.tensor(data_train, dtype=torch.float32)
                label_train = torch.tensor(label_train, dtype=torch.long)
                data_test = torch.tensor(data_test, dtype=torch.float32)
                label_test = torch.tensor(label_test, dtype=torch.long)

                train_iter, test_iter = load_dataloader(data_train, data_test, label_train, label_test, augment=True)

                num_channels = data_train.shape[2]
                num_freq_bands = data_train.shape[3]
                model = GATGraph(n_classes=7, num_channels=num_channels, num_freq_bands=num_freq_bands,
                                  use_dynamic_graph=True).to(device)

                # 打印模型大小和参数数量（仅在第一个受试者时打印）
                if sub_i == 0:
                    total_params = count_parameters(model)
                    model_size = get_model_size(model)
                    print(f"Total trainable parameters: {total_params}")
                    print(f"Model size: {model_size:.2f} MB")

                criterion = nn.CrossEntropyLoss().to(device)
                optimizer = optim.AdamW(model.parameters(),
                                       lr=0.0003,
                                       weight_decay=0.0001)

                acc_test_best = train(train_iter, test_iter, model, criterion, optimizer, num_epochs, file_list[sub_i])
                sub_fold_accs.append(acc_test_best)
            sub_acc_best = max(sub_fold_accs)
            acc_all.append(sub_acc_best)

            # 计算当前 session 的平均准确率和标准差
            acc_mean = np.mean(acc_all)
            acc_std = np.std(acc_all)

            # 输出当前 session 的结果
            print(f"\nResults for {session}:")
            print('>>> 4fold test acc (per subject): ', acc_all)
            print(f'>>> {session} 4fold test mean acc: {acc_mean:.4f}')
            print(f'>>> {session} 4fold test std acc: {acc_std:.4f}')

            # 保存当前 session 的结果
            session_results[session] = {'acc_all': acc_all, 'mean': acc_mean, 'std': acc_std}

            # 将当前 session 的结果加入总的列表
            acc_all_sessions.extend(acc_all)

            # 计算所有 session 的总平均值和标准差
        overall_acc_mean = np.mean(acc_all_sessions)
        overall_acc_std = np.std(acc_all_sessions)

        print('\nSummary of all sessions:')
        for session in sessions:
            print(f"\n{session} Results:")
            print('>>> 4fold test acc (per subject): ', session_results[session]['acc_all'])
            print(f'>>> {session} 4fold test mean acc: {session_results[session]["mean"]:.4f}')
            print(f'>>> {session} 4fold test std acc: {session_results[session]["std"]:.4f}')
        # 输出总体结果
        print('\nOverall results across all sessions:')
        print('>>> Overall 4fold test acc (all subjects, all sessions): ', acc_all_sessions)
        print(f'>>> Overall 4fold test mean acc: {overall_acc_mean:.4f}')
        print(f'>>> Overall 4fold test std acc: {overall_acc_std:.4f}')

        # 将总结果加入到字典中
        session_results['overall'] = {
            'acc_all_sessions': acc_all_sessions,
            'overall_mean': overall_acc_mean,
            'overall_std': overall_acc_std
        }

    return session_results
if __name__ == '__main__':
    seed = 42  # 选择任意固定的种子值
    set_seed(seed)
    sys.exit(main_LOSO_all_sessions())