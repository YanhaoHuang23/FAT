from torch import nn
from utils import normalize_A
from einops import rearrange
from FANLayer import FANLayer
import torch

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


class FAA(nn.Module):
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
        if isinstance(self.fn, FAA):
            return x + self.fn(x, **kwargs)
        else:
            return x + self.fn(x)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )


class AttentionBlock(nn.Module):
    def __init__(self, emb_size, num_heads, drop_p, use_dynamic_graph=False):
        super().__init__()
        self.layernorm = nn.LayerNorm(emb_size)
        self.faa = FAA(emb_size, num_heads, drop_p, use_dynamic_graph=use_dynamic_graph)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x, mask=None, dynamic_graph1=None, dynamic_graph2=None):
        # 1. LayerNorm
        x_norm = self.layernorm(x)
        # 2. Multi-Head Attention（传入额外的 dynamic_graph 参数）
        out = self.faa(x_norm, mask=mask, dynamic_graph1=dynamic_graph1, dynamic_graph2=dynamic_graph2)
        # 3. Dropout
        out = self.dropout(out)
        # 4. 残差连接
        return x + out


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
    def __init__(self, depth, emb_size, num_heads=4, drop_p=0.1, num_dim=63, use_dynamic_graph=False):
        super().__init__()

        self.use_dynamic_graph = use_dynamic_graph
        self.dynamic_graph_learner1 = DynamicGraphLearner(n=num_dim, num_heads=num_heads // 2) if use_dynamic_graph else None
        self.dynamic_graph_learner2 = DynamicGraphLearner(n=num_dim, num_heads=num_heads // 2) if use_dynamic_graph else None

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


class FAT(nn.Module):
    def __init__(self, emb_size=40, depth=6, n_classes=4,
                 num_channels=62, num_freq_bands=5, num_heads=8,
                 use_dynamic_graph=True):
        super(FAT, self).__init__()
        self.patch_embedding = ModifiedPatchEmbedding2D(emb_size, num_channels, num_freq_bands)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_encoding = PositionalEncoding(emb_size, dropout=0.1,
                                                      max_len=num_channels + 1)
        self.transformer_encoder = TransformerEncoder(
            depth, emb_size, num_heads, drop_p=0.2, num_dim=num_channels+1, use_dynamic_graph=use_dynamic_graph
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