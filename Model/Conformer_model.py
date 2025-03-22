from torch import nn
from einops import rearrange, reduce
import torch.nn.functional as F
import torch

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



class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
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
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes, drop_p=0.3):
        super().__init__(
            nn.LayerNorm(emb_size),  # Normalize features
            nn.Linear(emb_size, n_classes)  # Map to class logits
        )


class Conformer(nn.Module):
    def __init__(self, emb_size=40, depth=6, n_classes=4,
                 num_channels=62, num_freq_bands=5):
        super(Conformer, self).__init__()
        self.patch_embedding = ModifiedPatchEmbedding2D(emb_size, num_channels, num_freq_bands)
        self.transformer_encoder = TransformerEncoder(depth, emb_size)
        self.classification_head = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)  # [B, C, emb_size]
        x = self.transformer_encoder(x)       # [B, C, emb_size]
        cls_output = x[:, 0, :]  # [B, emb_size]
        logits = self.classification_head(cls_output)
        return logits
