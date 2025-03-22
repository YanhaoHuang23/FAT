import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from einops import rearrange
from scipy.spatial import distance
import scipy.sparse as sp
from scipy.stats import zscore
#############################
# feature_trans.py / graphpool.py
#############################

def feature_trans(subgraph_num, feature):
    """
    根据 subgraph_num 选用不同的分割策略，将原始特征进行重排序。
    """
    if subgraph_num == 7:   # 脑区分割
        return feature_trans_7(feature)
    elif subgraph_num == 4: # 跨区域分割
        return feature_trans_4(feature)
    elif subgraph_num == 2: # 半脑分割
        return feature_trans_2(feature)
    else:
        return feature  # 默认不做变换

def location_trans(subgraph_num, location):
    """
    同理，对于坐标也进行相应重排。
    """
    if subgraph_num == 7:
        return location_trans_7(location)
    elif subgraph_num == 4:
        # 这里你未提供 location_trans_4 的实现，可以根据需要自行添加
        # 先注释或使用原始:
        # return location_trans_4(location)
        return location
    elif subgraph_num == 2:
        return location_trans_2(location)
    else:
        return location

##################################################
# 2.1) 半脑分割
##################################################
def feature_trans_2(feature):
    """
    将 feature 按半脑分块拼接
    feature.shape = [batch_size, 62, feats_dim]
    这里根据特定通道索引进行重新排序。
    """
    # 下列索引要根据实际需要确认
    reassigned_feature = torch.cat((
        feature[:, 0:1], feature[:, 3:4], feature[:, 5:9],
        feature[:, 14:18], feature[:, 23:27], feature[:, 32:36],
        feature[:, 41:45], feature[:, 50:53], feature[:, 57:59],

        feature[:, 2:3], feature[:, 4:5], feature[:, 10:14],
        feature[:, 19:23], feature[:, 28:32], feature[:, 37:41],
        feature[:, 46:50], feature[:, 54:57], feature[:, 60:62],
    ), dim=1)
    return reassigned_feature

def location_trans_2(location):
    """
    对应特征的半脑分割下的坐标重排序
    location.shape = [62, 3] (每个通道3D坐标)
    """
    reassigned_location = torch.cat((
        location[0:1], location[3:4], location[5:9],
        location[14:18], location[23:27], location[32:36],
        location[41:45], location[50:53], location[57:59],

        location[2:3], location[4:5], location[10:14],
        location[19:23], location[28:32], location[37:41],
        location[46:50], location[54:57], location[60:62],
    ), dim=0)
    return reassigned_location


##################################################
# 2.2) 跨区域分割
##################################################
def feature_trans_4(feature):
    """
    这里仅给出了 feature 的重排示例，但没有相应的 location_trans_4。
    若需要对坐标做类似操作，可自行实现同样的逻辑
    """
    reassigned_feature = torch.cat((
        feature[:, 5:6], feature[:, 14:15], feature[:, 23:24],
        feature[:, 13:14], feature[:, 22:23], feature[:, 31:32],

        feature[:, 6:7], feature[:, 15:16], feature[:, 24:25],
        feature[:, 12:13], feature[:, 21:22], feature[:, 30:31],

        feature[:, 32:33], feature[:, 41:42],
        feature[:, 40:41], feature[:, 49:50],

        feature[:, 33:34], feature[:, 42:43], feature[:, 50:51],
        feature[:, 39:40], feature[:, 48:49], feature[:, 56:57],
    ), dim=1)
    return reassigned_feature


##################################################
# 2.3) 脑区分割
##################################################
def feature_trans_7(feature):
    """
    适合 subgraph_num=7 的脑区划分
    """
    reassigned_feature = torch.cat((
        feature[:, 0:5],

        feature[:, 5:8], feature[:, 14:17], feature[:, 23:26],
        feature[:, 23:26], feature[:, 32:35], feature[:, 41:44],

        feature[:, 7:12], feature[:, 16:21], feature[:, 25:30],
        feature[:, 34:39], feature[:, 43:48],

        feature[:, 11:14], feature[:, 20:23], feature[:, 29:32],
        feature[:, 29:32], feature[:, 38:41], feature[:, 47:50],

        feature[:, 50:62]
    ), dim=1)
    return reassigned_feature

def location_trans_7(location):
    """
    对应坐标的同理操作
    """
    reassigned_location = torch.cat((
        location[0:5],

        location[5:8], location[14:17], location[23:26],
        location[23:26], location[32:35], location[41:44],

        location[7:12], location[16:21], location[25:30],
        location[34:39], location[43:48],

        location[11:14], location[20:23], location[29:32],
        location[29:32], location[38:41], location[47:50],

        location[50:62]
    ), dim=0)
    return reassigned_location
class LocalLayer(Module):
    """
    在局部图上做一次类 GCN 的操作:
      forward(input, lap) => (batch_size, 62, out_features)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(LocalLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lrelu = nn.LeakyReLU(0.1)
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, lap, is_weight=True):
        """
        input.shape = (batch_size, 62, in_features)
        lap.shape   = (62, 62)  # 邻接/拉普拉斯
        """
        if is_weight:
            # (b, 62, inF) x (inF, outF) => (b, 62, outF)
            weighted_feature = torch.einsum('b i j, j d -> b i d', input, self.weight)
            # lap x weighted_feature => (b, 62, outF)
            output = torch.einsum('i j, b j d -> b i d', lap, weighted_feature)
            if self.bias is not None:
                output = output + self.bias
        else:
            # 不加权的情况
            output = torch.einsum('i j, b j d -> b i d', lap, input)
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features} -> {self.out_features})"
class GlobalLayer(nn.Module):
    """
    类似 GAT 的全局注意力层:
      - 在输入特征 + 坐标嵌入后, 计算注意力 (QK), 再和 values 做加权.
    """
    def __init__(self, in_features, out_feature):
        super(GlobalLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_feature
        self.num_heads = 6
        self.lrelu = nn.LeakyReLU(0.1)

        # 坐标嵌入, 仅举例
        self.embed = nn.Linear(3, 30)  # location embedding

        # 用于生成 queries, keys
        self.get_qk = nn.Linear(self.in_features, self.in_features * 2)

        self.equ_weights = Parameter(torch.FloatTensor(self.num_heads))  # 可能用于多头加权
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.bias   = Parameter(torch.FloatTensor(self.out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.equ_weights.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, h, res_coor):
        """
        h.shape        = [batch_size, N, in_features]
        res_coor.shape = [N, 3] or [batch_size, N, 3]?
        """
        # 1) 位置编码
        #   如果 res_coor 是 [N,3], 需要 broadcast => [b,N,3]
        #   此处只给出示例, 你可根据实际 shape 做 reshape
        if res_coor.dim() == 2:
            # 假设(62,3) => repeat到 batch_size
            b_size = h.size(0)
            res_coor = res_coor.unsqueeze(0).expand(b_size, -1, -1)
        coor_embed = self.lrelu(self.embed(res_coor))  # => (b,N,30)

        h_with_embed = h + coor_embed  # (b,N, in_features=30)
        attn_out = self.cal_att_matrix(h, h_with_embed)  # => (b,N,in_features)

        # 投影 => (b,N,out_features)
        output = torch.matmul(attn_out, self.weight) + self.bias
        return output

    def cal_att_matrix(self, feature, feature_with_embed):
        """
        feature:            (b,N,in_features)
        feature_with_embed: (b,N,in_features)
        """
        b, N, _ = feature.shape

        # 1) 生成 queries, keys
        qk = self.get_qk(feature_with_embed)  # => (b,N,2*in_features)
        # reshape => (qk=2, b, heads=6, N, d=?)
        qk = rearrange(qk, "b n (h d qk) -> (qk) b h n d", h=self.num_heads, qk=2)
        queries, keys = qk[0], qk[1]  # => (b, h, N, d)

        values = feature  # => (b, N, in_features)

        dim_scale = (queries.size(-1)) ** -0.5
        # dots => (b,h,N,N)
        dots = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) * dim_scale

        # reduce到(b,N,N), 先把heads压缩
        attn = torch.einsum("b h i j -> b i j", dots)

        # sparse-drop: 保留大权重
        attn = self.dropout_80_percent(attn)
        # softmax along dim=2
        attn = F.softmax(attn/0.3, dim=2)  # => (b, N, N)

        # out => (b,N,in_features)
        out_feature = torch.einsum('b i j, b j d -> b i d', attn, values)
        return out_feature

    def dropout_80_percent(self, attn):
        """
        保留 top ~20% 的attention值, 其余置为 -1e-7
        """
        # 排序, 取 1/5 位置
        att_subview_, _ = attn.sort(dim=2, descending=True)
        threshold = att_subview_[:, :, att_subview_.size(2) // 6]  # top 1/6
        threshold = rearrange(threshold, 'b i -> b i 1').repeat(1,1, attn.size(2))

        attn[attn < threshold] = -1e-7
        return attn
class MesoLayer(nn.Module):
    """
    对某些子图(分块)进行聚合:
      - 根据 subgraph_num(2,4,7) 决定怎么在 feature 里切片
      - attention方式把子图内节点聚合成1个
    """
    def __init__(self, subgraph_num, num_heads, coordinate, trainable_vector):
        super(MesoLayer, self).__init__()
        self.subgraph_num = subgraph_num
        self.coordinate   = coordinate  # shape=[62,3], EEG通道3D坐标
        self.num_heads    = num_heads

        self.graph_list = self.sort_subgraph(subgraph_num)  # 每个子图的节点数列表
        self.emb_size   = 30  # example

        self.trainable_vec = Parameter(torch.FloatTensor(trainable_vector))
        # 自行决定此 param 的含义与 shape，这里只做示例

        # 用于子图内部的 attention
        self.weight = Parameter(torch.FloatTensor(self.emb_size, 10))
        self.lrelu  = nn.LeakyReLU(0.1)
        self.att_softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.trainable_vec.size(0))
        self.trainable_vec.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        x: (batch_size, 62, 30)
        return: coarsen_x, coarsen_coor
        """
        coarsen_x, coarsen_coor = self.att_coarsen(x)
        return coarsen_x, coarsen_coor

    def att_coarsen(self, features):
        # 1) 对特征/坐标进行重排
        # subgraph_num => 2 / 4 / 7, 这里会去调用 feature_trans_x
        new_features = feature_trans(self.subgraph_num, features)       # => [b, reIndexed, 30]
        new_location = location_trans(self.subgraph_num, self.coordinate) # => [reIndexed, 3]

        coarsen_feature_list = []
        coarsen_coord_list   = []

        idx_head = 0
        for index_length in self.graph_list:
            idx_tail = idx_head + index_length
            sub_feat = new_features[:, idx_head:idx_tail]  # => (b, subLen, 30)
            sub_coord= new_location[idx_head:idx_tail]     # => (subLen,3)

            # 2) 计算注意力
            # feature_with_weight => (b, subLen, 10)
            feature_with_weight = torch.einsum('b j d, d h -> b j h', sub_feat, self.weight)
            feature_T = rearrange(feature_with_weight, 'b j h -> b h j')
            # => (b, subLen, subLen)
            att_weight_matrix = torch.einsum('b j h, b h i -> b j i', feature_with_weight, feature_T)
            # 沿最后维度 sum => (b, subLen)
            att_weight_vector = torch.sum(att_weight_matrix, dim=2)
            att_vec = self.att_softmax(att_weight_vector)  # => (b, subLen)

            # 3) 聚合 => (b, feats)
            sub_feature_ = torch.einsum('b j, b j d -> b d', att_vec, sub_feat)
            sub_coord_   = torch.einsum('b j, j d -> b d', att_vec, sub_coord)

            # 均值或别的操作 => (b, d) => 取其中 batch 维再 cat
            coarsen_feature_list.append(rearrange(sub_feature_, "b d -> b 1 d"))
            # 坐标 => 先在 batch 里 average?
            mean_sub_coord = torch.mean(sub_coord_, dim=0) # => shape=(d,)
            coarsen_coord_list.append(rearrange(mean_sub_coord, "d -> 1 d"))

            idx_head = idx_tail

        coarsen_features   = torch.cat(coarsen_feature_list, dim=1) # => (b, #subgraph, d)
        coarsen_coordinates= torch.cat(coarsen_coord_list, dim=0)   # => (#subgraph, d=coords)
        return coarsen_features, coarsen_coordinates

    def sort_subgraph(self, subgraph_num):
        """
        根据子图数量返回每个子图在分割后的节点数
        """
        # 你在代码中自己定义了:
        subgraph_7 = [5, 9, 9, 25, 9, 9, 12]
        subgraph_4 = [6, 6, 4, 6]
        subgraph_2 = [27, 27]

        if subgraph_num == 7:
            return subgraph_7
        elif subgraph_num == 4:
            return subgraph_4
        elif subgraph_num == 2:
            return subgraph_2
        else:
            return [62]  # 默认整张图
class PGCN(nn.Module):
    """
    组合 LocalLayer, MesoLayer, GlobalLayer,
    最后接 MLP 输出分类。
    """
    def __init__(self, args, local_adj, coor):
        """
        args:
          n_class   => 输出类别数
          in_feature=> 输入特征维度 (e.g. 5)
          lr        => LeakyReLU 的斜率
          dropout   => dropout 概率
          module    => 记录使用了哪些模块
        local_adj : shape=(62,62) 用于 local GCN
        coor      : shape=(62,3) 结点坐标, MesoLayer/GlobalLayer 用
        """
        super(PGCN, self).__init__()
        self.args = args
        self.nclass = args.n_class
        self.dropout = args.dropout
        self.l_relu  = args.lr
        self.adj     = local_adj
        self.coordinate = coor

        # Local GCN
        self.local_gcn_1 = LocalLayer(args.in_feature, 10, bias=True)
        self.local_gcn_2 = LocalLayer(10, 15, bias=True)

        # Meso
        # 将原始 feature(可能=5) 先 embed => 30
        self.meso_embed  = nn.Linear(args.in_feature, 30)
        # subgraph_num=7, 2 => 可自己改
        self.meso_layer_1 = MesoLayer(subgraph_num=7, num_heads=6,
                                      coordinate=self.coordinate,
                                      trainable_vector=78)
        self.meso_layer_2 = MesoLayer(subgraph_num=2, num_heads=6,
                                      coordinate=self.coordinate,
                                      trainable_vector=54)
        self.meso_dropout = nn.Dropout(0.2)

        # Global
        self.global_layer_1 = GlobalLayer(30, 40)

        # MLP part
        # 这里你把最终拼接的特征维度写死为了 (71, 70),
        # 需根据你拼接后的 shape 计算.
        self.mlp0 = nn.Linear(71 * 70, 2048)
        self.mlp1 = nn.Linear(2048, 1024)
        self.mlp2 = nn.Linear(1024, self.nclass)

        self.bn = nn.BatchNorm1d(1024)
        self.lrelu = nn.LeakyReLU(self.l_relu)
        self.dp    = nn.Dropout(self.dropout)

    def forward(self, x):
        """
        x.shape = (batch_size, 62, in_feature=5)
        return: logits, lap_matrix, ""
        """

        # step1: Local
        lap_matrix = normalize_adj(self.adj)  # 见下方
        local_x1 = self.lrelu(self.local_gcn_1(x, lap_matrix, True))
        local_x2 = self.lrelu(self.local_gcn_2(local_x1, lap_matrix, True))
        # 这里把 x/local_x1/local_x2 拼成 (batch,62,feats+..)
        res_local = torch.cat((x, local_x1, local_x2), dim=2)  # (b,62, 5+10+15=30)

        if "local" not in self.args.module:
            self.args.module += "local "

        # step2: mesoscopic
        # Meso 使用原始 x 做 embedding => [b,62,30]
        meso_input  = self.meso_embed(x)
        coarsen_x1, coarsen_coor1 = self.meso_layer_1(meso_input)
        coarsen_x1 = self.lrelu(coarsen_x1)

        coarsen_x2, coarsen_coor2 = self.meso_layer_2(meso_input)
        coarsen_x2 = self.lrelu(coarsen_x2)

        # 把 local 和 meso 拼在一起 => (b, 62 + #subgraph7 + #subgraph2, 30) ?
        # 你这里直接 cat((res_local, coarsen_x1, coarsen_x2), dim=1)
        # => res_local.shape=(b,62,30)
        # => coarsen_x1.shape=(b,7,30)  (7 = sum of subgraph_7?),
        # => coarsen_x2.shape=(b,2,30)
        # => => (b, 62+7+2=71, 30)
        res_meso  = torch.cat((res_local, coarsen_x1, coarsen_x2), dim=1)
        res_coor1 = torch.cat((self.coordinate, coarsen_coor1, coarsen_coor2), dim=0)
        # => shape=(62+7+2=71,3)

        if "meso" not in self.args.module:
            self.args.module += "meso "

        # step3: global
        global_x1  = self.lrelu(self.global_layer_1(res_meso, res_coor1))
        # => shape=(b,71,40)

        res_global = torch.cat((res_meso, global_x1), dim=2)
        # => (b,71,30+40=70)

        if "global" not in self.args.module:
            self.args.module += "global"

        # step4: classification
        # flatten => (b, 71*70= 4970)
        x_flat = res_global.view(res_global.size(0), -1)

        x_out = self.lrelu(self.mlp0(x_flat))
        x_out = self.dp(x_out)
        x_out = self.lrelu(self.mlp1(x_out))
        x_out = self.bn(x_out)
        x_out = self.mlp2(x_out)  # => logits, shape=(b,nclass)

        return x_out # , lap_matrix, ""
def normalize_adj(adj):
    """
    传入 shape=(62,62) 的邻接矩阵, 做标准 GCN-like 归一化:
      A_ = D^-1/2 * A * D^-1/2
    返回 (62,62) 张量
    """
    # 先转成 torch
    if not isinstance(adj, torch.Tensor):
        adj = torch.from_numpy(adj).float()
    # 加自环
    adj = adj + torch.eye(adj.size(0), device=adj.device)
    # 计算度
    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    # 做 D^-1/2 * A * D^-1/2
    lap = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    return lap
def get_ini_dis_m():
    """get initial distance matrix"""
    m1 = [(-2.285379, 10.372299, 4.564709),
          (0.687462, 10.931931, 4.452579),
          (3.874373, 9.896583, 4.368097),
          (-2.82271, 9.895013, 6.833403),
          (4.143959, 9.607678, 7.067061),

          (-6.417786, 6.362997, 4.476012),
          (-5.745505, 7.282387, 6.764246),
          (-4.248579, 7.990933, 8.73188),
          (-2.046628, 8.049909, 10.162745),
          (0.716282, 7.836015, 10.88362),
          (3.193455, 7.889754, 10.312743),
          (5.337832, 7.691511, 8.678795),
          (6.842302, 6.643506, 6.300108),
          (7.197982, 5.671902, 4.245699),

          (-7.326021, 3.749974, 4.734323),
          (-6.882368, 4.211114, 7.939393),
          (-4.837038, 4.672796, 10.955297),
          (-2.677567, 4.478631, 12.365311),
          (0.455027, 4.186858, 13.104445),
          (3.654295, 4.254963, 12.205945),
          (5.863695, 4.275586, 10.714709),
          (7.610693, 3.851083, 7.604854),
          (7.821661, 3.18878, 4.400032),

          (-7.640498, 0.756314, 4.967095),
          (-7.230136, 0.725585, 8.331517),
          (-5.748005, 0.480691, 11.193904),
          (-3.009834, 0.621885, 13.441012),
          (0.341982, 0.449246, 13.839247),
          (3.62126, 0.31676, 13.082255),
          (6.418348, 0.200262, 11.178412),
          (7.743287, 0.254288, 8.143276),
          (8.214926, 0.533799, 4.980188),

          (-7.794727, -1.924366, 4.686678),
          (-7.103159, -2.735806, 7.908936),
          (-5.549734, -3.131109, 10.995642),
          (-3.111164, -3.281632, 12.904391),
          (-0.072857, -3.405421, 13.509398),
          (3.044321, -3.820854, 12.781214),
          (5.712892, -3.643826, 10.907982),
          (7.304755, -3.111501, 7.913397),
          (7.92715, -2.443219, 4.673271),

          (-7.161848, -4.799244, 4.411572),
          (-6.375708, -5.683398, 7.142764),
          (-5.117089, -6.324777, 9.046002),
          (-2.8246, -6.605847, 10.717917),
          (-0.19569, -6.696784, 11.505725),
          (2.396374, -7.077637, 10.585553),
          (4.802065, -6.824497, 8.991351),
          (6.172683, -6.209247, 7.028114),
          (7.187716, -4.954237, 4.477674),

          (-5.894369, -6.974203, 4.318362),
          (-5.037746, -7.566237, 6.585544),
          (-2.544662, -8.415612, 7.820205),
          (-0.339835, -8.716856, 8.249729),
          (2.201964, -8.66148, 7.796194),
          (4.491326, -8.16103, 6.387415),
          (5.766648, -7.498684, 4.546538),

          (-6.387065, -5.755497, 1.886141),
          (-3.542601, -8.904578, 4.214279),
          (-0.080624, -9.660508, 4.670766),
          (3.050584, -9.25965, 4.194428),
          (6.192229, -6.797348, 2.355135),
          ]
    dis_m1 = distance.cdist(m1, m1, 'euclidean')

    # 对元素进行检查，小于0时置0
    zero_matrix = np.zeros((62, 62))
    dis_m1 = np.where(dis_m1 > 0, dis_m1, zero_matrix)

    return dis_m1


def convert_dis_m(adj_matrix, delta=8):
    """
    将距离矩阵 => 距离平方反比, 并做阈值
    """
    eye_ = np.eye(adj_matrix.shape[0], dtype=float)
    mat_ = adj_matrix + eye_  # 对角线置1 避免除0
    mat_sq = np.power(mat_, 2)
    mat_out= delta / mat_sq
    mat_out= np.where(mat_out > 1, 1, mat_out)   # clip max=1
    mat_out= np.where(mat_out < 0.1, 0, mat_out) # 稀疏化
    return mat_out

def global_dis_m(adj_matrix, denominator=0.2):
    """
    全局视野的距离矩阵 (举例)
    """
    lower_bound = np.ones_like(adj_matrix) * 1.6
    adj_matrix = np.where(adj_matrix > 1.6, adj_matrix, lower_bound)
    normal_dis = denominator / np.log10(adj_matrix)
    return normal_dis

def return_coordinates():
    "return absolute coordinates"
    m1 = [(-2.285379, 10.372299, 4.564709),
          (0.687462, 10.931931, 4.452579),
          (3.874373, 9.896583, 4.368097),
          (-2.82271, 9.895013, 6.833403),
          (4.143959, 9.607678, 7.067061),

          (-6.417786, 6.362997, 4.476012),
          (-5.745505, 7.282387, 6.764246),
          (-4.248579, 7.990933, 8.73188),
          (-2.046628, 8.049909, 10.162745),
          (0.716282, 7.836015, 10.88362),
          (3.193455, 7.889754, 10.312743),
          (5.337832, 7.691511, 8.678795),
          (6.842302, 6.643506, 6.300108),
          (7.197982, 5.671902, 4.245699),

          (-7.326021, 3.749974, 4.734323),
          (-6.882368, 4.211114, 7.939393),
          (-4.837038, 4.672796, 10.955297),
          (-2.677567, 4.478631, 12.365311),
          (0.455027, 4.186858, 13.104445),
          (3.654295, 4.254963, 12.205945),
          (5.863695, 4.275586, 10.714709),
          (7.610693, 3.851083, 7.604854),
          (7.821661, 3.18878, 4.400032),

          (-7.640498, 0.756314, 4.967095),
          (-7.230136, 0.725585, 8.331517),
          (-5.748005, 0.480691, 11.193904),
          (-3.009834, 0.621885, 13.441012),
          (0.341982, 0.449246, 13.839247),
          (3.62126, 0.31676, 13.082255),
          (6.418348, 0.200262, 11.178412),
          (7.743287, 0.254288, 8.143276),
          (8.214926, 0.533799, 4.980188),

          (-7.794727, -1.924366, 4.686678),
          (-7.103159, -2.735806, 7.908936),
          (-5.549734, -3.131109, 10.995642),
          (-3.111164, -3.281632, 12.904391),
          (-0.072857, -3.405421, 13.509398),
          (3.044321, -3.820854, 12.781214),
          (5.712892, -3.643826, 10.907982),
          (7.304755, -3.111501, 7.913397),
          (7.92715, -2.443219, 4.673271),

          (-7.161848, -4.799244, 4.411572),
          (-6.375708, -5.683398, 7.142764),
          (-5.117089, -6.324777, 9.046002),
          (-2.8246, -6.605847, 10.717917),
          (-0.19569, -6.696784, 11.505725),
          (2.396374, -7.077637, 10.585553),
          (4.802065, -6.824497, 8.991351),
          (6.172683, -6.209247, 7.028114),
          (7.187716, -4.954237, 4.477674),

          (-5.894369, -6.974203, 4.318362),
          (-5.037746, -7.566237, 6.585544),
          (-2.544662, -8.415612, 7.820205),
          (-0.339835, -8.716856, 8.249729),
          (2.201964, -8.66148, 7.796194),
          (4.491326, -8.16103, 6.387415),
          (5.766648, -7.498684, 4.546538),

          (-6.387065, -5.755497, 1.886141),
          (-3.542601, -8.904578, 4.214279),
          (-0.080624, -9.660508, 4.670766),
          (3.050584, -9.25965, 4.194428),
          (6.192229, -6.797348, 2.355135),
          ]

    m1 = np.array(m1)
    return m1
