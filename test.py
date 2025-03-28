import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, drug_features, protein_features):
        # 对齐样本数（以药物特征为基准）
        protein_features = protein_features[:drug_features.size(0), :]

        # 计算注意力权重
        Q = self.query(drug_features)
        K = self.key(protein_features)
        V = self.value(protein_features)
        attention_scores = torch.matmul(Q, K.T) / (self.feature_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权求和
        attended_features = torch.matmul(attention_weights, V)
        return attended_features

# 示例
drug_features = torch.randn(708, 128)
protein_features = torch.randn(508, 128)


attention_fusion = AttentionFusion(feature_dim=128)
fused_features = attention_fusion(drug_features, protein_features)
print(fused_features.shape)  # 输出: torch.Size([708, 128])


# import argparse
# parser = argparse.ArgumentParser(description='Graph Neural Network Configuration')
# args = parser.parse_args()
# args.task = "lp"
#
# args.val_prop =0.05
# args.test_prop =0.1
# args.use_feats =1
# args.normalize_feats =1
# args.normalize_adj =1
# args.split_seed =1234
#
#
# from utils.data_utils import load_data
# data = load_data(args)#+datapath
# args.n_nodes, args.feat_dim = data['features'].shape
# print(args)
# print(data.keys())