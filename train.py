import torch
import torch.nn as nn
import geoopt
from geoopt.manifolds import PoincareBall,Lorentz
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)
import manifolds
from models import encoders
from layers.layers import FermiDiracDecoder
from utils.eval_utils import MarginLoss
import torch.nn.functional as F
from utils.train_utils import get_dir_name, format_metrics
from models.models import GWTNet
# 双曲特征提取模块
class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = 'Lorentz'
        self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)
        h = self.encoder.encode(x, adj)
        return h

class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """
    def __init__(self, args,drug_dim, hidden_dim):
        super(LPModel, self).__init__(args)
        # self.dc = FermiDiracDecoder(r=2., t=1.)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.loss = MarginLoss(2.)

    def decode(self, h, idx):

        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        return -sqdist

    def compute_metrics(self, embeddings, data, split):

        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        preds = torch.stack([pos_scores, neg_scores], dim=-1)

        loss = self.loss(preds)

        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())

        # 计算准确率
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)



        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1,'accuracy': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

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


# 全双曲神经网络模型

class FullyHyperbolicNN(nn.Module):
    def __init__(self, args, drug_dim, protein_dim, hidden_dim):
        super().__init__()
        self.manifold = Lorentz()  # 双曲空间
        self.drug_extractor = LPModel(args,drug_dim, hidden_dim)
        self.protein_extractor = LPModel(args,protein_dim, hidden_dim)

        
        self.fusion = AttentionFusion(feature_dim=128)
        self.dropout = nn.Dropout(0.2)

    def forward(self, drug_feat, protein_feat, idx,protein_idx):
        # 特征提取
        
        drug_feat = self.drug_extractor.encode(drug_feat,idx)

        # protein_feat = protein_feat[:drug_feat.size(0), :]
        
        # protein_feat = self.drug_extractor.encode(protein_feat,protein_idx)
        # # 特征融合
        # fused_feat = self.fusion(drug_feat, protein_feat)

        return drug_feat
    def contrastive_loss(self,z1, z2, temperature=0.2):
        """
        HLCL 对比学习损失函数
        :param z1: 第一个视图的节点嵌入
        :param z2: 第二个视图的节点嵌入
        :param temperature: 温度参数
        :return: 对比损失
        """
        z1 = F.normalize(z1, p=2, dim=1)  # L2 归一化
        z2 = F.normalize(z2, p=2, dim=1)  # L2 归一化

        # 计算相似度矩阵
        sim_matrix = torch.exp(torch.mm(z1, z2.t()) / temperature)

        # 正样本对是同一节点的两个视图
        pos_sim = torch.diag(sim_matrix)

        # 负样本对是不同节点的视图
        neg_sim = sim_matrix.sum(dim=1) - pos_sim

        # InfoNCE 损失
        loss = -torch.log(pos_sim / neg_sim).mean()
        return loss

    def generate_graph_features_with_dropout(self,embeddings):
        return self.dropout(embeddings)

#设置参数
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.model = "HyboNet"
args.num_layers = 2
args.dropout = 0.0
args.bias = 1
args.use_att = 1
args.local_agg = 0
args.val_prop =0.05
args.test_prop =0.1
args.use_feats =1
args.normalize_feats =1
args.normalize_adj =1
args.split_seed =1234
args.manifold = "Lorentz"
args.act = None
args.dim = 128
args.task = "lp"
args.c = 1.
args.cuda = -1
args.n_heads = 8
args.alpha = 0.2
args.pretrained_embeddings = None
args.save = 0
args.patience = 200
args.min_epochs = 200
args.scales = [1,2,3,4]
# 数据加载
from utils.data_utils import load_data
data = load_data(args)
args.n_nodes, args.feat_dim = data['features'].shape
args.nb_false_edges = len(data['train_edges_false'])
args.nb_edges = len(data['train_edges'])


# 初始化模型
model = FullyHyperbolicNN(args=args,drug_dim=128, protein_dim=128, hidden_dim=64)
GWTmodel = GWTNet(args)
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val_metrics = model.drug_extractor.init_metric_dict()
best_test_metrics = None
best_emb = None
counter = 0


# 训练循环
for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    embeddings = model(data['features'], data['protein_features'],data["adj_train_norm"],data['protein_adj'])



    protein_embeddings_view1 = model.generate_graph_features_with_dropout(embeddings)
    protein_embeddings_view2 = model.generate_graph_features_with_dropout(embeddings)
    contrastive_loss_drug = model.contrastive_loss(protein_embeddings_view1, protein_embeddings_view2)



    embeddings = embeddings.detach().numpy()
    embeddings = GWTmodel(embeddings,data["adj_train_norm"])
    edges_false = data['train_edges_false'][np.random.randint(0, args.nb_false_edges, args.nb_edges)]
    train_metrics = GWTmodel.compute_metrics(embeddings, data["train_edges"], edges_false)



    all_train_metrics = 0.1*contrastive_loss_drug + train_metrics['loss']



    # train_metrics = model.drug_extractor.compute_metrics(embeddings, data, 'train')
    all_train_metrics.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {train_metrics['loss'].item()},roc:{train_metrics['roc'].item()},ap:{train_metrics['ap'].item()}")

    with torch.no_grad():
        model.eval()
        embeddings = model(data['features'], data['protein_features'],data["adj_train_norm"],data['protein_adj'])



        embeddings = embeddings.detach().numpy()
        embeddings = GWTmodel(embeddings,data["adj_train_norm"])



        val_metrics = model.drug_extractor.compute_metrics(embeddings, data, 'val')
        print(f"Epoch {epoch}, Loss: {val_metrics['loss'].item()},roc:{val_metrics['roc'].item()},ap:{val_metrics['ap'].item()}")
        if model.drug_extractor.has_improved(best_val_metrics, val_metrics):
            best_test_metrics = model.drug_extractor.compute_metrics(
                embeddings, data, 'test')
            best_emb = embeddings.cpu()
            if args.save:
                # np.save(os.path.join(save_dir, 'embeddings.npy'),
                #         best_emb.detach().numpy())
                pass
            best_val_metrics = val_metrics
            counter = 0
        else:
            counter += 1
            if counter == args.patience and epoch > args.min_epochs:
                print("Early stopping")
                break
# model.eval()
# embeddings = model(data['features'], data['protein_features'], data["adj_train_norm"], data['protein_adj'])
# test_metrics = model.drug_extractor.compute_metrics(embeddings, data, 'test')

if not best_test_metrics:
    model.eval()
    best_emb = model(data['features'], data['protein_features'], data["adj_train_norm"], data['protein_adj'])

    # best_emb = best_emb.detach().numpy()
    # best_emb = GWTmodel(best_emb,data["adj_train_norm"])

    # best_test_metrics = model.drug_extractor.compute_metrics(best_emb, data, 'test')
print(" ".join(
    ["Val set results:",
     format_metrics(best_val_metrics, 'val')]))
print(" ".join(
    ["Test set results:",
     format_metrics(best_test_metrics, 'test')]))
