import torch
import numpy as np
import torch.nn as nn
from scipy.sparse import  diags
from scipy.sparse.linalg import inv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse import csc_matrix
class GraphWaveletTransform(nn.Module):
    def __init__(self):
        super().__init__()


    def random_walk_matrix(self, A):
        """
        构建随机游走矩阵P
        :param x:邻接矩阵
        :return:随机游走矩阵P
        """
        n = A.shape[0]
        A_dense = A.to_dense()
        A_self_loops = A_dense + np.eye(n)
        d = diags(np.array(A_self_loops.sum(axis=1)).flatten())# 度矩阵
        P = inv(d) @ A_self_loops  # 随机游走矩阵 P = D^{-1} A
        return P

    def graph_wavelet_transform(self,P, scales):
        """
        图小波变换（基于随机游走扩散）
        :param P:随机游走矩阵
        :param scales: 尺度列表
        :return:小波系数列表
        """
        wavelets = []
        for scale in scales:
            T = np.linalg.matrix_power(P, scale)  # 扩散算子 T = P^scale
            wavelets.append(T)
        return wavelets

    def extract_features(self,wavelets, X):
        """
        提取图小波特征
        :param wavelets: 小波系数列表
        :param X: 节点特征矩阵
        :return: 图小波特征
        """
        features = []
        for wavelet in wavelets:
            features.append(wavelet @ X)  # 将小波系数应用于节点特征
        return np.hstack(features)  # 拼接所有尺度的特征

class GWTNet(nn.Module):
    def __init__(self,args):
        self.scales = args.scales
        super().__init__()
        self.graphwavelettransform = GraphWaveletTransform()
        self.endfeature = nn.Linear(381,128)
        self.normalize = nn.LayerNorm(len(args.scales)*128)

    def forward(self,featrue,X):

        P = self.graphwavelettransform.random_walk_matrix(X)
        wavelets = self.graphwavelettransform.graph_wavelet_transform(P,self.scales)
        GWTfeature = self.graphwavelettransform.extract_features(wavelets,featrue)


        # return self.normalize(self.endfeature(torch.tensor(GWTfeature, dtype=torch.float32)))

        return self.normalize(torch.tensor(GWTfeature, dtype=torch.float32))
    def decode(self, h, idx):
        h = self.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        dist = (emb_in * emb_out).sum(dim=1)
        return dist


    def compute_metrics(self,embedding,edges,edges_false):
        pos_scores = self.decode(embedding, edges)
        neg_scores = self.decode(embedding, edges_false)
        preds = torch.cat([pos_scores, neg_scores])
        pos_labels = torch.ones(pos_scores.size(0), dtype=torch.float)
        neg_labels = torch.zeros(neg_scores.size(0), dtype=torch.float)
        labels = torch.cat([pos_labels, neg_labels])
        loss = F.binary_cross_entropy_with_logits(preds, labels)
        preds_list = preds.tolist()
        labels_list = labels.tolist()


        roc = roc_auc_score(labels_list, preds_list)
        ap = average_precision_score(labels_list, preds_list)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics


