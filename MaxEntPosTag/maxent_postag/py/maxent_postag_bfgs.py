# coding:utf-8
"""

"""
import numpy as np
from collections import defaultdict


class BFGS:

    def __init__(self):
        # 词性标记集，也就是最大熵模型可预测的类别集合
        self.labels = set()
        # 每个特征函数的权重，也就是模型要学习的参数
        self.w = None

    # 训练模型
    def train(self, train_toks, max_iter=100, acc=0.001):
        # 特征函数关于经验分布f(x,y)的期望值 Ep_wan(fi)
        ep_wan_fi_dict = defaultdict(int)
        # fi的频次
        fi_dict = defaultdict(int)
        # 经验分布P(X)
        px_dict = defaultdict(int)

        # 遍历每个事件
        for fea_dict, label in train_toks:
            # 更新词性标记集
            self.labels.add(label)
            # 更新(x, y)计数
            ep_wan_fi_dict[(fea_dict, label)] += 1
            # 更新(x)计数
            px_dict[fea_dict] += 1
            # 更新fi的频次
            fi_dict[(fea_dict, label)] += 1

        # 事件总数
        N = len(train_toks)
        for fi in ep_wan_fi_dict.keys():
            # 计算每种特征函数关于分布 f(x,y)的期望值Ep_wan(fi)
            ep_wan_fi_dict[fi] = (ep_wan_fi_dict[fi]/ N)*ep_wan_fi_dict[fi]
            # 计算边缘分布P(X)
            px_dict[fi[0]] /= N

        # 特征函数总数
        n = len(ep_wan_fi_dict)
        # 选定初始点
        w_k = np.zeros(n)
        # 取B0正定对称矩阵
        B0 = np.eye(n)
        # 置迭代次数为0
        k = 0
        # 计算gk
        gk = np.zeros(n)
        for i, (fea_dict, label), ep_wan_fi in enumerate(ep_wan_fi_dict.items()):
            gk[i] = px_dict[fea_dict]*fi_dict[(fea_dict, label)]-ep_wan_fi
        # 如果||gk||<acc，则停止计算
        if np.linalg.norm(gk) < acc:
            self.w = w_k

    # 预测
    def predict(self, event):
        fea_dict, label = event











    # 计算Pw(y|x)
    def probwgt(self, features, label):
        wgt = 0.0
        for f in features:
            if (label, f) in self.feats:
                wgt += self.w[self.feats[(label, f)]]
        return np.exp(wgt)
