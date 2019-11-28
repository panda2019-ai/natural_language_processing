# encoding: utf-8
"""
最大熵词形标注示例
"""

from collections import defaultdict
import math
import codecs


class MaxEnt(object):
    def __init__(self):
        # 存储特征函数fi(x,y)关于经验分布p(x,y)的期望值 Ep_(fi)
        self.feats = defaultdict(int)
        # 文本经过特征抽取后建成的训练集每一项用一个事件表示(词性标记,特征1,特征2,特征3)，
        self.trainset = []
        # 词性标记集，表示Y可取的值的集合，比如名词、动词、形容词...
        self.labels = set()

    def generate_events(self, text, train_flag=False):
        """
        text: 以空格分隔的文本
        train_flag: train_flag=True时，要求text必须标注好词语词性
        功能：对输入的一句话，抽取出事件列表
        """
        # 定义事件序列，注意输入的1行文本可以抽取出多个事件，事件序列称为数据集或许更合适
        event_li = []
        # 分词，要求输入的字符串中词语之间以空白分隔，对于训练集在词语后还要加词性标记，格式为word/pos
        word_li = text.split()
        # 分离词语和词性
        if train_flag:
            # 对于含有词性标记的词语序列，分离词语和词性：[(word1,pos1),(word2,pos2),...]
            word_li = [tuple(w.split(u'/')) for w in word_li if len(w.split(u'/')) == 2]
        else:
            # 对于没有词性标记的词语序列，添加词性"x_pos"：[(word1,'x_pos'),(word2,'x_pos'),...]
            word_li = [(w, u'x_pos') for w in word_li]
        # 为词语序列添加头元素和尾元素，便于后续抽取事件
        word_li = [(u'pre1', u'pre1_pos')] + word_li + [(u'pro1', u'pro1_pos')]
        # 遍历中心词抽取1个事件，每个事件由1个词性标记和多个特征项构成
        for i in range(1, len(word_li) - 1):
            # 特征函数1 中心词
            fea_1 = word_li[i][0]
            # 特征函数2 前一个词
            fea_2 = word_li[i - 1][0]
            # 特征函数3 后一个词
            fea_3 = word_li[i + 1][0]
            # 词性y
            y = word_li[i][1]
            # 构建1个事件，注意1个事件由3个特征项构成，同一事件中的3个特征项共享1个输出标记y
            # 因此1个事件对应3个特征函数或者叫做联合特征：f1(fea_1,y),f2(fea_2,y),f3(fea_3,y)
            # 注意这里并没有区分3个特征的先后顺序
            fields = [y, fea_3, fea_2, fea_1]
            # 将事件添加到事件序列
            event_li.append(fields)
        # 返回事件序列
        return event_li

    def load_data(self, file):
        """
        file: 训练文件路径和文件名
        功能：读取文本数据构建训练集
        """
        with codecs.open('../data/199801.txt', 'rb', 'utf8', 'ignore') as infile:
            for line_ser, line in enumerate(infile):
                if line_ser >= 100:
                    break
                line = line.strip()
                if line:
                    # 抽取事件序列
                    events_li = self.generate_events(line, train_flag=True)
                    # 遍历每个事件，对每个事件构建3个特征函数或者叫做联合特征fi(x,y)
                    for event in events_li:
                        # 第1列是输出词性标记y
                        label = event[0]
                        # 更新词性标记
                        self.labels.add(label)
                        # 对联合特征(词性标记，特征)计数，以便后续计算P(X,Y)的经验分布
                        # 注意一个联合特征为(y,x)的组合，x表示一个特征值，并且丢失掉了x的位置信息
                        for f in set(event[1:]):
                            self.feats[(label, f)] += 1
                        # 将事件添加到训练集
                        self.trainset.append(event)

    def _initparams(self):
        """
        功能：根据load_data方法加载进来的事件集（训练集）计算Ep_(fi)，并且给联合特征(y,x)编号
        """
        # 训练集中抽取的事件总数
        self.N = len(self.trainset)
        # 1个事件对应的特征数量，取值为遍历训练集中每个事件抽取出的最大特征数量
        self.M = max([len(record) - 1 for record in self.trainset])
        # 计算每一种特征函数fi(x,y)关于经验分布P_(X,Y)的期望值，指的是《统计学习方法》P82页最下边的式子
        self.ep_ = [0.0] * len(self.feats)
        # 计算Ep_(fi)，并且用self.feats存储联合特征的索引
        for i, union_feature in enumerate(self.feats):
            # 第i个特征函数的经验值=联合特征(y,x)出现次数/训练集中事件总数。
            self.ep_[i] = float(self.feats[union_feature]) / float(self.N)
            # 记录特征函数索引，经过这一步后self.feats中记录的是特征函数的索引值，以便后续预测时读取特征函数对应的权值
            self.feats[union_feature] = i
        # 权重序列初始化为全0，每一个特征函数都对应一个权重值，就是所求的系数
        self.w = [0.0] * len(self.feats)
        self.lastw = self.w

    def Ep(self):
        """
        计算Ep(fi) = sum (x,y) [p_(x)*p(y|X)fi(x,y)] 即《统计学习方法》P83页最上边的式子
        此过程涉及调用模型预测方法self.calprob，预测输出p(y|X)
        经验分布p_(x)刚好等于 1/训练集样本数，实际它的公式应该为p_(x)=v(X=x)/N
        """
        # 期望值的个数等于特征函数的个数
        ep = [0.0] * len(self.feats)
        # 遍历每个事件
        for record in self.trainset:
            # 分离出特征集
            features = record[1:]
            # 计算特征集的预测输出P(y|X)
            prob_li = self.calprob(features)
            # 遍历每一个特征值
            for x in features:
                # 遍历每个标记和其对应的概率值
                for prob, y in prob_li:
                    # 由标记和1个特征构建出1个特征函数fi(y,x)
                    if (y, x) in self.feats:
                        # 读取出特征函数的索引值
                        idx = self.feats[(y, x)]
                        # p_(x)的经验分布计算 p(x)=1/N
                        ep[idx] += prob * (1.0 / self.N)
        return ep

    def _convergence(self, lastw, w):
        """
        lastw:上一次迭代的系数向量
        w: 本次迭代的系数向量
        返回：True表示已经收敛；False表示未收敛
        """
        for w1, w2 in zip(lastw, w):
            if abs(w1 - w2) >= 0.01:
                return False
        return True

    #  训练过程
    def train(self, max_iter=1000):
        """
        最大熵分词模型训练过程
        :param max_iter: 模型训练的最大迭代次数
        :return:无
        """
        # 初始化模型参数
        self._initparams()
        # 开始迭代
        for i in range(max_iter):
            print('iter %d ...' % (i + 1))
            # 计算特征函数关于模型P(Y|X)和经验分布P(X)的期望值序列，每个特征函数对应一个值
            ep = self.Ep()
            # 保存当前的权值向量
            self.lastw = self.w[:]
            # 计算delta并更新权值向量
            for i, w in enumerate(self.w):
                # 计算delta
                delta = 1.0 / self.M * math.log(self.ep_[i] / ep[i])
                # 更新权值
                self.w[i] += delta
            # 检查是否收敛
            if self._convergence(self.lastw, self.w):
                break
    
    def probwgt(self, features, label):
        """
        features: 特征集
        label: 候选标记
        返回: P(y|X)的概率值
        功能：给定(X,y)计算条件概率P(y|X)
        """
        wgt = 0.0
        # 遍历每个特征f
        for f in features:
            # 如果训练集中见过特征函数(y,f)则读出权重
            if (label, f) in self.feats:
                # 累加权重
                wgt += self.w[self.feats[(label, f)]]
        # 取exp(权重和)作为预测值，这里没有进行归一化
        return math.exp(wgt)

    # 最大熵模型的预测函数P(y|x)，输入为1个事件中的特征x，输出为标记集中每种标记对应的归一化的概率P(y1|x),P(y2|x),...
    # 返回[(p1, y1),(p2,y2),...]
    def calprob(self, features):
        """
        features: 特征集
        返回：预测为标记集self.labels中每种标记的概率
        功能：给定特征集X,输出预测为标记集中每种标记的概率P(y|X)
        """
        # 计算标记集中每种标记的条件概率
        wgts = [(self.probwgt(features, l), l) for l in self.labels]
        # 使每个概率值相加后等于1
        Z = sum([w for w, l in wgts])
        prob = [(w / Z, l) for w, l in wgts]
        return prob

    def predict(self, text):
        """
        text: 输入的经过分词但未经词性标注的文本
        返回：词性标注后的文本
        功能：对经过分词的文本进行词性标注
        """
        word_li = []
        events_li = self.generate_events(text)
        for features in events_li:
            prob = self.calprob(features)
            prob.sort(reverse=True)
            word_li.append(u'%s/%s' % (features[1], prob[0]))
        return word_li


if __name__ == "__main__":
    maxent = MaxEnt()
    maxent.load_data("../data/199801.txt")
    maxent.train(50)
    word_li = maxent.predict("中国 政府 将 继续 坚持 奉行 独立自主 的 和平 外交 政策 。")
    print(word_li)
