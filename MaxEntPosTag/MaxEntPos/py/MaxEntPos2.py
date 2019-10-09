# encoding: utf-8
from collections import defaultdict
import math
import codecs


class MaxEnt(object):
    def __init__(self):
        # 存储特征函数关于经验分布f(x,y)的期望值 Ep_wan(fi)
        self.feats = defaultdict(int)
        # 文本经过特征抽取后建成的训练集每一项用一个事件表示(词性标记,特征1,特征2,特征3)
        self.trainset = []
        # 词性标记集
        self.labels = set()

    def generate_events(self, line, train_flag=False):
        """
        输入一个以空格为分隔符的已分词文本，返回生成的事件序列
        :param line: 以空格为分隔符的已分词文本
        :param train_flag: 真时为训练集生成事件序列；假时为测试集生成事件
        :return: 事件序列
        """
        event_li = []
        # 分词
        word_li = line.split()
        # 为词语序列添加头元素和尾元素，便于后续抽取事件
        if train_flag:
            word_li = [tuple(w.split(u'/')) for w in word_li if len(w.split(u'/')) == 2]
        else:
            word_li = [(w, u'x_pos') for w in word_li]
        word_li = [(u'pre1', u'pre1_pos')] + word_li + [(u'pro1', u'pro1_pos')]
        # 每个中心词抽取1个event，每个event由1个词性标记和多个特征项构成
        for i in range(1, len(word_li) - 1):
            # 特征函数a 中心词
            fea_1 = word_li[i][0]
            # 特征函数b 前一个词
            fea_2 = word_li[i - 1][0]
            # 特征函数d 下一个词
            fea_4 = word_li[i + 1][0]
            # 构建一个事件
            fields = [word_li[i][1], fea_1, fea_2, fea_4]
            # 将事件添加到事件序列
            event_li.append(fields)
        # 返回事件序列
        return event_li

    def load_data(self, file):
        with codecs.open('../data/199801.txt', 'rb', 'utf8', 'ignore') as infile:
            for line_ser, line in enumerate(infile):
                if line_ser >= 100:
                    break
                line = line.strip()
                if line:
                    # 生成事件序列
                    events_li = self.generate_events(line, train_flag=True)
                    for fields in events_li:
                        # 第1列是标记
                        label = fields[0]
                        # 添加标记
                        self.labels.add(label)
                        # 更新(标记，特征)计数
                        for f in set(fields[1:]):
                            # 更新p_wan(t, x)的数量，注意是（标记，特征）构成1种特征函数，而不是(标记,特征序列)构成1种特征函数
                            self.feats[(label, f)] += 1
                        # 存储event
                        self.trainset.append(fields)

    def _initparams(self):
        """
        初始化模型参数
        :return: 无
        """
        # 训练集中抽取的事件总数
        self.size = len(self.trainset)
        # 1个事件中所含特征数量的最大值，此示例中该特征数量为定值
        self.M = max([len(record) - 1 for record in self.trainset])
        # 期望值序列初始化，每一种特征函数都对应一个经验值
        self.ep_ = [0.0] * len(self.feats)
        # 遍历每一个特征函数
        for i, f in enumerate(self.feats):
            # 第i个特征函数的经验值=特征(x,y)出现次数/样本数量。注：这里认为一个event中没有重复的特征
            self.ep_[i] = float(self.feats[f]) / float(self.size)
            # 记录特征函数索引
            self.feats[f] = i
        # 权重序列初始化为全0，每一个特征函数都对应一个权重值。就是所求的系数
        self.w = [0.0] * len(self.feats)
        self.lastw = self.w

    def probwgt(self, features, label):
        """
        计算指数族的值
        :param features: 一个事件的特征序列
        :param label: 期待输出的标记
        :return: p(标记label|1个事件的特征序列)
        """
        wgt = 0.0
        for f in features:
            if (label, f) in self.feats:
                wgt += self.w[self.feats[(label, f)]]
        return math.exp(wgt)

    # calculate feature expectation on model distribution
    def Ep(self):
        ep = [0.0] * len(self.feats)
        for record in self.trainset:
            features = record[1:]
            # calculate p(y|x)
            prob = self.calprob(features)
            for f in features:
                for w, l in prob:
                    # only focus on features from training data.
                    if (l, f) in self.feats:
                        # get feature id
                        idx = self.feats[(l, f)]
                        # sum(1/N * f(y,x)*p(y|x)), p(x) = 1/N
                        ep[idx] += w * (1.0 / self.size)
        return ep

    def _convergence(self, lastw, w):
        """
        判断训练是否已经收敛，收敛条件为W中的每个元素都不再发生变化
        :param lastw: 原来的w序列值
        :param w: 当前的w序列值
        :return: 是否收敛的标志
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
        for i in range(max_iter):
            print('iter %d ...' % (i + 1))
            # 计算特征函数关于模型P(Y|X)和经验分布P(X)的期望值序列，每个特征函数对应一个值
            self.ep = self.Ep()
            self.lastw = self.w[:]
            for i, w in enumerate(self.w):
                delta = 1.0 / self.M * math.log(self.ep_[i] / self.ep[i])
                # 更新参数
                self.w[i] += delta
            # 检查是否收敛
            if self._convergence(self.lastw, self.w):
                break

    def calprob(self, features):
        """
        计算条件分布p(y|x)，即特征序列x关于每个可能的标记y的概率值
        :param features: 1个事件的多个特征序列
        :return:返回条件分布
        """
        # 计算事件中所有特征关于各标记的指数族的值
        wgts = [(self.probwgt(features, l), l) for l in self.labels]
        # 使每个概率值相加后等于1
        Z = sum([w for w, l in wgts])
        prob = [(w / Z, l) for w, l in wgts]
        return prob

    def predict(self, text):
        """
        对新文本进行词性标注
        :param text: 已分词的文本，词语之间以空格作为分隔
        :return: 返回词性标注的文本
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



