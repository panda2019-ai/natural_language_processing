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
        # 词性标记集，表示Y可取的值的集合，比如名词、动词、形容词...
        self.labels = set()

    # 对一行文本抽取多个事件序列
    def generate_events(self, text, train_flag=False):
        # 定义事件序列，注意输入的1行文本可以抽取出多个事件
        event_li = []
        # 分词，要求输入的字符串中词语之间以空白分隔，对于训练集在词语后还要加词性标记：word/pos
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
        # 遍历中心词抽取1个event，每个event由1个词性标记和多个特征项构成
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
            # 因此1个事件对应3个特征函数：f1(fea_1,y),f2(fea_2,y),f3(fea_3,y)
            fields = [y, fea_3, fea_2, fea_1]
            # 将事件添加到事件序列
            event_li.append(fields)
        # 返回事件序列
        return event_li

    # 对一行文本抽取多个事件序列
    def generate_events2(self, text, train_flag=False):
        # 定义事件序列，注意输入的1行文本可以抽取出多个事件
        event_li = []
        # 分词，要求输入的字符串中词语之间以空白分隔，对于训练集在词语后还要加词性标记：word/pos
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
        # 遍历中心词抽取1个event，每个event由1个词性标记和多个特征项构成
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
            # 因此1个事件对应3个特征函数：f1(fea_1,y),f2(fea_2,y),f3(fea_3,y)
            fields = [y, fea_1, fea_2, fea_3]
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
                    # 抽取事件序列
                    events_li = self.generate_events(line, train_flag=True)
                    # 遍历每个事件，对每个事件构建3个特征函数fi(x,y)
                    for event in events_li:
                        # 第1列是输出词性标记y
                        label = event[0]
                        # 更新词性标记
                        self.labels.add(label)
                        # 对(词性标记，特征)计数，以便后续计算P(X,Y)的经验分布
                        for f in set(event[1:]):
                            self.feats[(label, f)] += 1
                        # 将event添加到训练集
                        self.trainset.append(event)

    # 初始化模型参数
    def _initparams(self):
        # 训练集中抽取的事件总数
        self.N = len(self.trainset)
        # 1个事件对应的特征数量为定值，因此1个事件对应的特征函数数量也为定值
        self.M = max([len(record) - 1 for record in self.trainset])
        # 计算每一种特征函数fi(x,y)关于经验分布P_(X,Y)的期望值，指的是《统计学习方法》P82页最下边的式子
        self.ep_ = [0.0] * len(self.feats)
        # 遍历每一个特征函数，注意这里的f对应的是(label,特征)这个元组
        for i, f in enumerate(self.feats):
            # 第i个特征函数的经验值=特征(x,y)出现次数/样本数量。注：这里认为一个event中没有重复的特征
            self.ep_[i] = float(self.feats[f]) / float(self.N)
            # 记录特征函数索引，注意这里的f对应的是(label,特征)这个元组，
            # 经过这一步后self.feats中记录的是特征函数的索引值，以便后续预测时读取特征函数对应的权值
            self.feats[f] = i
        # 权重序列初始化为全0，每一个特征函数都对应一个权重值，就是所求的系数
        self.w = [0.0] * len(self.feats)
        self.lastw = self.w

    # 计算每一种特征函数fi(x,y)关于模型与经验分布P_(X)的期望值，指的是《统计学习方法》P83页最上边的式子
    def Ep(self):
        # 期望值的个数等于特征函数的个数
        ep = [0.0] * len(self.feats)
        # 遍历每个事件，目的是遍历P83页式子中的所有x
        for record in self.trainset:
            # 分离出特征值
            features = record[1:]
            # 根据特征值计算对应的每种标记y的概率，实际上表示出式子中的P(y|x)
            # 显然针对一个事件，只有一个p(y|x)
            prob_li = self.calprob(features)
            # 遍历每一个特征值
            for f in features:
                # 遍历每个标记和其对应的概率值
                for prob, label in prob_li:
                    # 由标记和特征构建出1个特征函数
                    if (label, f) in self.feats:
                        # 读取出特征函数的索引值
                        idx = self.feats[(label, f)]
                        # P(x)的经验分布计算 p(x)=1/N
                        ep[idx] += prob * (1.0 / self.N)
        return ep

    # 检查权值向量是否收敛，收敛条件为W中的每个元素都不再发生变化，
    # 收敛返回True，否则返回False
    def _convergence(self, lastw, w):
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

    # 最大熵模型的预测函数P(y|x),第1个输入为1个事件中的特征x，可以对应3个特征函数f1(x,y),f2(x,y),f3(x,y)
    # features对应特征x=(x1,x2,x3)，第2个输入为label，它对应输出标记y，输出为P(y|x)的未归一化的概率值
    def probwgt(self, features, label):
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
        # 计算标记集中每种标记的条件概率
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
        events_li = self.generate_events2(text)
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
    word_li = maxent.predict("中国 领土 将 继续 坚持 奉行 独立自主 的 和平 外交 政策 。")
    print(word_li)



