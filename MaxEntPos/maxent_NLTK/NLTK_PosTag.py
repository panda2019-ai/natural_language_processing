# coding:utf-8
"""
NLTK的最大熵模型实现词性标注
"""

import nltk
import codecs


# 事件生成器，1个分词后的句子可以产生多个事件
def generate_events(word_li):
    events = []
    # 为词语序列添加头元素和尾元素，便于后续抽取事件
    word_li = [(u'pre1', u'pre1_pos')] + word_li + [(u'pro1', u'pro1_pos')]

    # 每个中心词抽取1个event，每个event由1个词性标记和多个特征项构成
    for i in range(1, len(word_li) - 1):
        # 定义特征词典
        features_dict = dict()
        # 特征函数a 中心词
        features_dict['fea_1'] = word_li[i][0]
        # 特征函数b 前一个词
        features_dict['fea_2'] = fea_2 = word_li[i - 1][0]
        # 特征函数d 下一个词
        features_dict['fea_4'] = word_li[i + 1][0]
        # 标记
        label = word_li[i][1]
        # 添加一个事件
        events.append((features_dict, label))

    return events


# 加载数据，生成事件集，返回
def load_data(file_name):
    data_set = []
    with codecs.open(file_name, 'rb', 'utf8', 'ignore') as infile:
        for line_ser, line in enumerate(infile):
            if line_ser >= 100:
                break
            line = line.strip()
            if line:
                word_li = line.split()
                word_li = [tuple(w.split(u'/')) for w in word_li if len(w.split(u'/')) == 2]
                # 生成事件并更新到data_set
                data_set.extend(generate_events(word_li))
    print("抽取出 %d 个事件" % len(data_set))
    return data_set


# 标注词性
def pos_tag(classifier, line):
    new_word_li = []
    word_li = line.split()
    word_li = [(w, u'x_pos') for w in word_li]
    events_li = generate_events(word_li)
    for i, (features, label) in enumerate(events_li):
        predict_pos = classifier.classify(features)
        new_word_li.append(word_li[i][0]+"/"+predict_pos)
    return new_word_li


if __name__ == "__main__":
    # 抽取特征构建训练和测试集
    data_set = load_data('data/199801.txt')
    train_size = int(len(data_set)*0.8)
    test_size = int(len(data_set)*0.2)
    train_set = data_set[:train_size]
    test_set = data_set[-test_size:]
    print("训练集事件数=", len(train_set))
    print("测试集事件数=", len(test_set))
    # GIS学习算法的最大熵模型
    classifier_gis = nltk.classify.maxent.MaxentClassifier.train(train_set,  trace=2, algorithm='gis', max_iter=10)
    print("GIS模型准确率= ", nltk.classify.accuracy(classifier_gis, test_set))
    print("GIS模型测试：")
    print(pos_tag(classifier_gis, "中国 政府 将 继续 坚持 奉行 独立自主 的 和平 外交 政策 。"))
    # IIS学习算法的最大熵模型
    classifier_iis = nltk.classify.maxent.MaxentClassifier.train(train_set, trace=2, algorithm='iis', max_iter=10)
    print("IIS模型准确率= ", nltk.classify.accuracy(classifier_iis, test_set))
    print("IIS模型测试：")
    print(pos_tag(classifier_iis, "中国 政府 将 继续 坚持 奉行 独立自主 的 和平 外交 政策 。"))
