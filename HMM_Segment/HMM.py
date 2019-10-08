#coding:utf-8
"""
基于隐马尔可夫模型的分词
"""
import codecs
import pickle
import numpy as np

idx_to_state = ['b', 'm', 'e', 's']
state_to_idx = {'b':0, 'm':1, 'e':2, 's':3}
# 极大似然估计法训练模型参数
def train():
    # 构建字表
    character_set = set()
    with codecs.open('RenMinData.txt_utf8', 'rb', 'utf-8', 'ignore') as infile:
        for line in infile:
            line = line.strip()
            if line:
                word_li = line.split()
                if word_li:
                    for word in word_li:
                        if word:
                            character_set.update(word)
    # 输出字表
    print('输出字表...')
    with open('character_list.txt', 'wb') as outfile:
        idx = 0
        for character in character_set:
            out_str = u'%d\t%s\n' % (idx, character)
            outfile.write(out_str.encode('utf-8', 'ignore'))
            idx += 1

    # 加载字索引以及索引字
    character_to_idx = dict()
    idx_to_character = []
    with codecs.open('character_list.txt', 'rb', 'utf-8', 'ignore') as infile:
        for line in infile:
            line = line.strip()
            if line:
                idx, character = line.split(u'\t')
                character_to_idx[character] = int(idx)
                idx_to_character.append(character)
    print('加载字索引表 长度=%d' % len(character_to_idx))
    print('加载索引字表 长度=%d' % len(idx_to_character))


    global state_li
    # 矩阵初始化
    print('初始化模型矩阵')
    A = np.zeros((len(idx_to_state), len(idx_to_state)))
    print('A=\n', A)
    B = np.zeros((len(idx_to_state), len(idx_to_character)))
    print('B.shape', B.shape)
    PI = np.zeros(len(idx_to_state))
    print('PI=', PI)

    # 训练
    with codecs.open('RenMinData.txt_utf8', 'rb', 'utf-8', 'ignore') as infile:
        for line_ser, line in enumerate(infile):
            line = line.strip()
            if line:
                # 对句子中的每个字打状态标记
                word_li = line.split()
                character_li = []
                if word_li:
                    for word in word_li:
                        if len(word) == 0:
                            continue
                        elif len(word) == 1:
                            character_li.append((word[0],'s'))
                        elif len(word) == 2:
                            character_li.append((word[0],'b'))
                            character_li.append((word[1], 'e'))
                        else:
                            character_li.append((word[0], 'b'))
                            character_li.extend([(w, 'm') for w in word[1:-1]])
                            character_li.append((word[-1], 'e'))
                # 统计相关次数
                # 更新PI列表
                PI[state_to_idx[character_li[0][1]]] += 1
                # 更新B字典
                for character, state in character_li:
                    B[state_to_idx[state], character_to_idx[character]] += 1
                # 更新A字典
                if len(character_li) >= 2:
                    for ser_num, cur_state in enumerate([w[1] for w in character_li[:-1]]):
                        next_state = character_li[ser_num+1][1]
                        cur_state_idx = state_to_idx[cur_state]
                        next_state_idx = state_to_idx[next_state]
                        A[cur_state_idx][next_state_idx] += 1

    # 计算PI
    PI = PI/sum(PI)
    # 计算A
    for row_ser in range(A.shape[0]):
        A[row_ser,::] /= sum(A[row_ser,::])
    # 计算B
    for row_ser in range(B.shape[0]):
        B[row_ser,::] /= sum(B[row_ser, ::])

    # 输出PI
    print("输出PI矩阵...")
    with open('model_PI', 'wb') as outfile:
        pickle.dump(PI, outfile)

    # 输出A
    print("输出A矩阵...")
    with open('model_A', 'wb') as outfile:
        pickle.dump(A, outfile)

    # 输出B
    print("输出B矩阵...")
    with open('model_B', 'wb') as outfile:
        pickle.dump(B, outfile)

def viterbe(O, PI, A, B, str=u''):
    """
    viterbi算法计算HMM问题2
    :param O:    观测序列
    :param PI:   初始状态分布
    :param A:    状态转移矩阵
    :param B:    发射矩阵
    :return:
    """

    # viterbi算法中初始化delta_1
    delta_1 = PI * B[:, 0]
    # viterbi算法中初始化kesi_1
    kesi_1 = np.zeros(PI.size, dtype=np.int)
    # 最优路径记录初始化
    kesi = np.array(kesi_1.T)

    # 递推
    delta_tplusone = delta_1.copy()
    for t in range(1, len(O)):
        max_delta_tminus_aji = np.tile(delta_tplusone, PI.size).reshape(A.shape).T * A
        delta_t = np.max(max_delta_tminus_aji, 0) * B[:, O[t]]
        kesi_t = np.argmax(max_delta_tminus_aji, 0)
        kesi = np.column_stack((kesi, kesi_t.T))
        delta_tplusone = delta_t.copy()

    # 终止
    P_star = np.max(delta_t)
    i_T_star = np.argmax(kesi_t)

    # 最优路径回溯
    I_star = [i_T_star]
    i_tplus_star = i_T_star
    for t in range(kesi.shape[1] - 1, 0, -1):
        i_t_star = kesi[:, t][i_tplus_star]
        I_star.append(i_t_star)
        i_tplus_star = i_t_star
    I_star = I_star[::-1]

    # 输出分词结果
    if str:
        out_str = u""
        state_li = [idx_to_state[w] for w in I_star]
        i = 0
        while i<len(state_li):
            if state_li[i] == 'b':
                j = i+1
                while j<len(state_li):
                    if state_li[j] not in ['m', 'e']:
                        break
                    j += 1
                out_str += str[i:j] + u'/'
                i = j
            else:
                out_str += str[i] + u'/'
                i+=1
        print(out_str)


if __name__ == '__main__':
    # 训练模型
    # train()
    # 加载模型
    PI = pickle.load(open('model_PI', 'rb'))
    A = pickle.load(open('model_A', 'rb'))
    B = pickle.load(open('model_B', 'rb'))
    # 加载字索引以及索引字
    character_to_idx = dict()
    idx_to_character = []
    with codecs.open('character_list.txt', 'rb', 'utf-8', 'ignore') as infile:
        for line in infile:
            line = line.strip()
            if line:
                idx, character = line.split(u'\t')
                character_to_idx[character] = int(idx)
                idx_to_character.append(character)
    print('加载字索引表 长度=%d' % len(character_to_idx))
    print('加载索引字表 长度=%d' % len(idx_to_character))
    # 测试
    # A = np.array([[0.5, 0.2, 0.3],
    #               [0.3, 0.5, 0.2],
    #               [0.2, 0.3, 0.5]])
    # B = np.array([[0.5, 0.5],
    #               [0.4, 0.6],
    #               [0.7, 0.3]])
    # PI = np.array([0.2, 0.4, 0.4])
    # O = [0, 1, 0]
    str = u'小明硕士毕业于中国科学院计算所'
    try:
        O = [character_to_idx[w] for w in str]
    except:
        print('有未登录字')
        exit(1)
    print(str)
    viterbe(O, PI, A, B, str)
