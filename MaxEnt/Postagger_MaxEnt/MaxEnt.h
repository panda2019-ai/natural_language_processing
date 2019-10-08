#ifndef _MAXENT_H_
#define _MAXENT_H_

#include <unordered_map>
#include "common.h"

using namespace std;

class Instance {
public:
    int label;              // 事件（类别），如Outdoor
    vector<int> fieldList;  // 事件发生的环境集合，如[Sunny, Happy]
    size_t count;           // 记录instance的发生次数
    //        
    Instance(int label, vector<int> fieldList) {
        this->label = label;
        this->fieldList = fieldList;
        sort(this->fieldList.begin(), this->fieldList.end());
        this->count = 1;
    }
    //
    bool operator == (const Instance &instance) {
        if (this->label != instance.label || this->fieldList.size() != instance.fieldList.size()) return false;
        for (size_t i = 0; i < this->fieldList.size(); ++i) {
            if (this->fieldList[i] != instance.fieldList[i]) return false;
        }
        return true;
    }
    //
    string toString() const {
        string line = itoa(label);
        for (size_t i = 0; i < fieldList.size(); ++i) {
            line = line + " " + itoa(fieldList[i]);
        }
        return line;
    }
};

/**
 * 特征(二值函数)
 */
class Feature {
public:
    int label;  // 事件，如Outdoor
    int value;  // 事件发生的环境，如Sunny
    /**
     * 特征函数
     * @param label 类别
     * @param value 环境
     */
    Feature(int label, int value) {
        this->label = label;
        this->value = value;
    }
    //
    bool operator == (const Feature &feature) {
        if (this->label == feature.label && this->value == feature.value) return true;
        return false;
    }
    //
    string toString() const {
        return itoa(label) + " " + itoa(value);
    }
};

/**
 * 最大熵的简明实现，提供训练与预测接口，训练算法采用LBFGS
 * @author zhanghuanlp@gmail.com
 */
class MaxEnt {
public:
    MaxEnt() {
        _l2reg = 1.0;
        _debug = 0;
        _thread_num = 12;
        _instances_total_num = 0;
        _cutoff = 1;
    }
    ~MaxEnt() {}

    /************* datas *************/
    vector <Instance>                              _instance_list;       // 样本数据集
    vector <Feature>                               _feature_list;        // 特征列表，来自所有事件的统计结果
    vector <int>                                   _feature_count_list;  // 每个特征的出现次数
    vector <string>                                _labels;              // 事件（类别）集
    vector <string>                                _fields;              // field 集

    unordered_map <string, int>                    _instance_index_map;  // _instances的索引
    unordered_map <string, int>                    _labels_index_map;    // _labels的索引
    unordered_map <string, int>                    _fields_index_map;    // _fields的索引
    unordered_map <int, unordered_map<int, int> >  _feature_index_map;   // _featureList的索引 <field, <label, val>>， val是feature在featureList中的位置
    unordered_map <int, vector<int> >              _field_to_labels;     // 记录每一个field关联的labels
    
    vector <double>                                _weight;              // 每个特征函数的权重
    vector <double>                                _gradient;            // 对每一个权重的偏导数
    vector <double>                                _modelE;              // 模型期望
    vector <double>                                _empiricalE;          // 经验期望
    double                                         _l2reg;               // 使用L2正则化
    double                                         _train_accurary;      // 训练数据的正确率
    int                                            _debug;               // 执行状态码
    int                                            _thread_num;          // 线程数
    size_t                                         _instances_total_num; // 训练实例总量
    size_t                                         _cutoff;              // 训练实例的频率阈值

    bool                                           _binary;              // 模型文件是否为二进制(存、读), 暂时不支持

    /************ methods ************/
	// 调试用
	void show();

    // 获得instance的编号(位置)
    int instance_index(string instance_str);
    // 获得feature的编号(位置)
    int feature_index(int label, int value);
    // 查询label的的编号
    int labels_index(string label);
    // 查询filed的编号
    int fields_index(string field);
    // 查询field是否有对应的labels列表
    int field_to_labels_index(int field);

    // 把一行string读到一个vector里
    string string_split(string line, vector<string> &v);
    // 加载数据，并填充一系列的数据结构
    void load_data(string path);

    // 为训练定制的prob计算，增加了和每一个label关联的features，用于后续的计算
    int cal_prob_train(vector<int> &fieldList, vector<double> &prob, vector<double> &weight, vector< vector<int> > &rel_feat);
    // 计算模型期望，即在当前的特征函数的权重下，计算特征函数的模型期望值
    double cal_modelE(vector<double> &weight);
    // 多线程版的 cal_modelE 的worker, 被cal_gradient直接调用
    static void * cal_modelE_worker(void * dat);
    //计算梯度，返回对数似然函数的负数，也就是把-logl作为了目标函数f，这个和最大熵是一致的
    double cal_gradient(vector<double> &weight, vector<double> &gradient);
    // 训练模型
    void train_lbfgs(size_t maxIt);
    // 检查是否收敛
    bool check_convergence(vector<double> &w1, vector<double> &w2);

    // 预测类别
    vector<pair<string, double> > predict(vector<string> &fieldList);
    // 计算p(y|x),此时的x指的是instance里的field,返回概率最大的label的index
    int cal_prob(vector<int> &fieldList, vector<double> &prob, vector<double> &weight);

    // 保存模型
    void save_model(string model_file);
    // 加载模型
    void load_model(string model_file);
};

#endif