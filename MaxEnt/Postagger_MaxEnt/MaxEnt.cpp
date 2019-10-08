#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <pthread.h>
#include "lbfgs.h"
#include "MaxEnt.h"
#include "IO.h"
#include "common.h"

using namespace std;
using namespace CRFPP;

typedef struct mt_share_data {
    MaxEnt *maxEnt_obj;
    int thread_id;
    int thread_ncorrect;
    double thread_logl;
    vector<double> thread_modelE;
} Mt_Share_Data;

// 获得instance的编号(位置)，加载数据时，检测重复使用
int MaxEnt::instance_index(string instance_str) {
    int index = -1;
    if (_instance_index_map.find(instance_str) != _instance_index_map.end()) {
        index = _instance_index_map[instance_str];
    }
    return index;
}

/**
 * 获得feature的编号(位置)
 * @param label 的编号   value field的编号
 */
int MaxEnt::feature_index(int label, int value) {
    int index = -1;
    if (_feature_index_map.find(value) != _feature_index_map.end()) {
        if (_feature_index_map[value].find(label) != _feature_index_map[value].end()) {
            index = _feature_index_map[value][label];
        }
    }
    return index;
}

// 查询field是否有对应的labels列表
int MaxEnt::field_to_labels_index(int field) {
    int index = -1;
    if (_field_to_labels.find(field) != _field_to_labels.end()) {
        index = 1;
    }
    return index;
}

// 查询label的的编号
int MaxEnt::labels_index(string label) {
    int index = -1;
    if (_labels_index_map.find(label) != _labels_index_map.end()) {
        index = _labels_index_map[label];
    }
    return index;
}

// 查询field的编号
int MaxEnt::fields_index(string field) {
    int index = -1;
    if (_fields_index_map.find(field) != _fields_index_map.end()) {
        index = _fields_index_map[field];
    }
    return index;
}

// 把一行string读到一个vector里
string MaxEnt::string_split(string line, vector<string> &v) {
    string label;
    v.clear();
    string word;
    int n = 0;
    istringstream inword(line);
    while (inword >> word) {
        if (word != "" && word != " " && word != "\t") {
            n++;
            if (n > 1) {
                if (find(v.begin(), v.end(), word) == v.end()) {
                    v.push_back(word);  //同一行做了特征去重
                }
            } else if (n == 1) label = word;
        }
    }
    inword.clear();
    return label;
}

/**
 * 加载数据，并且创建如下域
 * featureList：特征函数的list
 * featureCountList:与特征函数一一对应的，特征函数出现的次数
 * instanceList:样本数据list
 * labels:类别list
 *
 * @param path
 */
void MaxEnt::load_data(string path) {
    Mmap_IO f_io(path);
    size_t i;
    int index, label_index, field_index, ins_index;
    string line, label, field, ins_str;
    vector<string> fieldList;
    vector<int> fieldList_num;
    //
    cout << "Loading training events from " << path << endl;
    clock_t load_data_start, load_data_end;
    //
    load_data_start = clock();
    while (f_io.get_line(line)) {
        if (line == "" || line == " " || line == "\t") continue;
        label = string_split(line, fieldList);
        if (fieldList.size() <= 0) continue;
        label_index = labels_index(label);
        if (label_index == -1) {
            label_index = _labels.size();
            _labels.push_back(label);
            _labels_index_map[label] = label_index;
        }
        fieldList_num.clear();
        for (i = 0; i < fieldList.size(); ++i) {
            field = fieldList[i];
            field_index = fields_index(field);
            if (field_index == -1) {
                field_index = _fields.size();
                _fields.push_back(field);
                _fields_index_map[field] = field_index;
            }
            fieldList_num.push_back(field_index);
            index = feature_index(label_index, field_index);
            if (index == -1) {
                Feature feature(label_index, field_index);
                _feature_list.push_back(feature);
                _feature_count_list.push_back(1);
                _feature_index_map[field_index][label_index] = _feature_list.size() - 1;  //添加feature_index
                _field_to_labels[field_index].push_back(label_index);  //特征不存在，则field到label的映射就不存在，相反亦然
            } else {
                _feature_count_list[index] ++;
            }
        }
        Instance instance(label_index, fieldList_num);
        ins_str = instance.toString();
        ins_index = instance_index(ins_str);
        if (ins_index == -1) {
            _instance_list.push_back(instance);
            _instance_index_map[ins_str] = _instance_list.size() - 1;
        } else {
            _instance_list[ins_index].count ++;
        }
        
        _instances_total_num ++;
        if (_instances_total_num % 1000 == 0) {
            printf("%csamples: %zd", 13, _instances_total_num);
        }
    }
    printf("%cSamples: %zd\n", 13, _instances_total_num);
    load_data_end = clock();
    cout << "Total " << _instances_total_num << " training events and " << "0 heldout events added in " << cal_time(load_data_start, load_data_end) << "ms" << endl;    
    cout << "Reducing events (cutoff is " << _cutoff << ")..." << endl;
    cout << "Reduced to " << _instance_list.size() << " training events" << endl;
    cout << "Total " << _feature_list.size() << " features" << endl;
    // 释放一些后续用不到的资源
    _labels_index_map.clear();
    _instance_index_map.clear();
    return;
}

/**
 * 计算p(y|x),此时的x指的是instance里的field
 * @param fieldList 实例的特征列表
 * @return 该实例属于每个类别的概率
 */
int MaxEnt::cal_prob_train(vector<int> &fieldList, vector<double> &prob, vector<double> &weight, vector< vector<int> > &rel_feat) {
    size_t i, j;
    rel_feat.clear();
    vector<int> tmp_v;
    for (i = 0; i < _labels.size(); ++i) {
        rel_feat.push_back(tmp_v);
    }
    int index, f2l_index, label_index, field_index;
    prob.clear();
    for (i = 0; i < _labels.size(); ++i) prob.push_back(0.0);
    vector<int> labels_index_vec;
    double sum = 0;  // 正则化因子，保证概率和为1
    for (i = 0; i < fieldList.size(); ++i) {
        field_index = fieldList[i];
        f2l_index = field_to_labels_index(field_index);
        if (f2l_index != -1) {
            labels_index_vec = _field_to_labels[field_index];
            for (j = 0; j < labels_index_vec.size(); ++j) {
                label_index = labels_index_vec[j];
                index = feature_index(label_index, field_index);
                if (index != -1) {
                    prob[label_index] += weight[index];
                    rel_feat[label_index].push_back(index);
                }
            }
        }
    }
    // 取偏移量，防止溢出，将所有的数都归到（-inf, 0）
    double curr_prob_max = VecMaxElement(prob);
    double offset = curr_prob_max;
    //
    double tmp;
    double max_prob = 0.0;
    int max_prob_pos = -1;
    for (i = 0; i < prob.size(); ++i) {
        tmp = prob[i] - offset;
        prob[i] = exp1024(tmp);
        if (prob[i] != 0) {
            sum += prob[i];
            if (prob[i] > max_prob) {
                max_prob = prob[i];
                max_prob_pos = i;
            }
        }
    }
    //
    if (sum != 0) {
        for (i = 0; i < prob.size(); ++i) prob[i] /= sum;
    }
    return max_prob_pos;
}

/**
 * 计算模型期望，即在当前的特征函数的权重下，计算特征函数的模型期望值。
 * @param weight特征权重，直接更新了类的_modelE
 * @return loglikelihood，有时要考虑正则化项
 */
double MaxEnt::cal_modelE(vector<double> &weight) {
    double logl = 0;
    int ncorrect = 0;
    int max_label_index = -1;

    size_t i, j, k;
    size_t instanceList_size = _instance_list.size();
    vector<int> fieldList;
    vector<double> pro;
    vector< vector<int> > rel_feat;
    for (i = 0; i < _modelE.size(); ++i) _modelE[i] = 0.0;
    for (i = 0; i < instanceList_size; ++i) {
        fieldList.clear();
        fieldList = _instance_list[i].fieldList;
        //计算当前样本X对应所有类别的概率
        pro.clear();
        rel_feat.clear();
        max_label_index = cal_prob_train(fieldList, pro, weight, rel_feat);
        if (max_label_index >= 0) logl += _instance_list[i].count * fast_log(pro[_instance_list[i].label]);
        if (max_label_index == _instance_list[i].label) ncorrect ++;
        for (j = 0; j < rel_feat.size(); ++j) {
            for (k = 0; k < rel_feat[j].size(); ++k) {
                _modelE[rel_feat[j][k]] += _instance_list[i].count * pro[j];
            }
        }
    }
    for (i = 0; i < _modelE.size(); ++i) {
        _modelE[i] /= (double)_instances_total_num;
    }
    
    _train_accurary = (double)ncorrect / (double)instanceList_size;
    logl /= (double)_instances_total_num;
    if (_l2reg > 0) {
        const double c = _l2reg;
        for (i = 0; i < _feature_list.size(); ++i) {
            logl -= weight[i] * weight[i] * c;
        }
    }
    return logl;
}

void * MaxEnt::cal_modelE_worker(void * dat) {
    Mt_Share_Data * share_data = (Mt_Share_Data *)dat;
    MaxEnt &obj = *share_data->maxEnt_obj;
    double logl = 0;
    int ncorrect = 0;
    int max_label_index = -1;

    size_t i, j, k;
    size_t instanceList_size = obj._instance_list.size();
    int thread_id = share_data->thread_id;
    size_t begin_pos, end_pos;
    if (instanceList_size > obj._thread_num) {
        int part_size = instanceList_size/obj._thread_num;
        begin_pos = thread_id * part_size;
        if (thread_id == obj._thread_num - 1) end_pos = instanceList_size;
        else end_pos = begin_pos + part_size;
    } else {
        if (thread_id < instanceList_size) {
            begin_pos = thread_id;
            end_pos = begin_pos + 1;
        } else begin_pos = end_pos = 0;
    }

    vector<int> fieldList;
    vector<double> pro;
    vector< vector<int> > rel_feat;
    vector<double> loc_modelE(obj._modelE.size(), 0.0);
    for (i = begin_pos; i < end_pos; ++i) {
        fieldList.clear();
        fieldList = obj._instance_list[i].fieldList;
        //计算当前样本X对应所有类别的概率
        pro.clear();
        rel_feat.clear();
        max_label_index = obj.cal_prob_train(fieldList, pro, obj._weight, rel_feat);
        if (max_label_index >= 0) logl += obj._instance_list[i].count * fast_log(pro[obj._instance_list[i].label]);
        if (max_label_index == obj._instance_list[i].label) ncorrect ++;
        for (j = 0; j < rel_feat.size(); ++j) {
            for (k = 0; k < rel_feat[j].size(); ++k) {
                loc_modelE[rel_feat[j][k]] += obj._instance_list[i].count * pro[j];
            }
        }
    }
    
    share_data->thread_modelE = loc_modelE;
    share_data->thread_ncorrect = ncorrect;
    share_data->thread_logl = logl;
    return 0;
}

// 计算梯度，返回对数似然函数的负数，也就是把-logl作为了目标函数f，这个和最大熵是一致的
double MaxEnt::cal_gradient(vector<double> &weight, vector<double> &gradient) {
    size_t i, j;
    // 多线程计算modelE
    vector<Mt_Share_Data> pid_data_list(_thread_num);
    vector<pthread_t> pid_list(_thread_num);
    for (i = 0; i < _thread_num; ++ i) {
        pid_data_list[i].maxEnt_obj = this;
        pid_data_list[i].thread_id = i;
    }
    for (i = 0; i < _thread_num; ++ i) pthread_create(&(pid_list[i]), NULL, cal_modelE_worker, &(pid_data_list[i]));
    for (i = 0; i < _thread_num; ++ i) pthread_join(pid_list[i], NULL);
    // 多各个线上返回的数据做合并计算
    double logl = 0;
    int ncorrect = 0;
    size_t instanceList_size = _instance_list.size();
    for (i = 0; i < _modelE.size(); ++i) _modelE[i] = 0.0;
    for (i = 0; i < _thread_num; ++ i) {
        logl += pid_data_list[i].thread_logl;
        ncorrect += pid_data_list[i].thread_ncorrect;
        for (j = 0; j < _modelE.size(); ++j) _modelE[j] += pid_data_list[i].thread_modelE[j];
    }
    _train_accurary = (double)ncorrect / (double)instanceList_size;
    logl /= (double)_instances_total_num;
    if (_l2reg > 0) {
        const double c = _l2reg;
        for (i = 0; i < _feature_list.size(); ++i) {
            logl -= _weight[i] * _weight[i] * c;
        }
    }
    for (i = 0; i < _modelE.size(); ++i) _modelE[i] /= (double)_instances_total_num;
    
    //多线程时使用上面的一堆，单线程时只用这一句就可以了
//    double logl = cal_modelE(weight);
    if (_l2reg == 0) {
        for (i = 0; i < weight.size(); ++i) {
            gradient[i] = -(_empiricalE[i] - _modelE[i]);
        }
    } else {
        const double c = _l2reg * 2;
        for (i = 0; i < weight.size(); ++i) {
            gradient[i] = -(_empiricalE[i] - _modelE[i] - c * weight[i]);
        }
    }
    return -logl;
}


/**
 * 训练模型
 * @param maxIt 最大迭代次数
 */
void MaxEnt::train_lbfgs(size_t maxIt) {
    _l2reg /= _instances_total_num;
    size_t i, j;
    size_t size = _feature_list.size();
    _weight.clear();                         // 特征权重
    for (i = 0; i < size; ++i) _weight.push_back(0.0);
    _gradient.clear();                       //梯度
    for (i = 0; i < size; ++i) _gradient.push_back(0.0);
    _modelE.clear();       //模型期望
    _empiricalE.clear();   //经验期望
    for (i = 0; i < size; ++i) {
        _modelE.push_back(0.0);
        _empiricalE.push_back((double)_feature_count_list[i] / (double)_instances_total_num);
    }
    if (_debug) {
        cout << "=====empiricalE=====" << endl;
        for (i = 0; i < size; ++i) cout << _empiricalE[i] << endl;
    }

    vector<double> lastWeight(size, 0.0);   //上次迭代的权重
    vector<double> delta;
    double f = cal_gradient(_weight, _gradient);   //计算初始点的梯度， 并且返回目标函数-logl(考虑正则化项)
	clock_t lbfgs_start, lbfgs_end;
	clock_t cal_f_g_start, cal_f_g_end;
    cout << "Starting L-BFGS iterations..." << endl;
    cout << "Number of Predicates:  " << endl;
    cout << "Number of Outcomes:    " << _labels.size() << endl;
    cout << "Number of Parameters:  " << _feature_list.size() << endl;
    cout << "Number of Corrections: " << endl;
    cout << "Tolerance:             " << 1.000000E-05 << endl;
    cout << "Optimized version" << endl;
    cout << "iter     eval     loglikelihood   training accuracy   heldout accuracy     time(ms)" << endl;
    cout << "===================================================================================" << endl;
    LBFGS lbfgs;
    for (i = 0; i < maxIt; ++i) {
        for (j = 0; j < _weight.size(); ++j) {
            lastWeight[j] = _weight[j];
        }
		lbfgs_start = clock();
        lbfgs.optimize(_weight.size(), &_weight[0], f, &_gradient[0], false, 1.0);  //猜测：倒数第二个参数 true-L1rg false-L2rg
        lbfgs_end = clock();
		if (check_convergence(lastWeight, _weight)) {
            cout << "end : 2" << endl;
            break;
        }
		cal_f_g_start = clock();
        f = cal_gradient(_weight, _gradient);   //计算f = -logl, 同时更新梯
		cal_f_g_end = clock();
		if (Norm(_gradient) < 0.00001) {
			cout << "end : 1" << endl;
			break;
		}
        double lbfgs_time = cal_time(lbfgs_start, lbfgs_end);
		double cal_f_g_time = cal_time(cal_f_g_start, cal_f_g_end)/_thread_num;
        double time_c = cal_f_g_time + lbfgs_time;
		fprintf(stdout, "%4zd              %6.4f         %1.4f                                  %6.2f    %6.2f    %6.2f\n", i + 1, -f, _train_accurary, time_c, lbfgs_time, cal_f_g_time);
    }
    return;
}

/**
 * 检查是否收敛
 * @param w1
 * @param w2
 * @return 是否收敛
 */
bool MaxEnt::check_convergence(vector<double> &w1, vector<double> &w2) {
    for (size_t i = 0; i < w1.size(); ++i) {
        if (fabs(w1[i] - w2[i]) >= (double)1/pow(10, 9)) return false;   // 收敛阀值0.01可自行调整
    }
    return true;
}

/**
 * 预测类别
 * @param fieldList
 * @return
 */
vector<pair<string, double> > MaxEnt::predict(vector<string> &fieldList) {
    size_t i;
    vector<double> prob;
    vector<int> fieldList_num;
    int index;
    for (i = 0; i < fieldList.size(); ++i) {
        index = fields_index(fieldList[i]);
        if (index != -1) fieldList_num.push_back(index);
    }
    cal_prob(fieldList_num, prob, _weight);
    vector< pair<string, double> > pairResult;  
    for (i = 0; i < prob.size(); ++i) {
        pairResult.push_back(make_pair(_labels[i], prob[i]));
    }
    sort(pairResult.begin(), pairResult.end(), pair_compare);
    return pairResult;
}

/**
 * 计算p(y|x),此时的x指的是instance里的field
 * @param fieldList 实例的特征列表
 * @return 该实例属于每个类别的概率
 */
int MaxEnt::cal_prob(vector<int> &fieldList, vector<double> &prob, vector<double> &weight) {
    size_t i, j;
    int index, f2l_index, label_index, field_index;
    prob.clear();
    for (i = 0; i < _labels.size(); ++i) prob.push_back(0.0);
    vector<int> labels_index_vec;
    double sum = 0;  // 正则化因子，保证概率和为1
    for (i = 0; i < fieldList.size(); ++i) {
        field_index = fieldList[i];
        f2l_index = field_to_labels_index(field_index);
        if (f2l_index != -1) {
            labels_index_vec = _field_to_labels[field_index];
            for (j = 0; j < labels_index_vec.size(); ++j) {
                label_index = labels_index_vec[j];
                index = feature_index(label_index, field_index);
                if (index != -1) {
                    prob[label_index] += weight[index];
                }
            }
        }
    }
    // 取偏移量，防止溢出，将所有的数都归到（-inf, 0）
    double curr_prob_max = VecMaxElement(prob);
    double offset = curr_prob_max;
    //
    double tmp;
    double max_prob = 0.0;
    int max_prob_pos = -1;
    for (i = 0; i < prob.size(); ++i) {
        tmp = prob[i] - offset;
        prob[i] = exp1024(tmp);
        if (prob[i] != 0) {
            sum += prob[i];
            if (prob[i] > max_prob) {
                max_prob = prob[i];
                max_prob_pos = i;
            }
        }
    }
    //
    if (sum != 0) {
        for (i = 0; i < prob.size(); ++i) prob[i] /= sum;
    }
    return max_prob_pos;
}

/**
 * 保存模型，目前使用zhangle的格式，可以兼容
 */
void MaxEnt::save_model(string model_file) {
    size_t i, j;
    ofstream out(model_file.c_str());
    out.precision(20);
    //
    size_t fields_size = _fields.size();
    size_t labels_size = _labels.size();
    //
    out << "#txt,maxent" << endl;
    out << fields_size << endl;
    for (i = 0; i < fields_size; ++i) out << _fields[i] << endl;
    out << labels_size << endl;
    for (i = 0; i < labels_size; ++i) out << _labels[i] << endl;
    for (i = 0; i < fields_size; ++i) {
        out << _field_to_labels[i].size();
        sort(_field_to_labels[i].begin(), _field_to_labels[i].end());
        for (j = 0; j < _field_to_labels[i].size(); ++j) out << " " << _field_to_labels[i][j];
        out << endl;
    }
    size_t field_index, label_index;
    out << _weight.size() << endl;
    for (i = 0; i < fields_size; ++i) {
        field_index = i;
        for (j = 0; j < _field_to_labels[i].size(); ++j) {
            label_index = _field_to_labels[i][j];
            out << _weight[_feature_index_map[field_index][label_index]] << endl;
        }
    }
    out.close();
}

/**
 * 加载模型
 * 填充
 * vector <string>                                _labels;              // 事件（类别）集
 * vector <string>                                _fields;              // field 集
 * unordered_map <int, unordered_map<int, int> >  _feature_index_map;   // _featureList的索引 <field, <label, val>>， val是feature在featureList中的位置
 * unordered_map <int, vector<int> >              _field_to_labels;     // 记录每一个field关联的labels
 * unordered_map <string, int>                    _fields_index_map;    // _fields的索引
 * vector <double>                                _weight;              // 每个特征函数的权重
 */
void MaxEnt::load_model(string model_file) {
    _labels.clear();
    _fields.clear();
    _fields_index_map.clear();
    _feature_index_map.clear();
    _field_to_labels.clear();
    _weight.clear();
    //
    size_t i;
    Mmap_IO f_io(model_file);
    string line;
    size_t line_num = 0, fields_size = 0, labels_size = 0, weights_size = 0;
    string field_index_str;
    vector<string> field_to_labels_index_str;
    int field_index = 0, label_index;
    size_t feature_num = 0;
    while (f_io.get_line(line)) {
        line_num ++;
        if (line_num == 2) {
            fields_size = atoi(line.c_str());
        } else if (line_num > 2 && line_num <= 2 + fields_size) {
            _fields.push_back(line);
            _fields_index_map[line] = _fields.size() - 1;
        } else if (line_num == 3 + fields_size) {
            labels_size = atoi(line.c_str());  
        } else if (line_num > 3 + fields_size && line_num <= 3 + fields_size + labels_size) {
            _labels.push_back(line);
        } else if (line_num > 3 + fields_size + labels_size && line_num <= 3 + fields_size + labels_size + fields_size) {
            field_index_str = string_split(line, field_to_labels_index_str);
            for (i = 0; i < field_to_labels_index_str.size(); ++i) {
                label_index = atoi(field_to_labels_index_str[i].c_str());
                _feature_index_map[field_index][label_index] = feature_num;
                feature_num ++;
                _field_to_labels[field_index].push_back(label_index);
            }
            field_index ++;
        } else if (line_num == 4 + fields_size + labels_size + fields_size) {
            weights_size = atoi(line.c_str());
        } else if (line_num > 4 + fields_size + labels_size + fields_size && line_num <= 4 + fields_size + labels_size + fields_size + weights_size) {
            _weight.push_back(atof(line.c_str()));
        }
    }
    return;
}

// 调试用
void MaxEnt::show() {
    size_t i = 0;
    cout << "-----labels-----" << endl;
    for (i = 0; i < _labels.size(); ++i) cout << _labels[i] << endl;
    cout << "-----instances-----" << endl;
    for (i = 0; i < _instance_list.size(); ++i) cout << _instance_list[i].toString() << endl;
    cout << "-----features-----" << endl;
    for (i = 0; i < _feature_list.size(); ++i) cout << _feature_list[i].toString() << "\t" << _feature_count_list[i] << endl;
    cout << "-----weight-----" << endl;
    for (i = 0; i < _weight.size(); ++i) cout << _weight[i] << endl;
    return;
}