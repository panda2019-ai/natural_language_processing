#ifndef _MAXENT_COMMON_H_
#define _MAXENT_COMMON_H_

#include <vector>
#include <string>
#include <sstream>

using namespace std;

//用于最终预测结果的排序，降序排列
inline bool pair_compare(const pair<string, double> &a, const pair<string, double> &b) {
    if (a.second > b.second) return true;
    return false;
}

//计算时间间隔，单位为ms
inline double cal_time(clock_t start, clock_t end) {
	double time_c = (double)(end - start + 1) / (double)CLOCKS_PER_SEC * 1000;
	return time_c;
}

//数字转字符串
inline string itoa (int i) {
    stringstream ss;
    ss << i;
    string str = ss.str();
    ss.clear();
    return str;
}

inline string ftoa (double f) {
    stringstream ss;
    ss << f;
    string str = ss.str();
    ss.clear();
    return str;
}

/**
 * 下面的两个函数提速约1/10，且速度的提升是由inline带来的
 */
//快速的log近似实现
inline float mFast_Log2(float val) {
	union { float val; int32_t x; } u = { val };
    float log_2 = (float)(((u.x >> 23) & 255) - 128);
	u.x   &= ~(255 << 23);
	u.x   += 127 << 23;
	log_2 += ((-0.3358287811f) * u.val + 2.0f) * u.val  -0.65871759316667f;
	return (log_2);
}

inline float fast_log(const float &val) {
   return (mFast_Log2(val) * 0.69314718f);
}

//快速的exp近似实现
inline double exp256(double x) {
	x = 1.0 + x / 256.0;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x; x *= x; x *= x;
	return x;
}

inline double exp1024(double x) {
	x = 1.0 + x / 1024;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x;
	return x;
}

// 得到两个数中大的一个
inline double Max(double x, double y) {
    if (x >=y) return x;
    return y;
}

// 计算点乘积 vector1^T * vector2
inline double VecDotProduct(vector<double> &vector1, vector<double> &vector2) {
    if (vector1.size() != vector2.size()) {
        cerr << "vector1和vector2的长度不一致" << endl;
        exit(1);
    }
    double result = 0.0;
    for (size_t i = 0; i < vector1.size(); ++i) {
        result += vector1[i] * vector2[i];
    }
    return result;
}

// 1-模
inline double Norm(vector<double> &v) {
    double result = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        result += v[i] * v[i];
    }
    return sqrt(result);
}

// 得到vector中的最大元素的值
inline double VecMaxElement(vector<double> &v) {
    double max = v[0];
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > max) max = v[i];
    }
    return max;
}

#endif
