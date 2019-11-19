#include <iostream>
#include <string>
#include <ctime>
#include "MaxEnt.h"
#include "common.h"

int main(int argc, char** argv) {
    if (argc != 5) {
		cout << "Usage : " << argv[0] << " train_file model_file thread_num is_debug(0,1)" << endl;
		exit(1);
	}
    clock_t start, mid, end;
	string path = argv[1];
    string model_file = argv[2];
    int thread_num = atoi(argv[3]);
	int debug = atoi(argv[4]);
    MaxEnt maxEnt;
    maxEnt._thread_num = thread_num;
    maxEnt._debug = debug;
    start = clock();
    maxEnt.load_data(path);
    mid = clock();
	maxEnt.train_lbfgs(30);

    end = clock();
    maxEnt.save_model(model_file);
    cout << "加载数据用时 ：" << cal_time(start, mid) << "ms" << endl;
    cout << "模型训练用时 ：" << cal_time(mid, end) << "ms" << endl;
    return 0;
}