#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <ctime>
#include "MaxEnt.h"
#include "common.h"

int main(int argc, char** argv) {
    if (argc != 4) {
		cout << "Usage : " << argv[0] << " test_file model_file output_file" << endl;
		exit(1);
	}
    clock_t start, mid, end;
	string path = argv[1];
    string model_file = argv[2];
    string output_file = argv[3];
    MaxEnt maxEnt;
    
    start = clock();
    maxEnt.load_model(model_file);
    mid = clock();

    size_t n = 0, total_line = 0, ncorrect = 0;
    string line, word, gold_ans, model_ans;
    vector< pair<string, double> > result;
    vector<string> cur_line;
    ifstream in(path.c_str());
    ofstream out(output_file.c_str());
    while(getline(in, line)) {
        if (line != "" && line != " " && line != "\t") {
            n = 0;
            cur_line.clear();
            result.clear();
            istringstream inword(line);
            while(inword >> word) {
                n ++;
                if (n == 1) {
                    gold_ans = word;
                } else {
                    cur_line.push_back(word);
                }
            }
            result = maxEnt.predict(cur_line);
            model_ans = result[0].first;
            total_line ++;
            if (gold_ans == model_ans) ncorrect ++;
            out << model_ans << " " << line << endl;
        }
    }
    out << "Total    : " << total_line << endl;
    out << "Correct  : " << ncorrect << endl;
    out << "Accuracy : " << (double)ncorrect/total_line << endl;
    in.close();
    out.close();
    end = clock();
    cout << "加载模型用时 ：" << cal_time(start, mid) << "ms" << endl;
    cout << "模型测试用时 ：" << cal_time(mid, end) << "ms" << endl;
    return 0;
}