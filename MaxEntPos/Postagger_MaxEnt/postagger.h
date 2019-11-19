#ifndef _POSTAGGER_H_
#define _POSTAGGER_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>
#include <string>
#include "MaxEnt.h"

using namespace std;

class POS {
	public:
		POS();
		~POS();
		//
		void init(string model_file, string punct_file);
		string postagger(vector<string> line_vec);
	
	private:
		MaxEnt m;
		set<string> punct_set;
		//
		string line_predict(string pre_2, string pre_1, string curr, string aft_1, string aft_2);
		size_t char_seg(const string &input, vector<string> &output);
		string run(vector<string>& cont);
};

#endif
