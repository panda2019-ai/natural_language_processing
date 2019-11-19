#include "postagger.h"

POS::POS() {
}

POS::~POS() {
}

void POS::init(string model_file, string punct_file) {
	 m.load_model(model_file);
	 //
	 ifstream in(punct_file.c_str());
	 string line, word;
	 while(getline(in, line)) {
		if(line != "" && line != "\t" && line != " ") {
			istringstream inword(line);
			while(inword >> word) {
				punct_set.insert(word);
			}
		}
	 }
}

string POS::run(vector<string>& cont) {
	vector<pair<string, double> > probs;
	probs.clear();
	probs = m.predict(cont);
	return probs[0].first;
}

size_t POS::char_seg(const string &input, vector<string> &output) {
    string ch;
    for (size_t i = 0, len = 0; i != input.length(); i += len) {  
        unsigned char byte = (unsigned)input[i];  
        if (byte >= 0xFC)       len = 6; // lenght 6
        else if (byte >= 0xF8)  len = 5;
        else if (byte >= 0xF0)  len = 4;
        else if (byte >= 0xE0)  len = 3;
        else if (byte >= 0xC0)  len = 2;
        else                    len = 1;
        ch = input.substr(i, len);  
        output.push_back(ch);  
    }
    return output.size();
}

string POS::line_predict(string pre_2, string pre_1, string curr, string aft_1, string aft_2) {
	size_t len;
	vector<string> context;
	//
	vector<string> tmp;
	len = char_seg(curr, tmp);
	context.push_back("word=" + curr);
	context.push_back("word_b=" + tmp[0]);
	context.push_back("word_e=" + tmp[tmp.size()-1]);
	//	
	tmp.clear();
	len = char_seg(pre_1, tmp);
	context.push_back("pre1=" + pre_1);
	context.push_back("pre1_b=" + tmp[0]);
	context.push_back("pre1_e=" + tmp[tmp.size()-1]);
	//	
	tmp.clear();
	len = char_seg(pre_2, tmp);
	context.push_back("pre2=" + pre_2);
	context.push_back("pre2_b=" + tmp[0]);
	context.push_back("pre2_e=" + tmp[tmp.size()-1]);
	//
	tmp.clear();
	len = char_seg(aft_1, tmp);
	context.push_back("aft1=" + aft_1);
	context.push_back("aft1_b=" + tmp[0]);
	context.push_back("aft1_e=" + tmp[tmp.size()-1]);
	//	
	tmp.clear();
	len = char_seg(aft_2, tmp);
	context.push_back("aft2=" + aft_2);
	context.push_back("aft2_b=" + tmp[0]);
	context.push_back("aft2_e=" + tmp[tmp.size()-1]);
	//
	return run(context);
}

string POS::postagger(vector<string> line_vec) {
	vector<string> predict_vec;
	predict_vec.clear();
	string predict_word = "";
	int line_size = line_vec.size();
	for(int i = 0; i < line_size; ++ i) {
		string curr = line_vec[i];
		if(punct_set.find(curr) != punct_set.end()) {
			predict_word = "w";
		} else {
			string pre_1 = "#";
			string pre_2 = "#";
			string aft_1 = "#";
			string aft_2 = "#";
			if(i >= 1) {pre_1 = line_vec[i - 1];}
			if(i >= 2) {pre_2 = line_vec[i - 2];}
			if(line_size - 1 - i >= 1) {aft_1 = line_vec[i + 1];}
			if(line_size - 1 - i >= 2) {aft_2 = line_vec[i + 2];}
			//
			predict_word = "";
			predict_word = line_predict(pre_2, pre_1, curr, aft_1, aft_2);
		}
		predict_vec.push_back(predict_word);
	}

	string result = line_vec[0] + "/" + predict_vec[0];
	for(int j = 1; j < line_size; ++ j) {
		result +=  " " + line_vec[j] + "/" + predict_vec[j];
	}

	return result;
}


int main(int argc, char* argv[]) {
	if(argc != 3) {
		cout << "demo: postagger <model_file> <punct_file>" << endl;
		exit(1);
	}
	string model_file = argv[1];
	string punct_file = argv[2];
	POS pos;
	pos.init(model_file, punct_file);
	string line, word;
	vector<string> temp;
	while(getline(cin, line)) {
		if(line == "" || line == " " || line == "\t") {
			continue;
		}
		temp.clear();
		istringstream inword(line);
		while(inword >> word) {
			temp.push_back(word);
		}
		cout << pos.postagger(temp) << endl;
	}

	return 0;
}