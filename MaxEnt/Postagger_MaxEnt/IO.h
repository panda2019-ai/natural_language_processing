#ifndef _MAXENT_IO_H_
#define _MAXENT_IO_H_

#include <string>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>

using namespace std;

#define MAXS 60*1024*1024
char buf[MAXS];

class Mmap_IO {
public:
	Mmap_IO(string file) {
		this->mbuf = 0;
		this->len = 0;
		this->cur_len = 0;
		this->fd = 0;
		mmap_init(file);
	}
	~Mmap_IO() {
		munmap(this->mbuf, this->len);
		close(this->fd);
	}

	bool get_line(string &line) {
		return analyse(line);
	}

private:
	char *mbuf;
	size_t len;
	size_t cur_len;
	size_t fd;
	//
	bool analyse(string &line) {
		size_t i;
		line = "";
		buf[i=0]='\0';
		size_t flag = 0;
		char *p = this->mbuf + this->cur_len;
		for ( ; *p && this->cur_len < this->len; p++) {
			flag ++;
			this->cur_len ++;
			if (*p == '\n') {
				buf[i] = '\0';
				line = buf;
				line = trim(line);
				buf[i=0]='\0';
				break;
			} else {
				buf[i] = *p;
				i ++;
			}
			if (this->cur_len == this->len) buf[i] = '\0';
		}
		if (flag == 0) return false;
		return true;
	}
	//
	void mmap_init(string file) {
	    this->fd = open(file.c_str(), O_RDONLY);
		this->len = lseek(this->fd, 0, SEEK_END);
	    this->mbuf = (char *) mmap(NULL, this->len, PROT_READ, MAP_PRIVATE, this->fd, 0);
	}
	//
	string trim(const string &str) {
		int i;
    	string::size_type pos = 0;
    	for (i = 0; i < str.size(); ++i) {
    		if (isspace(str[i]) == 0) {
    			pos = i;
    			break;
    		}
    	}
    	if (pos == string::npos) {
    		return str;
    	}
    	string::size_type pos2 = 0;
    	for (i = str.size() - 1; i >= 0; --i) {
    		if (isspace(str[i]) == 0) {
    			pos2 = i;
    			break;
    		}
    	}
    	if (pos2 != string::npos) {
    		return str.substr(pos, pos2 - pos + 1);
    	}
    	return str.substr(pos);
	}
};

#endif