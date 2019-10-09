CC = g++

CFLAG = -std=gnu++0x -O3 -Wall -lpthread

#objects = lbfgs.o MaxEnt.o

all : maxent_learn maxent_predict postagger

postagger : postagger.cpp MaxEnt.cpp lbfgs.cpp
	$(CC) $(CFLAG) lbfgs.cpp MaxEnt.cpp postagger.cpp -o postagger

maxent_learn : maxent_learn.cpp MaxEnt.cpp lbfgs.cpp
	$(CC) $(CFLAG) lbfgs.cpp MaxEnt.cpp maxent_learn.cpp -o maxent_learn

maxent_predict : maxent_predict.cpp MaxEnt.cpp lbfgs.cpp
	$(CC) $(CFLAG) lbfgs.cpp MaxEnt.cpp maxent_predict.cpp -o maxent_predict

#maxent : main.cpp $(objects)
#	$(CC) $(CFLAG) $(objects) main.cpp -o maxent

#MaxEnt.o : MaxEnt.cpp
#	$(CC) -c MaxEnt.cpp

#lbfgs.o : lbfgs.cpp
#	$(CC) -c lbfgs.cpp

clean :
	rm -rf  maxent_learn maxent_predict postagger *.o
