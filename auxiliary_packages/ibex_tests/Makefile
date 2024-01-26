.DEFAULT_GOAL := all

num_constraint_test.o: src/num_constraint_test.cc
	g++ -c -g src/num_constraint_test.cc -I/usr/include/ibex -libex

num_constraint_test: num_constraint_test.o
	g++ -g num_constraint_test.o -libex -o num_constraint_test

test: num_constraint_test

all: test