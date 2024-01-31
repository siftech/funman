.DEFAULT_GOAL := all

build:
	mkdir -p build

build/num_constraint_test.o: src/num_constraint_test.cc
	g++ -c -g src/num_constraint_test.cc -I/usr/include/ibex -libex -o build/num_constraint_test.o

build/num_constraint_test: build/num_constraint_test.o
	g++ -g build/num_constraint_test.o -libex -o build/num_constraint_test

test: build/num_constraint_test
	build/num_constraint_test

all: build test
	