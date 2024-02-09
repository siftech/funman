.DEFAULT_GOAL := all

build:
	mkdir -p build

build/num_constraint_test.o: src/num_constraint_test.cc
	g++ -c -g -pg src/num_constraint_test.cc -I/usr/include/ibex -L/root/ibex-lib/__build__/src/libibex.so  -o build/num_constraint_test.o

build/num_constraint_test: build/num_constraint_test.o
	g++ -g -pg build/num_constraint_test.o -L/root/ibex-lib/__build__/src/libibex.so -libex -o build/num_constraint_test

test: build/num_constraint_test
	build/num_constraint_test

profile: test
	gprof --flat-profile build/num_constraint_test

all: build test
	