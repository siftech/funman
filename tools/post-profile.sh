#!/bin/bash

# This script converts the gmon.out created by gprof into a png call graph

rm prof.out prof.dot prof.dot.png || echo "No stale files to remove ... skipping"

echo "Processing gmon.out ... "
# -c: generate all children including those not compile with -pg
# -L: print full path for functions
gprof -c -L /usr/bin/dreal > prof.out

echo "Generating call graph plot ... "
gprof2dot prof.out -o prof.dot

echo " Creating call graph figure ... "
dot -Tpng -O prof.dot
