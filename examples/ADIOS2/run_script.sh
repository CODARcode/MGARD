#!/bin/bash

set -e
set -x

IN_DATA=/home/jieyang/dev/data/xgc.f0.00200.bp
OUT_DATA=/home/jieyang/dev/data/xgc.f0.00200-out.bp

IN_DATA=/home/jieyang/dev/data/wrfout_d01_2019-11-26_23:50:00
OUT_DATA=/home/jieyang/dev/data/wrfout_d01_2019-11-26_23:50:00.mgard


./build_script.sh
rm -rf $OUT_DATA
./build/adios2-test -z -i $IN_DATA -c $OUT_DATA -t s -n 3 512 512 512 -m rel -e 1e-4 -s inf -v T2 -b 0 -d 1 
./build/adios2-test -x -i $IN_DATA -c $OUT_DATA -t s -n 3 512 512 512 -m rel -e 1e-4 -s inf -v T2 -b 0 -d 1 