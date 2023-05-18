#!/bin/bash

set -e
set -x

IN_DATA=/home/jieyang/dev/data/xgc.f0.00200.bp
OUT_DATA=/home/jieyang/dev/data/xgc.f0.00200-out.bp

IN_DATA=/home/jieyang/dev/data/wrfout_d01_2019-11-26_23:50:00
OUT_DATA=/home/jieyang/dev/data/wrfout_d01_2019-11-26_23:50:00.mgard


./build_script.sh
rm -rf $OUT_DATA
./build/adios2-test -z -i $IN_DATA -c $OUT_DATA -t s -n 2 1200 1500 -m rel -e 7e-6 -s inf -v T2 -b 0 -d 2 
./build/adios2-test -x -i $IN_DATA -c $OUT_DATA -t s -n 2 1200 1500 -m rel -e 7e-6 -s inf -v T2 -b 0 -d 2 