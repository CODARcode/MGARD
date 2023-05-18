#!/bin/bash

set -e
set -x

IN_DATA=/home/jieyang/dev/data/xgc.f0.00200.bp
OUT_DATA=/home/jieyang/dev/data/xgc.f0.00200-out.bp

IN_DATA=/home/jieyang/dev/data/wrfout_d01_2019-11-26_23:50:00.bp
OUT_DATA=/home/jieyang/dev/data/wrfout_d01_2019-11-26_23:50:00.mgard.bp
DEC_DATA=/home/jieyang/dev/data/wrfout_d01_2019-11-26_23:50:00.mgard_dec_linf_1e-6.bp

eb6=1.5e-5
eb4=2.5e-3
eb2=2.5e-1

./build_script.sh
rm -rf $OUT_DATA
./build/adios2-test -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t s -n 2 1200 1500 -m rel -e $eb6 -s inf -v T2 -b 0 -d 0 
./build/adios2-test -x -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t s -n 2 1200 1500 -m rel -e $eb6 -s inf -v T2 -b 0 -d 0 


# eb6=2e-6
# eb4=2.3e-4
# eb2=2.6e-2

# ./build_script.sh
# rm -rf $OUT_DATA
# ./build/adios2-test -z -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t s -n 2 1200 1500 -m rel -e $eb2 -s 0 -v T2 -b 0 -d 2 
# ./build/adios2-test -x -i $IN_DATA -c $OUT_DATA -o $DEC_DATA -t s -n 2 1200 1500 -m rel -e $eb2 -s 0 -v T2 -b 0 -d 2 