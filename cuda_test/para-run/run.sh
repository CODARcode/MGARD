#!/bin/bash
PLATFORM=rtx2080ti 
#PLATFORM=v100


mkdir $PLATFORM


N=8
NPROC=8
DEV=0
mpirun -np $NPROC ./build/test 66 66 66 33 33 33 $N $DEV
cp -r results-* $PLATFORM
./aggregate_results.py $N $NPROC 3 $DEV $PLATFORM 66 66 66 16 32 


NPROC=1
DEV=1
mpirun -np $NPROC ./build/test 66 66 66 33 33 33 $N $DEV
cp -r results-* $PLATFORM
./aggregate_results.py $N $NPROC 3 $DEV $PLATFORM 66 66 66 16 32 


