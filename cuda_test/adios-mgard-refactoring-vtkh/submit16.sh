#!/bin/bash 
### Begin BSUB Options 
#BSUB -P csc143 
#BSUB -J MG16 
#BSUB -W 00:20 
#BSUB -nnodes 16 
#BSUB -alloc_flags "smt1"


NPROC=64
R=4
N=513

JSRUN='jsrun -n '$NPROC' -a 1 -c 1 -g 1 -r '$R' -l CPU-CPU -d packed -b packed:1'
echo $JSRUN

DATA_PREFIX='/gpfs/alpine/scratch/jieyang/csc143/mgard-para-test'

rm -rf $DATA_PREFIX/$NPROC
mkdir -p $DATA_PREFIX/$NPROC
mkdir -p $DATA_PREFIX/$NPROC/cpu
mkdir -p $DATA_PREFIX/$NPROC/cuda

#$JSRUN js_task_info | sort
$JSRUN ./build/test $N 10 $DATA_PREFIX/$NPROC/cpu/ 0 
$JSRUN ./build/test $N 10 $DATA_PREFIX/$NPROC/cuda/ 1
