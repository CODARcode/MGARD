#!/bin/bash 
### Begin BSUB Options 
#BSUB -P csc143 
#BSUB -J MGRAD-CUDA 
#BSUB -W 2:00 
#BSUB -nnodes 1 
#BSUB -alloc_flags "smt1"

ml cuda gcc/7.4.0 cmake python
JSRUN='jsrun -n 1 -a 1 -c 42 -g 6 -r 1 -l CPU-CPU -d packed -b packed:42 --smpiargs="-disable_gpu_hooks"'

rm -rf /gpfs/alpine/scratch/jieyang/csc143/v100
mkdir -p /gpfs/alpine/scratch/jieyang/csc143/v100
$JSRUN python test.py
