#!/bin/bash

module purge
ml DefApps
ml gcc
ml hdf5

module unload darshan-runtime
module load gcc/9
module load cuda/11.0.3
module load cmake/3.20.2
module load adios2

MGARD_DIR=../../install-cuda-summit

rm -rf build
mkdir build 

CC=mpicc CXX=mpicxx FC=mpifort \
cmake -S .  -B ./build \
    -DCMAKE_PREFIX_PATH="$MGARD_DIR;$OLCF_ADIOS2_ROOT"
