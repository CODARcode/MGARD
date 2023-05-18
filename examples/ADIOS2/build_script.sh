#!/bin/sh

# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
# Script for building the example

set -x
set -e

mgard_install_dir=/home/jieyang/dev/MGARD/install-cuda-ampere
adios2_install_dir=/home/jieyang/dev/ADIOS2/install


rm -rf build
mkdir build 
cmake -S .  -B ./build \
	-DCMAKE_CUDA_ARCHITECTURES=86\
	-DCMAKE_PREFIX_PATH="${mgard_install_dir};${adios2_install_dir}"
	  
cmake --build ./build
