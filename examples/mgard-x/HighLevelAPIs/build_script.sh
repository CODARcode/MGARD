#!/bin/sh

# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
# Script for building the example

set -x
set -e

install_dir=$(pwd)/../../../install-hip-frontier

rm -rf build
mkdir build 
cmake -S .  -B ./build \
	    -Dmgard_ROOT=${install_dir}\
	    -DCMAKE_CUDA_ARCHITECTURES=75\
	    -DCMAKE_PREFIX_PATH="${install_dir}"
	  
cmake --build ./build
