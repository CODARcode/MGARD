#!/bin/sh

# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
# Script for building the example

set -x
set -e

mgard_src_dir=$(pwd)/../../
nvcomp_build_dir=${mgard_src_dir}/external/nvcomp/build

rm -rf build
mkdir build 
cmake -S .  -B ./build \
	  -DCMAKE_MODULE_PATH=${mgard_src_dir}/cmake\
	  -Dmgard_ROOT=${mgard_src_dir}/install\
	  -DNVCOMP_ROOT=${nvcomp_build_dir}
	  
cmake --build ./build
