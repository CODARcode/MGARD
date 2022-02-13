#!/bin/sh

# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
# Script for building the example

set -x
set -e

mgard_src_dir=$(pwd)/../../../
nvcomp_build_dir=${mgard_src_dir}/external-cuda-turing/nvcomp/build
zstd_install_dir=${mgard_src_dir}/external-cuda-turing/zstd/install
protobuf_install_dir=${mgard_src_dir}/external-cuda-turing/protobuf/install

export LD_LIBRARY_PATH=${nvcomp_build_dir}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${zstd_install_dir}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${zstd_install_dir}/lib:${LD_LIBRARY_PATH}

# echo ${LD_LIBRARY_PATH}

rm -rf build
mkdir build 
cmake -S .  -B ./build \
	    -DCMAKE_MODULE_PATH=${mgard_src_dir}/cmake\
	    -Dmgard_ROOT=${mgard_src_dir}/install-cuda-turing\
	    -DCMAKE_PREFIX_PATH="${nvcomp_build_dir};${zstd_install_dir}/lib/cmake/zstd"

cd build && make VERBOSE=1 && cd ..

# cmake --build ./build
