#!/bin/sh

# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
# Script for building MGARD-GPU

set -e
set -x

home_dir=$(pwd)
#build NVCOMP
nvcomp_src_dir=${home_dir}/external/nvcomp/src
nvcomp_build_dir=${home_dir}/external/nvcomp/build
nvcomp_library=${home_dir}/external/nvcomp/build/lib/libnvcomp.so
if [ ! -f "$nvcomp_library" ]; then
	rm -rf external && mkdir external 
	git clone https://github.com/NVIDIA/nvcomp.git ${nvcomp_src_dir}
	cmake -S ${nvcomp_src_dir} -B ${nvcomp_build_dir}
	cmake --build ${nvcomp_build_dir} -j8
fi

#build MGARD-CUDA
mgard_x_src_dir=${home_dir}
mgard_x_build_dir=${home_dir}/build
mgard_x_install_dir=${home_dir}/install
rm -rf ${mgard_x_build_dir} && mkdir -p ${mgard_x_build_dir}
cmake -S ${mgard_x_src_dir} -B ${mgard_x_build_dir} \
	  -DCMAKE_PREFIX_PATH=${nvcomp_build_dir}\
	  -DMGARD_ENABLE_CUDA=ON\
	  -DMGARD_ENABLE_CUDA_FMA=ON\
	  -DMGARD_ENABLE_CUDA_OPTIMIZE_VOLTA=OFF\
	  -DMGARD_ENABLE_CUDA_OPTIMIZE_TURING=ON\
	  -DCMAKE_BUILD_TYPE=Release\
	  -DCMAKE_INSTALL_PREFIX=${mgard_x_install_dir}
cmake --build ${mgard_x_build_dir} -j8
cmake --install ${mgard_x_build_dir}

