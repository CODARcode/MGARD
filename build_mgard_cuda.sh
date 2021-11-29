#!/bin/sh

# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
# Script for building MGARD-GPU

set -e
set -x

home_dir=$(pwd)
external_dir=${home_dir}/external
mkdir -p ${external_dir}

#build NVCOMP
nvcomp_dir=${external_dir}/nvcomp
nvcomp_src_dir=${nvcomp_dir}/src
nvcomp_build_dir=${nvcomp_dir}/build
nvcomp_install_dir=${nvcomp_dir}/install
nvcomp_library=${nvcomp_dir}/build/lib/libnvcomp.so
# if [ ! -f "${nvcomp_dir}" ]; then
	# rm -rf ${nvcomp_dir} && mkdir -p ${nvcomp_dir}
	# git clone https://github.com/NVIDIA/nvcomp.git ${nvcomp_src_dir}
	# cmake -S ${nvcomp_src_dir} -B ${nvcomp_build_dir}\
	# 	  -DCMAKE_INSTALL_PREFIX=${nvcomp_install_dir}
	# cmake --build ${nvcomp_build_dir} -j8
	# cmake --install ${nvcomp_build_dir}
# fi

#build Kokkos
kokkos_dir=${external_dir}/kokkos
kokkos_src_dir=${kokkos_dir}/src
kokkos_build_dir=${kokkos_dir}/build
kokkos_install_dir=${kokkos_dir}/install
# kokkos_library=${kokkos_dir}/build/lib/libnvcomp.so
# if [ ! -f "${kokkos_dir}" ]; then
	# rm -rf ${kokkos_dir} && mkdir ${kokkos_dir} 
	# git clone https://github.com/kokkos/kokkos.git ${kokkos_src_dir}
	# cmake -S ${kokkos_src_dir} -B ${kokkos_build_dir}\
	# 	-DCMAKE_CXX_COMPILER=${kokkos_src_dir}/bin/nvcc_wrapper\
	# 	-DKokkos_ENABLE_CUDA=ON\
	# 	-DCMAKE_INSTALL_PREFIX=${kokkos_install_dir}
	# cmake --build ${kokkos_build_dir} -j8
	# cmake --install ${kokkos_build_dir}
# fi


#build MGARD-CUDA
mgard_x_src_dir=${home_dir}
mgard_x_build_dir=${home_dir}/build
mgard_x_install_dir=${home_dir}/install
rm -rf ${mgard_x_build_dir} && mkdir -p ${mgard_x_build_dir}
cmake -S ${mgard_x_src_dir} -B ${mgard_x_build_dir} \
	  -DCMAKE_PREFIX_PATH="${nvcomp_build_dir};${kokkos_install_dir}"\
	  -DMGARD_ENABLE_SERIAL=ON\
	  -DMGARD_ENABLE_CUDA=ON\
	  -DMGARD_ENABLE_CUDA_FMA=ON\
	  -DMGARD_ENABLE_CUDA_OPTIMIZE_VOLTA=OFF\
	  -DMGARD_ENABLE_CUDA_OPTIMIZE_TURING=ON\
	  -DCMAKE_BUILD_TYPE=Release\
	  -DCMAKE_INSTALL_PREFIX=${mgard_x_install_dir}
cmake --build ${mgard_x_build_dir} -j8
cmake --install ${mgard_x_build_dir}

	  # -DCMAKE_PREFIX_PATH=${kokkos_install_dir}\
