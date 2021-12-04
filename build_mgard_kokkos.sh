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

#build kokkos
kokkos_dir=${external_dir}/kokkos
kokkos_src_dir=${kokkos_dir}/src
kokkos_build_dir=${kokkos_dir}/build
kokkos_install_dir=${kokkos_dir}/install
if [ ! -d "${kokkos_install_dir}" ]; then
	# rm -rf ${kokkos_dir} && mkdir -p ${kokkos_dir}
	# git clone -b master https://github.com/kokkos/kokkos.git ${kokkos_src_dir}
	# rm -rf ${kokkos_build_dir}
	cmake -S ${kokkos_src_dir} -B ${kokkos_build_dir}\
		  -DKokkos_ENABLE_CUDA=ON\
		  -DBUILD_SHARED_LIBS=ON\
		  -DCMAKE_INSTALL_PREFIX=${kokkos_install_dir}
	cmake --build ${kokkos_build_dir} -j8
	cmake --install ${kokkos_build_dir}
fi

#build MGARD
mgard_x_src_dir=${home_dir}
mgard_x_build_dir=${home_dir}/build
mgard_x_install_dir=${home_dir}/install
rm -rf ${mgard_x_build_dir} && mkdir -p ${mgard_x_build_dir}
cmake -S ${mgard_x_src_dir} -B ${mgard_x_build_dir} \
	  -DCMAKE_PREFIX_PATH=${kokkos_install_dir}\
	  -DMGARD_ENABLE_SERIAL=OFF\
	  -DMGARD_ENABLE_KOKKOS=ON\
	  -DCMAKE_BUILD_TYPE=Release\
	  -DCMAKE_INSTALL_PREFIX=${mgard_x_install_dir}
cmake --build ${mgard_x_build_dir} -j8
cmake --install ${mgard_x_build_dir}
