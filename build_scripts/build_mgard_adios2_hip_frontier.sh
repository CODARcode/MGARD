#!/bin/sh

# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
# Script for building MGARD-X

set -e
set -x

module load rocm/5.5.1
module load cmake

######## User Configurations ########
# Source directory
mgard_x_src_dir=.
# Build directory
build_dir=./build-hip-crusher
# Number of processors used for building
num_build_procs=$1
# Installtaion directory
install_dir=./install-hip-crusher

export LD_LIBRARY_PATH=$(pwd)/${install_dir}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/${install_dir}/lib64:$LD_LIBRARY_PATH
export CC=amdclang
export CXX=amdclang++

#build ZSTD
zstd_dir=${build_dir}/zstd
zstd_src_dir=${zstd_dir}/src
zstd_build_dir=${zstd_dir}/build
zstd_install_dir=${install_dir}
if [ ! -d "${zstd_src_dir}" ]; then
  git clone -b v1.5.0 https://github.com/facebook/zstd.git ${zstd_src_dir}
fi
mkdir -p ${zstd_build_dir}
cmake -S ${zstd_src_dir}/build/cmake -B ${zstd_build_dir}\
    -DZSTD_MULTITHREAD_SUPPORT=ON\
    -DCMAKE_INSTALL_LIBDIR=lib\
    -DCMAKE_INSTALL_PREFIX=${zstd_install_dir}
cmake --build ${zstd_build_dir} -j ${num_build_procs}
cmake --install ${zstd_build_dir}


#build Protobuf
protobuf_dir=${build_dir}/protobuf
protobuf_src_dir=${protobuf_dir}/src
protobuf_build_dir=${protobuf_dir}/build
protobuf_install_dir=${install_dir}
if [ ! -d "${protobuf_src_dir}" ]; then
  git clone -b v3.19.4 --recurse-submodules https://github.com/protocolbuffers/protobuf.git ${protobuf_src_dir}
fi
mkdir -p ${protobuf_build_dir}
cmake -S ${protobuf_src_dir}/cmake -B ${protobuf_build_dir}\
    -Dprotobuf_BUILD_SHARED_LIBS=ON\
    -Dprotobuf_BUILD_TESTS=OFF\
    -DCMAKE_INSTALL_PREFIX=${protobuf_install_dir}
cmake --build ${protobuf_build_dir} -j ${num_build_procs}
cmake --install ${protobuf_build_dir}


#build MGARD
mgard_x_build_dir=${build_dir}/mgard
mgard_x_install_dir=${install_dir}
mkdir -p ${mgard_x_build_dir}
cmake -S ${mgard_x_src_dir} -B ${mgard_x_build_dir} \
    -DCMAKE_PREFIX_PATH="${zstd_install_dir}/lib/cmake/zstd;${protobuf_install_dir}"\
    -DMGARD_ENABLE_HIP=ON\
    -DCMAKE_HIP_ARCHITECTURES="gfx90a"\
    -DCMAKE_BUILD_TYPE=Release\
    -DCMAKE_INSTALL_PREFIX=${mgard_x_install_dir}
cmake --build ${mgard_x_build_dir} -j ${num_build_procs}
cmake --install ${mgard_x_build_dir}

#build ADIOS2
adios2_dir=${build_dir}/adios2
adios2_src_dir=${adios2_dir}/src
adios2_build_dir=${adios2_dir}/build
adios2_install_dir=${install_dir}
if [ ! -d "${adios2_src_dir}" ]; then
  git clone https://github.com/ornladios/ADIOS2.git ${adios2_src_dir}
fi
mkdir -p ${adios2_build_dir}
cmake -S ${adios2_src_dir} -B ${adios2_build_dir}\
      -DADIOS2_USE_CUDA=OFF \
      -DADIOS2_USE_MGARD=ON \
      -DCMAKE_PREFIX_PATH=${mgard_x_install_dir} \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON \
      -DADIOS2_USE_MPI=ON \
      -DADIOS2_USE_Fortran=OFF \
      -DADIOS2_USE_SST=OFF \
      -DCMAKE_INSTALL_PREFIX=${adios2_install_dir}
cmake --build ${adios2_build_dir} -j ${num_build_procs}
cmake --install ${adios2_build_dir}
