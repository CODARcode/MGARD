#!/bin/sh

# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
# Script for building MGARD-X

set -e
set -x

######## User Configurations ########
# Source directory
mgard_x_src_dir=.
# Build directory
build_dir=./build-sycl-x86
# Number of processors used for building
num_build_procs=8
# Installtaion directory
install_dir=./install-sycl-x86


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
    -DCMAKE_CXX_COMPILER=icpx\
    -DCMAKE_C_COMPILER=icx\
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
    -DCMAKE_CXX_COMPILER=icpx\
    -DCMAKE_C_COMPILER=icx\
    -DCMAKE_INSTALL_PREFIX=${protobuf_install_dir}
cmake --build ${protobuf_build_dir} -j ${num_build_procs}
cmake --install ${protobuf_build_dir}


#build MGARD
mgard_x_build_dir=${build_dir}/mgard
mgard_x_install_dir=${install_dir}
mkdir -p ${mgard_x_build_dir}
cmake -S ${mgard_x_src_dir} -B ${mgard_x_build_dir} \
    -DCMAKE_PREFIX_PATH="${zstd_install_dir}/lib/cmake/zstd;${protobuf_install_dir}"\
    -DMGARD_ENABLE_SERIAL=OFF\
    -DMGARD_ENABLE_SYCL=ON\
    -DCMAKE_CXX_COMPILER=icpx\
    -DCMAKE_C_COMPILER=icx\
    -DCMAKE_CXX_FLAGS="-fsycl -fsycl-targets=x86_64"\
    -DMGARD_ENABLE_DOCS=OFF\
    -DCMAKE_BUILD_TYPE=Release\
    -DCMAKE_INSTALL_PREFIX=${mgard_x_install_dir}
cmake --build ${mgard_x_build_dir} -j ${num_build_procs}
cmake --install ${mgard_x_build_dir}
