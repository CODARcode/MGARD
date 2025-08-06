#!/bin/sh

# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
# Script for building MGARD-GPU

set -e
set -x

module load rocm/4.5.0
module load cmake

home_dir=$(pwd)
external_dir=${home_dir}/external-hip-spock
mkdir -p ${external_dir}

#build ZSTD
zstd_dir=${external_dir}/zstd
zstd_src_dir=${zstd_dir}/src
zstd_build_dir=${zstd_dir}/build
zstd_install_dir=${zstd_dir}/install
if [ ! -d "${zstd_install_dir}" ]; then
  rm -rf ${zstd_dir} && mkdir -p ${zstd_dir}
  git clone -b v1.5.0 https://github.com/facebook/zstd.git ${zstd_src_dir}
  cmake -S ${zstd_src_dir}/build/cmake -B ${zstd_build_dir}\
      -DZSTD_MULTITHREAD_SUPPORT=ON\
      -DCMAKE_C_COMPILER=amdclang\
      -DCMAKE_CXX_COMPILER=amdclang++\
			-DCMAKE_INSTALL_LIBDIR=lib\
      -DCMAKE_INSTALL_PREFIX=${zstd_install_dir}
  cmake --build ${zstd_build_dir} -j8
  cmake --install ${zstd_build_dir}
fi

#build Protobuf
protobuf_dir=${external_dir}/protobuf
protobuf_src_dir=${protobuf_dir}/src
protobuf_build_dir=${protobuf_dir}/build
protobuf_install_dir=${protobuf_dir}/install
if [ ! -d "${protobuf_install_dir}" ]; then
  rm -rf ${protobuf_dir} && mkdir -p ${protobuf_dir}
  git clone -b v3.19.4 https://github.com/protocolbuffers/protobuf.git ${protobuf_src_dir}
  cd ${protobuf_src_dir} && git submodule update --init --recursive && cd ${home_dir}
  cmake -S ${protobuf_src_dir}/cmake -B ${protobuf_build_dir}\
      -DCMAKE_C_COMPILER=amdclang\
      -DCMAKE_CXX_COMPILER=amdclang++\
      -Dprotobuf_BUILD_SHARED_LIBS=ON\
      -DCMAKE_INSTALL_PREFIX=${protobuf_install_dir}
  cmake --build ${protobuf_build_dir} -j8
  cmake --install ${protobuf_build_dir}
fi

#build MGARD
mgard_x_src_dir=${home_dir}
mgard_x_build_dir=${home_dir}/build-hip-spock
mgard_x_install_dir=${home_dir}/install-hip-spock
rm -rf ${mgard_x_build_dir} && mkdir -p ${mgard_x_build_dir}
cmake -S ${mgard_x_src_dir} -B ${mgard_x_build_dir} \
    -DCMAKE_PREFIX_PATH="${zstd_install_dir}/lib/cmake/zstd;${protobuf_install_dir}"\
    -DMGARD_ENABLE_SERIAL=ON\
    -DMGARD_ENABLE_HIP=ON\
    -DCMAKE_CXX_COMPILER=hipcc\
    -DCMAKE_CXX_FLAGS="--amdgpu-target=gfx908"\
    -DCMAKE_BUILD_TYPE=Release\
    -DCMAKE_INSTALL_PREFIX=${mgard_x_install_dir}
cmake --build ${mgard_x_build_dir} -j8
cmake --install ${mgard_x_build_dir}
