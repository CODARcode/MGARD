#!/bin/sh

# Copyright 2021, Oak Ridge National Laboratory.
# MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
# Author: Jieyang Chen (chenj3@ornl.gov)
# Date: April 2, 2021
# Script for building MGARD-GPU

set -e
set -x

home_dir=$(pwd)
external_dir=${home_dir}/external-cuda-turing
mkdir -p ${external_dir}

#build NVCOMP
nvcomp_dir=${external_dir}/nvcomp
nvcomp_src_dir=${nvcomp_dir}/src
nvcomp_build_dir=${nvcomp_dir}/build
nvcomp_install_dir=${nvcomp_dir}/install
nvcomp_library=${nvcomp_dir}/build/lib/libnvcomp.so
if [ ! -d "${nvcomp_install_dir}" ]; then
  rm -rf ${nvcomp_dir} && mkdir -p ${nvcomp_dir}
  git clone https://github.com/NVIDIA/nvcomp.git ${nvcomp_src_dir}
  cmake -S ${nvcomp_src_dir} -B ${nvcomp_build_dir}\
      -DCMAKE_INSTALL_PREFIX=${nvcomp_install_dir}
  cmake --build ${nvcomp_build_dir} -j8
  cmake --install ${nvcomp_build_dir}
fi

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
      -DCMAKE_INSTALL_LIBDIR=lib\
      -DCMAKE_INSTALL_PREFIX=${zstd_install_dir}
  cmake --build ${zstd_build_dir} -j8
  cmake --install ${zstd_build_dir}
fi

export LD_LIBRARY_PATH=${zstd_install_dir}/lib:$LD_LIBRARY_PATH

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
      -Dprotobuf_BUILD_SHARED_LIBS=ON\
      -DCMAKE_INSTALL_PREFIX=${protobuf_install_dir}
  cmake --build ${protobuf_build_dir} -j8
  cmake --install ${protobuf_build_dir}
fi



#build MGARD
mgard_x_src_dir=${home_dir}
mgard_x_build_dir=${home_dir}/build-cuda-turing
mgard_x_install_dir=${home_dir}/install-cuda-turing
rm -rf ${mgard_x_build_dir} && mkdir -p ${mgard_x_build_dir}
cmake -S ${mgard_x_src_dir} -B ${mgard_x_build_dir} \
    -DCMAKE_PREFIX_PATH="${nvcomp_build_dir};${zstd_install_dir}/lib/cmake/zstd;${protobuf_install_dir}"\
    -DMGARD_ENABLE_SERIAL=ON\
    -DMGARD_ENABLE_CUDA=ON\
    -DCMAKE_CUDA_ARCHITECTURES="75"\
    -DMGARD_ENABLE_DOCS=ON\
    -DCMAKE_BUILD_TYPE=Release\
    -DCMAKE_INSTALL_PREFIX=${mgard_x_install_dir}
cmake --build ${mgard_x_build_dir} -j6
cmake --install ${mgard_x_build_dir}
