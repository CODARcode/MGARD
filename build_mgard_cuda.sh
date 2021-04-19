#!/bin/sh
set -e

home_dir=$(pwd)
#build NVCOMP
nvcomp_src_dir=${home_dir}/external/nvcomp/src
nvcomp_build_dir=${home_dir}/external/nvcomp/build
# rm -rf external && mkdir external 
# git clone https://github.com/NVIDIA/nvcomp.git ${nvcomp_src_dir}
# cmake -S ${nvcomp_src_dir} -B ${nvcomp_build_dir}
# cmake --build ${nvcomp_build_dir} -j8

#build MGARD-CUDA
mgard_cuda_src_dir=${home_dir}
mgard_cuda_build_dir=${home_dir}/build
mgard_cuda_install_dir=${home_dir}/install
rm -rf ${mgard_cuda_build_dir} && mkdir -p ${mgard_cuda_build_dir}
cmake -S ${mgard_cuda_src_dir} -B ${mgard_cuda_build_dir} \
	  -DNVCOMP_ROOT=${nvcomp_build_dir}\
	  -DMGARD_ENABLE_CUDA=ON\
	  -DCMAKE_INSTALL_PREFIX=${mgard_cuda_install_dir}
cmake --build ${mgard_cuda_build_dir} -j8
cmake --install ${mgard_cuda_build_dir}
