#!/bin/sh

set -x
sex -e

mgard_src_dir=$(pwd)/../../
nvcomp_build_dir=${mgard_src_dir}/external/nvcomp/build

rm -rf build
mkdir build 
cmake -S .  -B ./build \
	  -DCMAKE_MODULE_PATH=${mgard_src_dir}/cmake\
	  -DNVCOMP_ROOT=${nvcomp_build_dir}
cmake --build ./build