#!/bin/sh
set -e

ROOT=$(pwd)
#build NVCOMP
rm -rf external
mkdir external 
cd external
git clone https://github.com/NVIDIA/nvcomp.git
cd nvcomp 
mkdir build 
cd build
cmake .. && make -j
cd ../../../

#build MGARD-CUDA
rm -rf build
mkdir build
cd build
cmake .. -DNVCOMP_DIR=$ROOT/external/nvcomp/build -DMGARD_ENABLE_CUDA=ON
make -j8

