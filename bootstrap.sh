#!/bin/bash

rm -rf c-blosc-* v1.16.3
wget https://github.com/Blosc/c-blosc/archive/v1.16.3.tar.gz
tar -zxvf v1.16.3
cd c-blosc-*
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../../blosc ..
#ccmake ..
cmake --build . --target install

rm -rf v1.16.3 c-blosc-* v1.16.3
