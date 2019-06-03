#!/bin/bash

#rm -rf c-blosc-* v1.16.3
wget https://github.com/Blosc/c-blosc/archive/v1.16.3.tar.gz
tar -zxvf v1.16.3.tar.gz
cd c-blosc-*
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../../blosc ..
#ccmake ..
cmake --build . --target install

cd ../..

rm -rf c-blosc-* v1.16.3*
