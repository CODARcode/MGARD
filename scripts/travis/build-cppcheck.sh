#!/usr/bin/env bash

cd ${HOME}
mkdir -p /opt/cppcheck/build /opt/cppcheck/install
cd /opt/cppcheck
tar -xzf ${HOME}/1.87.tar.gz
cd build

cmake \
  -DCMAKE_INSTALL_PREFIX:PATH=/opt/cppcheck/install \
  -DCMAKE_C_COMPILER:FILEPATH=${CC} \
  -DCMAKE_CXX_COMPILER:FILEPATH=${CXX} \
  ../cppcheck-1.87

make -j8
make install
