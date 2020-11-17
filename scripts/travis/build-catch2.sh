#!/usr/bin/env bash

cd "${HOME}/Catch2"
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX="${HOME}/.local" ..
make --jobs 8
make install
