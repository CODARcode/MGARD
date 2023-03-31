#!/bin/sh

set -x
set -e

install_dir=$(pwd)/../../install-serial/

rm -rf build
mkdir build

cmake -S .  -B ./build \
      -DCMAKE_PREFIX_PATH="${install_dir}"

cmake --build ./build

