#!/usr/bin/env bash

cd "${HOME}"
gunzip "2.3.tar.gz"
tar --file "2.3.tar" --extract
cd "cppcheck-2.3"
mkdir "build"
cd "build"
cmake -DCMAKE_INSTALL_PREFIX="${HOME}/.local" ..
make --jobs 8
make install
