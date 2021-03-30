#!/usr/bin/env bash

cd "${HOME}"
gunzip "v3.0.0-preview3.tar.gz"
tar --file "v3.0.0-preview3.tar" --extract
cd "Catch2-3.0.0-preview3"
mkdir "build"
cd "build"
cmake -DCMAKE_INSTALL_PREFIX="${HOME}/.local" ..
make --jobs 8
make install
