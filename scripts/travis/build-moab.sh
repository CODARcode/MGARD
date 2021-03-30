#!/usr/bin/env bash

cd "${HOME}"
gunzip "moab-5.2.1.tar.gz"
tar --file "moab-5.2.1.tar" --extract
cd "moab-5.2.1"
autoreconf --force --install
./configure --prefix="${HOME}/.local"
make --jobs 8
make install
