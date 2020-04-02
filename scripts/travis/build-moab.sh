#!/usr/bin/env bash

cd "${HOME}/moab"
git checkout 5.1.0
autoreconf --force --install
./configure --prefix="${HOME}/.local"
make --jobs 8
make install
