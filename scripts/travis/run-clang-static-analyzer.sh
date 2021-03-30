#!/usr/bin/env bash

CLANG_STATIC_ANALYZER_BUILD_DIR=build-clang-static-analyzer

cd "${TRAVIS_BUILD_DIR}"
mkdir "${CLANG_STATIC_ANALYZER_BUILD_DIR}"
cd "${CLANG_STATIC_ANALYZER_BUILD_DIR}"
cmake ..
scan-build --status-bugs make

exit $?
