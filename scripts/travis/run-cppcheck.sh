#!/usr/bin/env bash

CPPCHECK_BUILD_DIR=cppcheck-build-dir

cd "${TRAVIS_BUILD_DIR}"
mkdir "${CPPCHECK_BUILD_DIR}"
cppcheck --project=build/compile_commands.json --cppcheck-build-dir="${CPPCHECK_BUILD_DIR}" --enable=all --quiet

exit 0
