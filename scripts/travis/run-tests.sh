#!/usr/bin/env bash

cd "${TRAVIS_BUILD_DIR}"
build/bin/tests

if [ $? -eq 0 ]
then
  echo "Successfully ran Catch2 tests"
else
  echo "Error in running Catch2 tests" >&2
  exit 1
fi

exit 0
