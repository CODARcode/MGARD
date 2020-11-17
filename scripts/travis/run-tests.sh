#!/usr/bin/env bash

if [ -z "${SOURCE_DIR}" ]
then
  echo "Error: SOURCE_DIR is empty or undefined"
  exit 1
fi
if [ -z "${COMMIT_RANGE}" ]
then
  echo "Error: COMMIT_RANGE is empty or undefined"
  exit 1
fi

cd ${SOURCE_DIR}

build/bin/tests

if [ $? -eq 0 ]
then
  echo "Successfully ran Catch2 tests"
else
  echo "Error in running Catch2 tests" >&2
  exit 1
fi

exit 0
