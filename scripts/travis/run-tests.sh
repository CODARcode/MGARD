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

make check

if [ $? -eq 0 ]
then
  echo "Successfully built and ran unstructured tests"
else
  echo "Error in building or running unstructured tests" >&2
  exit 1
fi

exit 0
