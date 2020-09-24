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

cp tests/gray-scott/adios2.xml .
mpirun -n 3 build/bin/gray-scott tests/gray-scott/simulation/settings-files.json

if [ $? -eq 0 ]
then
  echo "Successfully ran Gray–Scott test"
else
  echo "Error in Gray–Scott test" >&2
  exit 1
fi

make check

if [ $? -eq 0 ]
then
  echo "Successfully built and ran unstructured tests"
else
  echo "Error in building or running unstructured tests" >&2
  exit 1
fi

exit 0
