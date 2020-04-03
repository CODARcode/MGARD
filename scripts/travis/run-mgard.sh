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

./build/bin/cosine

if [ $? -eq 0 ]
then
  echo "Successfully compressed/decompressed cosine"
else
  echo "Error in compressing/decompressing cosine" >&2
  exit 1
fi

./build/bin/constant3d

if [ $? -eq 0 ]
then
  echo "Successfully compressed/decompressed constant3d"
else
  echo "Error in compressing/decompressing constant3d" >&2
  exit 1
fi

./build/bin/simple1d

if [ $? -eq 0 ]
then
  echo "Successfully compressed/decompressed simple1d"
else
  echo "Error in compressing/decompressing simple1d" >&2
  exit 1
fi

./build/bin/dim2kplus1

if [ $? -eq 0 ]
then
  echo "Successfully compressed/decompressed dim2kplus1"
else
  echo "Error in compressing/decompressing dim2kplus1" >&2
  exit 1
fi


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
