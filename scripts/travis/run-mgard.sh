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
  echo "Successfully compress/decompress consine"
else
  echo "Error in compressing/decompressing consine" >&2
  exit 1
fi

./build/bin/constant3d

if [ $? -eq 0 ]
then
  echo "Successfully compress/decompress constant3d"
else
  echo "Error in compressing/decompressing constant3d" >&2
  exit 1
fi

./build/bin/simple1d

if [ $? -eq 0 ]
then
  echo "Successfully compress/decompress simple1d"
else
  echo "Error in compressing/decompressing simple1d" >&2
  exit 1
fi

./build/bin/dim2kplus1

if [ $? -eq 0 ]
then
  echo "Successfully compress/decompress dim2kplus1"
else
  echo "Error in compressing/decompressing dim2kplus1" >&2
  exit 1
fi


cp tests/gray-scott/adios2.xml .
mpirun -n 3 build/bin/gray-scott tests/gray-scott/simulation/settings-files.json

exit 0
