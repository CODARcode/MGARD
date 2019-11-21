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
  exit 0
else
  echo "Error in compressing/decompressing consine" >&2
  exit 1
fi

./build/bin/constant3d

if [ $? -eq 0 ]
then
  echo "Successfully compress/decompress constant3d"
  exit 0
else
  echo "Error in compressing/decompressing constant3d" >&2
  exit 1
fi

