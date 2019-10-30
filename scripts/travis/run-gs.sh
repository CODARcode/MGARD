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

wget https://github.com/ornladios/ADIOS2/archive/v2.5.0.tar.gz
tar -zxvf v2.5.0.tar.gz

cd ${SOURCE_DIR}


./scripts/

# Check python code with flake8
if ! ~/.local/bin/flake8 --config=flake8.cfg .
then
  exit 3
fi

exit 0
