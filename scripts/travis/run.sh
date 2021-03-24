#!/usr/bin/env bash

case ${BUILD_MATRIX_ENTRY} in
  clang-format)
    echo "Running formatting tests"
    if ! "${TRAVIS_BUILD_DIR}/scripts/travis/run-clang-format.sh"; then
      exit 1;
    fi
    ;;
  clang-static-analyzer)
    echo "Running static analysis (clang-analyzer)"
    if ! "${TRAVIS_BUILD_DIR}/scripts/travis/run-clang-static-analyzer.sh"; then
      exit 1;
    fi
    ;;
  cppcheck)
    echo "Running static analysis (cppcheck)"
    if ! "${TRAVIS_BUILD_DIR}/scripts/travis/run-cppcheck.sh"; then
      exit 1;
    fi
    ;;
  tests)
    echo "Running tests"
    if ! "${TRAVIS_BUILD_DIR}/scripts/travis/run-tests.sh"; then
      exit 1;
    fi
    ;;
  *)
    echo "Error: BUILD_MATRIX_ENTRY is undefined or set to an unknown value"
    exit 1;
    ;;
esac
