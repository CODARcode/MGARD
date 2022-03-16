# Compressing with MGARD-X Low-level APIs

First, build and install MGARD-X.
Then, run the following in `examples/mgard-x/LowLevelAPIs/Serial`, `examples/mgard-x/LowLevelAPIs/CUDA`, `examples/mgard-x/LowLevelAPIs/HIP`. Each folder contains a CMake project dedicated for a different kind of processor.

Build with CMake as follows or use the 'build_scripts.sh'.
```console
$ cmake -S . -B build
$ cmake --build build
$ build/Example
```


`build/main` creates a dataset, compresses it with MGARD-X on GPU or CPU, and decomrpess it with the same processor.
Read `Example.cpp/Example.cu` to see how the low-level compression API is used.