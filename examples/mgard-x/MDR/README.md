# Refactor and progressively reconstruct data with MDR-X

First, build and install MGARD-X.
Then, run the following in `examples/mgard-x/MDR/Serial`, `examples/mgard-x/MDR/CUDA`, `examples/mgard-x/MDR/HIP`. Each folder contains a CMake project dedicated for a different kind of processor.

Build with CMake as follows or use the 'build_scripts.sh'.
```console
$ cmake -S . -B build
$ cmake --build build
$ build/Example
```


`build/main` read in a dataset, refactor it with MDR-X on GPU or CPU, and reconstruct it according to the given error bounds.
Read `refactor.cpp/refactor.cu` and `reconstructor.cpp/reconstructor.cu` to see how the MDR API is used.