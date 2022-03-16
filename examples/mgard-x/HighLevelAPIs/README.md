# Compressing with MGARD-X High-level APIs

First, build and install MGARD-X.
Then, run the following in `examples/mgard-x/HighLevelAPIs`.

Build with CMake as follows or use the 'build_scripts.sh'.
```console
$ cmake -S . -B build
$ cmake --build build
$ build/Example
```


`build/main` creates a dataset, compresses it with MGARD-X on NVIDIA GPU, and decomrpess it on CPU.
Read `Example.cpp` to see how the high-level compression API is used.
