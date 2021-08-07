# Compressing with MGARD

First, build and install MGARD.
Then, run the following in `examples/compression`.

```console
$ cmake -S . -B build
$ cmake --build build
$ build/main
```

`build/main` creates a dataset, compresses it with MGARD, and reports the compression ratio.
Read `src/main.cpp` to see how the compression API is used.
