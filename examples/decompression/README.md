# Decompressing with MGARD

First, build and install MGARD.
Then, run the following in `examples/decompression`.

```console
$ cmake -S . -B build
$ cmake --build build
$ build/main
```

`build/main` creates a dataset and then compresses and decompresses it with MGARD.
Read `src/main.cpp` to see how the decompression API is used.
