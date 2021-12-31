# MGARD [![build status][push workflow badge]][push workflow] [![format status][format workflow badge]][format workflow]

MGARD (MultiGrid Adaptive Reduction of Data) is a technique for multilevel lossy compression of scientific data based on the theory of multigrid methods.
We encourage you to [make a GitHub issue][issue form] if you run into any problems using MGARD, have any questions or suggestions, etc.

[push workflow]: https://github.com/CODARcode/MGARD/actions/workflows/build.yml
[push workflow badge]: https://github.com/CODARcode/MGARD/actions/workflows/build.yml/badge.svg
[format workflow]: https://github.com/CODARcode/MGARD/actions/workflows/format.yml
[format workflow badge]: https://github.com/CODARcode/MGARD/actions/workflows/format.yml/badge.svg
[issue form]: https://github.com/CODARcode/MGARD/issues/new/choose

## Building and Installing

To build and install MGARD, run the following from the root of the repository.
You will need [CMake][cmake] and [Protobuf][protobuf].

```console
$ cmake -S . -B build -D CMAKE_INSTALL_PREFIX=<location to install MGARD>
$ cmake --build build
$ cmake --install build
```

[cmake]: https://cmake.org/
[protobuf]: https://opensource.google/projects/protobuf

### GPU Acceleration

Detailed instructions for using MGARD with GPU acceleration can be found [here][gpu instructions].

[gpu instructions]: doc/MGARD-GPU.md

### Documentation

To build the documentation, run `cmake` with `-D MGARD_ENABLE_DOCS=ON`.
You will need [Doxygen][doxygen].
The documentation will be installed to `${CMAKE_INSTALL_PREFIX}/share/doc/MGARD/` by default.
Open `index.html` with a browser to read.

[doxygen]: https://www.doxygen.nl/

### Benchmarks

To build the benchmarks, run `cmake` with `-D MGARD_ENABLE_BENCHMARKS=ON`.
You will need [Google Benchmark][benchmark].
You can then run the benchmarks with `build/bin/benchmarks`.

[benchmark]: https://github.com/google/benchmark

## Including and Linking

The API consists of a header file `compress.hpp` providing declarations for function templates `mgard::compress` and `mgard::decompress`.
See [the header][api] for documentation of these templates.

To use MGARD in your project, you will need to tell your compiler where to find the MGARD headers (by default, `${CMAKE_INSTALL_PREFIX}/include/mgard/`) and library (by default, `${CMAKE_INSTALL_PREFIX}/lib/`).
If you're using CMake, you can call `find_package(mgard)` and add a dependency to the `mgard::mgard` imported target.
See [the examples directory][examples] for a basic example.

[api]: include/compress.hpp
[examples]: examples/README.md

## Command Line Interface

Assuming the dependencies are met, a convenience executable called `mgard` will be built and installed.
*This executable is an experimental part of the API.*
You can get help with it by running the following commands.

```console
$ mgard --help
$ man mgard
```

## References

MGARD's theoretical foundation and software implementation are discussed in the following papers.
Reference [2] covers the simplest case and is a natural starting point.
Reference [6] covers the design and implementation on GPU heterogeneous systems.

1. Ben Whitney. [Multilevel Techniques for Compression and Reduction of Scientific Data.][thesis] PhD thesis, Brown University, 2018.
2. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Univariate Case.][univariate] *Computing and Visualization in Science* 19, 65–76, 2018.
3. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Multivariate Case.][multivariate] *SIAM Journal on Scientific Computing* 41 (2), A1278–A1303, 2019.
4. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—Quantitative Control of Accuracy in Derived Quantities.][quantities] *SIAM Journal on Scientific Computing* 41 (4), A2146–A2171, 2019.
5. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Unstructured Case.][unstructured] *SIAM Journal on Scientific Computing*, 42 (2), A1402–A1427, 2020.
6. Jieyang Chen et al. [Accelerating Multigrid-based Hierarchical Scientific Data Refactoring on GPUs.][gpu] *35th IEEE International Parallel & Distributed Processing Symposium*, May 17–21, 2021.

[thesis]: https://doi.org/10.26300/ya1v-hn97
[univariate]: https://doi.org/10.1007/s00791-018-00303-9
[multivariate]: https://doi.org/10.1137/18M1166651
[quantities]: https://doi.org/10.1137/18M1208885
[unstructured]: https://doi.org/10.1137/19M1267878
[gpu]: https://arxiv.org/pdf/2007.04457
