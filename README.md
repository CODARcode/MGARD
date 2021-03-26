# MGARD [![Build Status][travis status]][travis]

MGARD (MultiGrid Adaptive Reduction of Data) is a technique for multilevel lossy compression of scientific data based on the theory of multigrid methods.
This is an experimental C++ implementation for integration with existing software; use at your own risk!
We encourage you to [make a GitHub issue][issue form] if you run into any problems using MGARD, have any questions or suggestions, etc.

## Use MGARD

The API consists of a header file `include/mgard_api.h` providing declarations for function templates `mgard::compress` and `mgard::decompress`.
See [the header][api] for documentation of these templates.

[travis]: https://travis-ci.org/CODARcode/MGARD
[travis status]: https://travis-ci.org/CODARcode/MGARD.svg?branch=master
[issue form]: https://github.com/CODARcode/MGARD/issues/new/choose
[api]: include/mgard_api.h

## Building

To build and install MGARD, run the following from the root of the repository. You will need [CMake][cmake].

```console
$ cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<location to install MGARD> ..
$ cmake --build build
$ cmake --install build
```

By default, headers will be installed to `${CMAKE_INSTALL_PREFIX}/include/mgard/` and a library will be installed to `${CMAKE_INSTALL_PREFIX}/lib/`.
To use MGARD in your project, provide these locations to your compiler.
If you're using CMake, you can call `find_package(mgard)` and add a dependency to the `mgard::mgard` imported target.
See [the examples directory][examples] for a basic example.

[cmake]: https://cmake.org/
[examples]: examples/

### GPU acceleration

See [here][gpu] for detailed instructions for using MGARD with GPU acceleration.

[gpu]: README_MGARD_GPU.md

## References

The theory foundation and software implementation behind MGARD are developed in the following papers, which also address implementation issues and present numerical examples.
Reference [2] covers the simplest case and is a natural starting point.
Reference [6] covers its design and implementation on GPU heterogeneous systems.

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
