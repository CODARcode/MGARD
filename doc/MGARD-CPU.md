#MGARD-CPU

`MGARD-CPU` is a CPU implementation of the MGARD lossy compressor. It is built by defualt and used as base implementation for `MGARD-ROI` and `MGARD-QOI`.

### Building and Installing

To build and install MGARD-CPU, run the following from the root of the repository.
You will need [CMake][cmake] and [Protobuf][protobuf].

```console
$ cmake -S . -B build -D CMAKE_INSTALL_PREFIX=<location to install MGARD>
$ cmake --build build
$ cmake --install build
```

[cmake]: https://cmake.org/
[protobuf]: https://opensource.google/projects/protobuf

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

### Including and Linking

The API consists of a header file `compress.hpp` providing declarations for function templates `mgard::compress` and `mgard::decompress`.
See [the header][api] for documentation of these templates.

To use MGARD in your project, you will need to tell your compiler where to find the MGARD headers (by default, `${CMAKE_INSTALL_PREFIX}/include/mgard/`) and library (by default, `${CMAKE_INSTALL_PREFIX}/lib/`).
If you're using CMake, you can call `find_package(mgard)` and add a dependency to the `mgard::mgard` imported target.
See [the examples directory][examples] for a basic example.

[api]: include/compress.hpp
[examples]: examples/README.md

### Command Line Interface

To build the command line interface, run `cmake` with `-D MGARD_ENABLE_CLI=ON`.
You will need [TCLAP][tclap].
A convenience executable called `mgard` will be built and installed to `${CMAKE_INSTALL_PREFIX}/bin/` by default.
You can get help with the CLI by running the following commands.

```console
$ mgard --help
$ man mgard
```

*This executable is an experimental part of the API.*

[tclap]: http://tclap.sourceforge.net/
