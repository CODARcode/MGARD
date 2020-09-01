# MGARD [![Build Status][travis status]][travis]

MGARD (MultiGrid Adaptive Reduction of Data) is a technique for multilevel lossy compression of scientific data based on the theory of multigrid methods.
This is an experimental C++ implementation for integration with existing software; use at your own risk!
We encourage you to [make a GitHub issue][issue form] if you run into any problems using the software, have any questions or suggestions, etc.

The API consists of a header file `include/mgard_api.h` providing prototypes for overloaded functions `mgard_compress` and `mgard_decompress`.
See [the header][api] for documentation of these functions.

To use MGARD,

1. Run `make lib/libmgard.a` to generate a static library.
2. Include `mgard_api.h` in any source files making use of the API.
3. Link against `libmgard.a` when creating your executable.

[travis]: https://travis-ci.org/CODARcode/MGARD
[travis status]: https://travis-ci.org/CODARcode/MGARD.svg?branch=master
[issue form]: https://github.com/CODARcode/MGARD/issues/new/choose
[api]: include/mgard_api.h

## References

The theory behind MGARD is developed in the following papers, which also address implementation issues and present numerical examples.
Reference [2] covers the simplest case and is a natural starting point.

1. Ben Whitney. [Multilevel Techniques for Compression and Reduction of Scientific Data.][thesis] PhD thesis, Brown University, 2018.
2. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Univariate Case.][univariate] *Computing and Visualization in Science* 19, 65–76, 2018.
3. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Multivariate Case.][multivariate] *SIAM Journal on Scientific Computing* 41 (2), A1278–A1303, 2019.
4. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—Quantitative Control of Accuracy in Derived Quantities.][quantities] *SIAM Journal on Scientific Computing* 41 (4), A2146–A2171, 2019.
5. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Unstructured Case.][unstructured] *SIAM Journal on Scientific Computing*, 42 (2), A1402–A1427, 2020.

[thesis]: https://doi.org/10.26300/ya1v-hn97
[univariate]: https://doi.org/10.1007/s00791-018-00303-9
[multivariate]: https://doi.org/10.1137/18M1166651
[quantities]: https://doi.org/10.1137/18M1208885
[unstructured]: https://doi.org/10.1137/19M1267878

## Caveats

If you use a certain value of `s` to compress your data, *you must use the same value of `s` to decompress it*.
You cannot agnostically decompress the compressed representation, and the value of `s` is not currently stored in the compressed stream, so if you forget the value of `s` that you used when compressing your data, your data is gone.
In addition, there is currently no way to detect if an inconsistent value of `s` has been passed, so the code returns corrupted data silently.
