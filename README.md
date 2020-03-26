## MGARD

MGARD (MultiGrid Adaptive Reduction of Data) is a technique for multilevel lossy compression of scientific data based on the theory of multigrid methods.
This is an experimental C++ API for integration with existing software, use at your own risk!

The double precision API consists of a header file: `mgard_api.h`, for single precision the header file is `mgard_api_float.h`.

Users need only include this header file, and link against the static
library `libmgard.a`.

This header file provides prototypes for the following overloaded functions:

```
unsigned char *mgard_compress(int itype_flag, double/float *data, int& out_size,
                              int n1, int n2, int n3, double/float tol, [qoi, s = infinity])
```

It returns a pointer to an `unsigned char` array of compressed data.
The arguments are:

     itype_flag: Data type, 0 for float, 1 for double
     data : Pointer to the input buffer (interpreted as 2D matrix) to compress with MGARD,
            this buffer will be destroyed!
     out_size: size of input data, returns compressed size on exit
     n1: Size of first dimension
     n2: Size of second dimension
     n3: Size of third dimension
     tol: Upper bound for desired tolerance. Note that this tolerance is relative not absolute: ||u - C[u]||_s \le tol*||u||_s
     qoi: Function pointer to the quantity of interest
     s: The norm in which the error will be preserved, L-\infty assumed if not present in the function call.

# These APIs are outdated and will be updated. 
The next overload is

```
void *mgard_decompress(int itype_flag, unsigned char *data,
int data_len, int n1, int n2, int n3,[s = infinity])
```

This returns a void pointer to an array of decompressed data, which must be cast to `float` or `double`.
The arguments are:

     itype_flag: Data type, 0 for float, 1 for double
     data : Pointer to the input buffer (interpreted as 2D matrix) to decompress with MGARD,
            this buffer will be destroyed!
     data_len: size of input data in bytes
     n1: Size of first dimension
     n2: Size of second dimension
     n3: Size of third dimension
     s: The norm in which the error will be preserved, L-\infty assumed if not present in the function call.

The `qoi` function pointer must compute the quantity of interest, *Q(v)*.
Its only use is to estimate the Besov *s*-norm of the operator *Q*; if this can be derived independently, then there is no need to provide it.


## References

The theory behind MGARD is developed in the following papers, which also address implementation issues and present numerical examples.
Reference [2] covers the simplest case and is a natural starting point.

1. Ben Whitney. [Multilevel Techniques for Compression and Reduction of Scientific Data.][thesis] PhD thesis, Brown University, 2018.
2. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Univariate Case.][univariate] *Computing and Visualization in Science* 19, 65–76, 2018.
3. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—The Multivariate Case.][multivariate] *SIAM Journal on Scientific Computing* 41 (2), A1278–A1303, 2019.
4. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. [Multilevel Techniques for Compression and Reduction of Scientific Data—Quantitative Control of Accuracy in Derived Quantities.][quantities] *SIAM Journal on Scientific Computing* 41 (4), A2146–A2171, 2019.
5. Mark Ainsworth, Ozan Tugluk, Ben Whitney, and Scott Klasky. Multilevel Techniques for Compression and Reduction of Scientific Data—The Unstructured Case. *SIAM Journal on Scientific Computing*, to appear.

[thesis]: https://doi.org/10.26300/ya1v-hn97
[univariate]: https://doi.org/10.1007/s00791-018-00303-9
[multivariate]: https://doi.org/10.1137/18M1166651
[quantities]: https://doi.org/10.1137/18M1208885

## Caveats

If you use a certain value of `s` to compress your data, *you must use the same value of `s` to decompress it*.
You cannot agnostically decompress the compressed representation, and the value of `s` is not stored in the compressed stream.
In addition, there is currently no way to detect if an inconsistent value of `s` has been passed, so the code returns corrupted data silently.

If you forget the value of `s` that you used to compress your data, then your data is gone.
