# Refactor and progressively reconstruct data with MDR

First, build and install MGARD-X.
Then, run the following in `examples/mgard-x/MDR`.

Build with CMake as follows or use the 'build_scripts.sh'.
```console
$ cmake -S . -B build
$ cmake --build build
$ build/Example
```


`build/main` read in a dataset, refactor it with MDR, and reconstruct it according to the given error bounds.
Read `refactor.cpp` and `reconstructor.cpp` to see how the MDR API is used.

The exectuables `refactor` and `reconstructor` can be used as follows:

* `refactor` 
	- `<input data>`
	- `<number of decomposition levels>` 
	- `<number of bitplanes>`
	- `<number of dimensions N> <dim 1> <dim 2> .. <dim N>`

* `reconstructor`
 	- `<original data>`
 	- `<error mode>`: 0 for L\_inf error; 1 for L\_2 error
 	- `<number of tolerances M> <tol 1> <tol 2> ... <tol M>`
 	- `<s>`