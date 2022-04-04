# Refactor and progressively reconstruct data with MDR-X

First, build and install MGARD-X.
Then, run the following in `examples/mgard-x/MDR-X/SERIAL`, `examples/mgard-x/MDR-X/CUDA`, `examples/mgard-x/MDR-X/HIP`. Each folder contains a CMake project dedicated for a different kind of processor.

Build with CMake as follows or use the 'build_scripts.sh'.
```console
$ cmake -S . -B build
$ cmake --build build
$ build/Example
```


`build/main` read in a dataset, refactor it with MDR-X on GPU or CPU, and reconstruct it according to the given error bounds.
Read `refactor.cpp/refactor.cu` and `reconstructor.cpp/reconstructor.cu` to see how the MDR-X API is used.

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