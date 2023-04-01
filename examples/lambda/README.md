# MGARD-$\lambda$ 

Data compression while preserving non-linear quantities of interest. 

*MGARD-$\lambda$ is an experimental part of MGARD. Currently only support certain QoIs derived from XGC 5D data.*

#### Prerequisites:

* GCC: version 9
* CUDA: version 11.0.3
* CMake: version > 19.0
* ADIOS2: any version

First, build and install MGARD. Then, build the MGARD- example as follows:

```
$ build_script.sh
$ build/make
```

#### Test dataset is available at: https://www.dropbox.com/s/ak0iaypb12d3ojt/dataset.zip?dl=0

The `TestXGCPostProcess.cpp` shows an example of how to compress tensor data using MGARD, and post process the output to better preserve the QoIs. The basic steps for the post processing are as follows:

*	Decompress data 
*	Compute QoIs with actual data and decompressed data
*	Use Lagrange Optimization method to reduce the QoI errors 
*	Use the Lagrange multipliers to post process the reconstructed data
*	Compute QoIs with the post processed data
*	Compare QoIs computed using the reconstructed data and the post processed data

To run the post processing program use the following command:

`jsrun <machine parameters> <input bp file> <input mesh file> <MGARD tolerance> <MGARD s>`

For example, on the Summit supercomputer, here is an example:

`jsrun -n6 -r6 -c1 -a1 -g1 ./xgc_postprocess ../xgc.f0.00760.bp ../xgc.f0.mesh.bp 1e15 0`

#### Inputs

* Filename of the input data, including its path. The input data must be provided in BP format for this example.
* Filename of mesh data, including its path. The mesh data must be provided in BP format for this example.
* MGARD tolerance parameter which is an L\_inf bound
* MGARD s parameter

#### Outputs
* Compression ratios in display
* Comparison of QoI errors between reconstructed data and post processed data.
* Reconstructed data written out in BP format to file `xgc_compressed.mgard.bp`
* Lagrange multipliers written out in BP format to file `xgc_lagrange.mgard.bp`
