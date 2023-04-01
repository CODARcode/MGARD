# XGC Case Study

First, build and install MGARD. Then, build the xgc-casestudy codes as follows:

```
$ mkdir ./build
$ build/cmake ..
$ build/make
```

The example code sets are designed to reproduce the experimental results in [1]. They demonstrate how to better preserve the accuracy of Quantities of Interest (QoI) based on a list of features of MGARD. The demonstration uses the ion f-data simulated by a fusion plasma code XGC (https://xgc.pppl.gov/html/index.html) and a few derived Quantities of Interest (QoI), e.g., density, temperature, flux surface average momentums, etc., as the examples. 

#### Test dataset is available at: https://www.dropbox.com/s/ak0iaypb12d3ojt/dataset.zip?dl=0

#### Feature demonstration:
*	TestXGCBasic
*	Demonstrate the baseline compression using MGARD on uniform grids.
*	Can be parameterized to show compression in dimension from 2D to 4D.
*	Can be parameterized to show the impact of smoothness parameter on the accuracy of QoIs (s>0 to preserve high frequency components and s<0 to preserve low frequency components). 

#### TestXGC4DNonuniform
Demonstrate the compression using MGARD on data resided on non-uniform Cartesian grid. The 4D test XGC data are non-uniform in three dimensions – the mesh nodes, velocity x, and velocity y. The example requires to load mesh structures from an xgc.f0.mesh.bp file. 

#### TestXGC5DTemporal
Demonstrate the compression using MGARD on batched timestep data, showing the benefit of compression across space-time.

#### CalcSnormDensity
Demonstrate how to calculate the operator norm used to preserve the accuracy of QoIs derived from the reduced data.

#### TestXGCDensity
Demonstrate how to bound the errors in a linearly derived QoI, density, during MGARD data compression. The example code requires the operator norms, which can be calculated using CalcSnormDensity. Since the value of operator norm only relates to the QoI function and the grid structure, it only needs to be calculated once for all timestep data.   


Run the command in cmd.txt with XGC test dataset for demonstration or see below for the input/output description:
*	TestXGCBasic: 
    *	[In] Path to the input data
    *	[In] Filename of the input data
    *	[In] Dims (2, 3, 4)
    *	[In] Error tolerance
    *	[In] Smoothness parameter
    *	[Out] Compression ratios in display
    *	[Out] Reconstructed data for evaluation

*	TestXGC4DNonuniform
    *	[In] Path to the input data
    *	[In] Filename of the input data
    *	[In] Error tolerance
    *	[In] Smoothness parameter
    *	[Out] Compression ratios in display
    *	[Out] Reconstructed data for evaluation

*	TestXGC5DTemporal
    *	[In] Path to the input data
    *	[In] Filename of the input data
    *	[In] Error tolerance
    *	[In] Smoothness parameter
    *	[In] Number of timesteps to be batched for compression
    *	[Out] Compression ratios in display
    *	[Out] Reconstructed data for evaluation

*	CalcSnormDensity
    *	[In] metadata file containing the XGC mesh structure 
    *	[Out] Operator norms used to preserve the density during lossy data compression

*	TestXGCDensity
    *	[In] Path to the input data
    *	[In] Filename of the input data
    *	[In] File containing the operator norms used for density preservation (calculated by CalcSnormDensity)
    *	[In] Error tolerance
    *	[In] Number of timesteps to be batched for compression
    *	[Out] Compression ratios in display
    *	[Out] Reconstructed data for evaluation

[1] Qian Gong et al. "Maintaining trust in reduction: Preserving the accuracy of quantities of interest for lossy compression." Driving Scientific and Engineering Discoveries Through the Integration of Experiment, Big Data, and Modeling and Simulation: 21st Smoky Mountains Computational Sciences and Engineering, SMC 2021, Virtual Event, October 18-20, 2021, Revised Selected Papers. Cham: Springer International Publishing, 2022. 22-39.

  
