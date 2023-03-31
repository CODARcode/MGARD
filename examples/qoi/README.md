# Data Compression Preserving Quantity-of-Interest (QoI) using MGARD

First, build and install MGARD. Then, build the MGARD-QoI example as follows:

```
$ build/cmake ..
$ build/make
```

The average.cpp shows an example of how to preserve the average quantities computed from data in a bounding box defined as `[x_l, x_r; y_l, y_r]`, during lossy compression. The code generates a 2D sinusoid data with random values, compresses it according to the prescribed error bounds placed on QoI (i.e., average), and reconstructs it to check the errors in primary data and the QoI.  

Below shows the inputs/outputs of the executable average:

* Inputs
   * Dim 1 of the input data (default value 128)
   * Dim 2 of the input data (default value 128)
   * Lower left coordinate of the bounding box in dim 1
   * Lower left coordinate of the bounding box in dim 2
   * Upper right coordinate of the bounding box in dim 1
   * Upper right coordinate of the bounding box in dim 2
   * Error tolerance prescribed on QoI (i.e., average in this example)
* Outputs in display
   * Compression ratio
   * Error of QoI vs the prescribed error bound 
