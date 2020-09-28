##How to enable GPU acceleration in MGARD

### Hardware and software requirements
* Supports NVIDIA GPUs: Kepler, Maxwell, Pascal, Volta, Turing, and Ampere
* CUDA 10.0+

### Configuration
* Use CMake to configure:
	* For normal use: configure with command  ```cmake <src_dir> -DMGARD_ENABLE_CUDA=ON``` to enable building the GPU acceleration part.
	* For debugging: configure with command ```cmake <src_dir> -DMGARD_ENABLE_CUDA=ON -DMGARD_ENABLE_CUDA_DEBUG=ON``` to enable building the GPU acceleration part with code that facilitates debugging. However, it will also impact performance.
* The CMake configuration summary will show whether or not the GPU acceleration is configured to be built.
* Finally, build MGARD as usual with ```make```.

### Running MGARD with GPU acceleration

Note that all APIs are templated with support of both double and single precision.

#### Header files
* Add ```mgard_api_cuda.h``` will include all necessary APIs

####Compression/Decompression routines
* To enable decomposition/recomposition on the GPU, certain initialization steps (e.g., preprocessing and memory pre-allocation) need to be done first. However, if the input size is unchanged, those only need to be done once. So, we design GPU-based MGARD to allow separating initialization and compression/decompression. At the same time, for compatibility with the original CPU design we still provide API functions that do self-initialization so that users do not need to explicitly do initialization. Note that using these functions may lead to lower performance. We explain these two ways of using MGARD separately as follows:

	* **Explicit Initialization**
		* First, the object ```mgard_cuda_handle``` needs to be created and initialized. This initializes the necessary environment for efficient decomposition/recomposition on the GPU (e.g., preprocessing, pre-allocation, etc.). It only needs to be created once for fixed input sizes. For example, compressing on the same variable on different timesteps only needs the ```mgard_cuda_handle``` to be created once. It supports both uniform and non-uniform 2D/3D grids. 
			* For uniform grids: ```mgard_cuda_handle(int nrow, int ncol, int nfib)```
			* For non-uniform grids: ```mgard_cuda_handle(int nrow, int ncol, int nfib, T *coords_r, T *coords_c, T *coords_f)```

		* ```mgard_cuda_handle``` also exposes several performance tuning parameters, which is only optional. Even if you don't set them, it will use 'good enough' configurations (althought not guarantee the best). If you prefer to explicitly set those parameters, pass them to the ```mgard_cuda_handle``` at initialization.
			* For uniform grids: ```mgard_cuda_handle(int nrow, int ncol, int nfib, int B, int num_of_queues,
	                    int opt)```
			* For non-uniform grids: ```mgard_cuda_handle(int nrow, int ncol, int nfib, T *coords_r, T *coords_c,
	                    T *coords_f, int B, int num_of_queues, int opt)``` 
				* ```B```: the block size used for turning each internal device kernel (range: 4 - 32)
				* ```num_of_queues```: the concurrency in between internal device kernels (range: 1 - 32)
				* ```opt```: 0 for unoptimized design and 1 for optimized design
		* After a ```mgard_cuda_handle``` object is created, we can call compression/decompression routines as follows:
			* For compression: ```
unsigned char *mgard_compress_cuda(mgard_cuda_handle<T> &handle, T *v,
                                   int &out_size, T tol)``` 
          * For decompression: ```T *mgard_decompress_cuda(mgard_cuda_handle<T> &handle, unsigned char *data,
                         int data_len)```
 * **Implicit Initialization**
 		* For compatibility with the original CPU design, we still provide API functions that do self-initialization so that users do not need to explicitly do initialization.
			* For compression (uniform): ```unsigned char *mgard_compress_cuda(T *data, int &out_size, int n1, int n2,
                                   int n3, T tol)```
       	* For decompression (uniform): ```T *mgard_decompress_cuda(unsigned char *data, int data_len, int n1, int n2,
                         int n3)```  
       	* For compression (non-uniform): ```unsigned char *mgard_compress_cuda(T *data, int &out_size, int n1, int n2,
                                   int n3, std::vector<T> &coords_x,
                                   std::vector<T> &coords_y,
                                   std::vector<T> &coords_z, T tol)``` 
       	* For decompression (non-uniform): ```T *mgard_decompress_cuda(unsigned char *data, int data_len, int n1, int n2,
                         int n3, std::vector<T> &coords_x,
                         std::vector<T> &coords_y, std::vector<T> &coords_z)```                                                   

#### Example
A comprehensive example of using MGARD with GPU acceleration is located in ```test/gpu-cuda```.