##Readme for MGARD-GPU
### Supporting features
* Double and single precision data types
* High dimensional data (upto 4D)
* Compression with S-norm
* Uniform and non-uniform data
* End-to-end high performance compression on GPUs
* Performance pretuned for Volta and Turing GPUs 

### Hardware and software requirements
* NVIDIA GPUs ( tested on Volta, Turing)
* CUDA 11.0+
* GCC 7.4.0+
* CMake 3.19+

### Software dependencies 
* NVCOMP: https://github.com/NVIDIA/nvcomp.git

### Configure and build
* **Option 1:** one-step configure and build MGARD with NVCOMP: ```build_mgard_cuda.sh```.
* **Option 2:** manual configre and build
	+ **Step 1:** configure and build NVCOMP.
	+ **Step 2:** configure MGARD as follows:

			
			cmake -S <MGARD_SRC_DIR> -B <MGARD_BUILD_DIR>
				  -DMGARD_ENABLE_CUDA=ON
				  -DNVCOMP_ROOT=<NVCOMP_INSTALL_DIR> 
			
	+ **Step 3:** build MGARD: ```cmake --build <MGARD_BUILD_DIR> -j8```

### Using MGARD-GPU APIs


* **Step 1**: MGARD-GPU APIs are included in both ```mgard_api.h``` and ```mgard_cuda_api.h```

	+ Use ```mgard_api.h``` if the user programs are to be compiled with ***C/C++*** compilers.
	+ Use ```mgard_cuda_api.h``` if the user programs are to be compiled with ***CUDA*** compilers.

* **Step 2**: an object ```mgard_cuda::mgard_cuda_handle``` needs to be created and initialized. This initializes the necessary environment for efficient decomposition/recomposition on the GPU. It only needs to be created once if the input shape is not changed. For example, compressing on the same variable on different timesteps only needs the handle to be created once. Also, the same handle can be shared in between compression and decompression APIs.

 * For ***uniform grids***: ```mgard_cuda::mgard_cuda_handle<D_type, N_dims>(std::vector<size_t> shape)```.
  - ```[In] D_type```: intput data type (float or double).
  - ```[In] N_dims```: total number of dimensions (<=4)
  - ```[In] shape```: stores the size in each dimension with the first being the leading dimension (fastest).
 * For ***non-uniform grids***: ```mgard_cuda::mgard_cuda_handle<D_type, N_dims>(std::vector<size_t> shape, std::vector<T*> coords)```. 
  - ```[In] coords```: the cooordinates in each dimension with the first being the leading dimension (fastest).
* **Step 3**: calling compression/decompression routines as follows:
  	+ For ***compression***: ```
			unsigned char *mgard_cuda::compress(mgard_cuda_handle<D_type, N_dims> &handle, D_type *v, size_t &out_size, D_type tol, D_type s)```
     	- ```[In] D_type *v```: input data stored in CPU memory.
	  	- ```[Out] size_t &out_size```: compressed size in number of bytes.
	  	- ```[In] D_type tol```: relative L_inf error bound.
	  	- ```[In] D_type s```: S-norm.
	  	- ```[Return]```: compressed data in CPU memory.
  	+ For ***decompression***: ```T *mgard_cuda::decompress(mgard_cuda_handle<D_type, N_dims> &handle, unsigned char *data, size_t data_len)```    
  		- ```[In] unsigned char *data```: compressed data stored in CPU memory.
   		- ```[In] size_t data_len```: size of compressed data in number of bytes.
  		- ```[Return]```: decompressed data in CPU memory.

### Performance optimization

* **Optimize for specific GPU architectures:** MGARD-GPU is pretuned for Volta and Turing GPUs. To enable this optimization, the follow CMake options need to be enable when during configuration:
	+ ***For Volta GPUs:*** ```-DMGARD_ENABLE_CUDA_OPTIMIZE_VOLTA=ON```
	+ ***For Turing GPUs:*** ```-DMGARD_ENABLE_CUDA_OPTIMIZE_TURING=ON```
* **Optimize for fast CPU-GPU data transfer speed:** It is recommanded to use pinned memory on CPU for holding the input data such that it can enable fast CPU-GPU data transfer. To allocate pinned memory on CPU, the following API can be used:

		mgard_cuda::cudaMallocHostHelper(void ** data_prt, size_t size)
	                                      

### Simple example
The following code shows how to compress/decompress a 3D dataset. 

		#include <vector>
		#include "mgard_api.h"
		int main() 
		{
			double *in_buff;
			size_t n1 = 10;
			size_t n2 = 20;
			size_t n3 = 30;
			mgard_cuda::cudaMallocHostHelper((void **)&in_buff, sizeof(double)*n1*n2*n3);
			//... load data to in_buff
			std::vector<size_t> shape{ n1, n2, n3 };
			mgard_cuda::mgard_cuda_handle<double, 3> handle(shape);
			size_t out_size;
			double tol = 0.01, s = 0;
		   unsigned char * mgard_comp_buff = mgard_cuda::compress(handle, in_buff, out_size, tol, s);
		   double * mgard_out_buff = mgard_cuda::decompress(handle, mgard_comp_buff, out_size);
		   mgard_cuda::cudaFreeHostHelper(in_buff);
		}

### Comprehensive example
* A comprehensive example of using MGARD-GPU is located in ```test/gpu-cuda```.