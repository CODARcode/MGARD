#MGARD-GPU

MGARD-GPU is a CUDA implementation of the MGARD lossy compressor, which significantly improves MGARD's throughput in GPU-based heterogeneous HPC systems.

## Supporting features
* Double and single precision data types
* High dimensional data (upto 4D)
* Compression with S-norm
* Uniform and non-uniform data
* End-to-end high performance compression on GPUs
* Performance pretuned for Volta and Turing GPUs 

## Hardware and software requirements
* NVIDIA GPUs ( tested on Volta, Turing)
* CUDA 11.0+
* GCC 7.4.0+
* CMake 3.19+

## Software dependencies 
* [NVCOMP][nvcomp]

[nvcomp]: https://github.com/NVIDIA/nvcomp.git
## Configure and build
* **Option 1:** one-step configure and build MGARD with NVCOMP: ```build_mgard_cuda.sh```
* **Option 2:** manual configre and build
	+ **Step 1:** configure and build NVCOMP.
	+ **Step 2:** configure MGARD as follows:

			cmake -S <MGARD_SRC_DIR> -B <MGARD_BUILD_DIR>
				  -DMGARD_ENABLE_CUDA=ON
				  -DNVCOMP_ROOT=<NVCOMP_INSTALL_DIR> 
			
	+ **Step 3:** build MGARD: ```cmake --build <MGARD_BUILD_DIR> -j8```

## Using MGARD-GPU APIs

* **Step 1: Include the header file.** MGARD-GPU APIs are included in both ```magrd/mgard_api.h``` and ```mgard/mgard_cuda_api.h```.
     + Use ```mgard/mgard_api.h``` if the user programs are to be compiled with ***C/C++*** compilers.
     + Use ```mgard/mgard_cuda_api.h``` if the user programs are to be compiled with ***CUDA*** compilers.

* **Step 2: Initialize mgard_cuda::Handle.**
An object ```mgard_cuda::Handle``` needs to be created and initialized. This initializes the necessary environment for efficient compression on the GPU. It only needs to be created once if the input shape is not changed. For example, compressing on the same variable on different timesteps only needs the handle to be created once. Also, the same handle can be shared in between compression and decompression APIs.
     + For ***uniform grids***: ```mgard_cuda::Handle<D_type, N_dims>(std::vector<size_t> shape)```.
        + ```[In] D_type```: intput data type (float or double).
        + ```[In] N_dims```: total number of dimensions (<=4)
        + ```[In] shape```: stores the size in each dimension with the first being the leading dimension (fastest).
     + For ***non-uniform grids***: ```mgard_cuda::Handle<D_type, N_dims>(std::vector<size_t> shape, std::vector<T*> coords)```. 
        + ```[In] coords```: the cooordinates in each dimension with the first being the leading dimension (fastest).
 
* **Step 3: Use mgard_cuda::Array.** ```mgard_cuda::Array``` is used to hold a managed array on GPU.
	  +  For ***creating*** an array. ```mgard_cuda::Array::Array<D_type, N_dims>(std::vector<size_t> shape)``` creates an manged array on GPU with ```shape```.
	  +  For ***loading data*** into an array. ```void mgard_cuda::Array::loadData(D_type *data, size_t ld = 0)``` copies ```data``` into the the managed array on GPU. ```data``` can be on either on CPU or GPU. An optional ```ld``` can be provided for specifying the size of the leading dimension.
	  +  For ***accessing data from CPU*** ```D_type * mgard_cuda::Array::getDataHost()``` returns a CPU pointer of the array.
	  +  For ***accessing data from GPU***```D_type * mgard_cuda::Array::getDataDevice(size_t &ld)``` returns a GPU pointer of the array with the leading dimension.
	  +  For ***getting the shape*** of an array. ```std::vector<size_t> mgard_cuda::Array::getShape()``` returns the shape of the managed array.

   ***Note:*** ```mgard_cuda::Array``` will automatically release its internal CPU/GPU array when it goes out of scope.

* **Step 4: Call compression/decompression API.**:
  	+ For ***compression***: ```
			mgard_cuda::Array<unsigned char, 1> mgard_cuda::compress(mgard_cuda::Handle<D_type, N_dims> &handle, mgard_cuda::Array<D_type, N_dims> in_array, D_type tol, D_type s)```
     	- ```[In] in_array ```: input data to be compressed (it value will be altered during compression).
	  	- ```[In] tol```: relative L_inf error bound.
	  	- ```[In] s```: S-norm.
	  	- ```[Return]```: compressed data.
  	+ For ***decompression***: ```mgard_cuda::Array<D_type, N_dims> mgard_cuda::decompress(mgard_cuda::Handle<D_type, N_dims> &handle, mgard_cuda::Array<unsigned char, 1> compressed_data)```    
  		- ```[In] compressed_data ```: compressed data.
  		- ```[Return]```: decompressed data.

## Performance optimization

* **Optimize for specific GPU architectures:** MGARD-GPU is pretuned for Volta and Turing GPUs. To enable this optimization, one of the following additional CMake options need to be enabled when configuring MGARD-GPU. ***Please note***: only one option can be turned on.
	+ For ***Volta*** GPUs: ```-DMGARD_ENABLE_CUDA_OPTIMIZE_VOLTA=ON```
	+ For ***Turing*** GPUs: ```-DMGARD_ENABLE_CUDA_OPTIMIZE_TURING=ON```
* **Optimize for fast CPU-GPU data transfer:** It is recommanded to use pinned memory on CPU for loading data into ```mgard_cuda::Array``` such that it can enable fast CPU-GPU data transfer. 
	+ To allocate pinned memory on CPU: ```mgard_cuda::cudaMallocHostHelper(void ** data_ptr, size_t size)```.
	+ To free pinned memory on CPU: ```mgard_cuda::cudaFreeHostHelper(void * data_ptr)```
	                                      

## A simple example
The following code shows how to compress/decompress a 3D dataset. 

		#include <vector>
		#include <iostream>
		#include "mgard/mgard_api.h"
		int main() 
		{
		  size_t n1 = 10;
		  size_t n2 = 20;
		  size_t n3 = 30;
		
		  //prepare 
		  std::cout << "Preparing data...";
		  double * in_array_cpu;
		  mgard_cuda::cudaMallocHostHelper((void **)&in_array_cpu, sizeof(double)*n1*n2*n3);
		  //... load data into in_array_cpu
		  std::vector<size_t> shape{ n1, n2, n3 };
		  mgard_cuda::Handle<double, 3> handle(shape);
		  mgard_cuda::Array<double, 3> in_array(shape);
		  in_array.loadData(in_array_cpu);
		  std::cout << "Done\n";
		
		  std::cout << "Compressing with MGARD-GPU...";
		  double tol = 0.01, s = 0;
		  mgard_cuda::Array<unsigned char, 1> compressed_array = mgard_cuda::compress(handle, in_array, tol, s);
		  size_t compressed_size = compressed_array.getShape()[0]; //compressed size in number of bytes.          
		  unsigned char * compressed_array_cpu = compressed_array.getDataHost();
		  std::cout << "Done\n";
		
		  std::cout << "Decompressing with MGARD-GPU...";
		  // decompression
		  mgard_cuda::Array<double, 3> decompressed_array = mgard_cuda::decompress(handle, compressed_array);
		  mgard_cuda::cudaFreeHostHelper(in_array_cpu);
		  double * decompressed_array_cpu = decompressed_array.getDataHost();
		  std::cout << "Done\n";
	}

## A comprehensive example
* A comprehensive example about how to use MGARD-GPU is located in [here][example].

[example]:tests/gpu-cuda

## Citing MGARD-GPU
***Kindly note:*** If you mention MGARD-GPU in your paper, please cite using:

```
@article{mgard-gpu,
   title = {Accelerating Multigrid-based Hierarchical Scientific Data Refactoring on GPUs},
  author = {Chen, Jieyang and Wan, Lipeng and Liang, Xin and Whitney, Ben and Liu, Qing and Pugmire, Dave and Thompson, Nicholas and Wolf, Matthew and Munson, Todd and Foster, Ian and others},
 journal = {arXiv preprint arXiv:2007.04457},
    year = {2020}
}
```