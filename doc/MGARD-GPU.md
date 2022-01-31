
# MGARD-GPU

MGARD-GPU is a CUDA implementation of the MGARD lossy compressor, which significantly improves MGARD's compression/decomrpession throughput via highly optimized GPU kernels.

## Supporting features
* **Data type:** Double and single precision floating-point data
* **Dimensions:** 1D-5D
* **Error-bound type:** L_Inf error and L_2 error
* **Error-bound mode:** Absoluate and relative
* **Data structure:** Uniform and non-uniform spaced Cartisan gird
* **Three sets of interfaces**:
  * **Command Line Inteface:** Full-featured command line executable for compression/decompression. 
  * **High-level APIs:** Encapsulated all GPU-related details for easiest integration with user's program.
  * **Low-level APIs:** Users can fully control different steps of compression (the preprocessing step, GPU buffers allocations, GPU compression kernel invokation, GPU-CPU data copy) for more fexiable and high performance compression.
  

## Hardware and software requirements
* NVIDIA GPUs ( tested on Volta, Turing)
* CUDA 11.0 and later
* CUDA compatiable C++ compiler ([Details](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html))
* CMake 3.19 and later

## Software dependencies 
* [NVCOMP][nvcomp]

[nvcomp]: https://github.com/NVIDIA/nvcomp.git
## Configure and build
* **Option 1:** one-step configure and build MGARD with NVCOMP: ```build_mgard_x.sh```
* **Option 2:** manual confiugre and build
	+ **Step 1:** configure and build NVCOMP.
	+ **Step 2:** configure MGARD as follows:

			cmake -S <MGARD_SRC_DIR> -B <MGARD_BUILD_DIR>
				  -DMGARD_ENABLE_CUDA=ON
				  -DCMAKE_PREFIX_PATH=<NVCOMP_INSTALL_DIR> 
			
	+ **Step 3:** build MGARD: ```cmake --build <MGARD_BUILD_DIR> -j8```

## Using command line interface (CLI)
* An executable ```MgardCudaExec``` will be built when building the MGARD-GPU library.
* To use the ```MgardCudaExec``` CLI, here are the options:

  + ```-z```: compress data
     + ```-i <path>``` path to data file to be compressed
     + ```-c <path>``` path to compressed file
     + ```-t <s|d>``` data type (s: single; d:double)
     + ```-n <D>``` total number of dimensions
       + ```<n_1>``` slowest dimention
       + ```<n_2>``` 2nd slowest dimention
       + ...
       + ```<n_D>``` fastest dimention
     + ```-u <path>``` path to coordinate file (non-uniform only)
     + ```-m <abs|rel>``` error bound mode (abs: abolute; rel: relative)
     + ```-e <error>``` error bound
     + ```-s <smoothness>``` smoothness parameter
     + ```-l <1|2|3>``` choose lossless compressor (0:CPU 1:Huffman@GPU 2:Huffman@GPU+LZ4@GPU)
  + ```-x```: decompress data
    + ```-c <path>``` path to compressed file
    + ```-d <path>``` path to decompressed file
  + ```-v``` enable verbose (show timing and statistics)
	
## For Using both the high-level APIs and low-level API
* **Include the header file.** MGARD-GPU APIs are included in both ```mgard/compress.hpp``` and ```mgard/compress_cuda.hpp```.
     + Use ```mgard/compress.hpp``` if the user programs are to be compiled with ***C/C++*** compilers.
     + Use ```mgard/compress_cuda.hpp``` if the user programs are to be compiled with ***CUDA*** compilers.
* **Configure using ```mgard_x::Config```** Both high-level APIs and low-level APIs have an optional parameter for users to configure the compression/decomrpession process via ```mgard_x::Config``` class. To configure, create a ```mgard_x::Config``` object and configure its fields:
  + ```Config.dev_id```: sepcifying a specific GPU to use in multi-GPU systems.
  + ```Config.timing```: timing each steps of compression and printing them out.
  + ```Config.lossless```: control the lossless compression used: 
    + ```mgard_x::lossless_type::CPU_Lossless```: CPU lossless (ZLIB/ZSTD)
    + ```mgard_x::lossless_type::GPU_Huffman```: GPU Huffman compression
    + ```mgard_x::lossless_type::GPU_Huffman_LZ4```: GPU Huffman and LZ4 compression
    + *Note:* there will be no effect configuring the lossless comrpessor for decompression as MGARD has to use the same lossless compressor that was used for compression.
## Using high-level APIs
* **For compression:** ```void mgard_x::compress(mgard_x::DIM D, mgard_x::data_type dtype, std::vector<mgard_x::SIZE> shape, double tol, double s, enum error_bound_type mode, const void *original_data, void *&compressed_data, size_t &compressed_size, mgard_x::Config config)```
  + ```[In] shape:``` Shape of the Dataset to be compressed (from slowest to fastest).
  + ```[In] data_type:``` mgard_x::data_type::Float or mgard_x::data_type::Double.
  + ```[In] type:``` mgard_x::error_bound_type::REL or mgard_x::error_bound_type::ABS.
  + ```[In] tol:``` Error tolerance.
  + ```[In] s:``` Smoothness parameter.
  + ```[In] compressed_data:``` Dataset to be compressed.
  + ```[Out] compressed_size:``` Size of comrpessed data.
  + ```[In][Optional] coords```: The coordinates in each dimension (from slowest to fastest).
  + ```[in][Optional] config:``` For configuring the compression process (optional).

* **For decompression:** ```void decompress(const void *compressed_data, size_t compressed_size, void *&decompressed_data, Config config)```
  + ```[In] compressed_data:``` Compressed data.
  + ```[In] compressed_size:``` Size of comrpessed data.
  + ```[Out] decompressed_data:``` Decompressed data.
  + ```[In][Optional] config:``` For configuring the decompression process (optional).

## Using low-level APIs
* **Step 2: Initialize mgard_x::Handle.**
An object ```mgard_x::Handle``` needs to be created and initialized. This initializes the necessary environment for efficient compression on the GPU. It only needs to be created once if the input shape is not changed. For example, compressing the same variable on different timesteps only needs the handle to be created once. Also, the same handle can be shared in between compression and decompression APIs.
     + ```mgard_x::Handle<N_dims, D_type>(std::vector<size_t> shape, std::vector<T*> coords, mgard_x::Config config)```.
        + ```[In] D_type```: Input data type (float or double).
        + ```[In] N_dims```: Total number of dimensions (<=4)
        + ```[In] shape```: Stores the size in each dimension (from slowest to fastest).
        + ```[In][Optional] coords```: The coordinates in each dimension (from slowest to fastest).
      	+ ```[In][Optional] config```: For configuring compression/decomrpession.
* **Step 3: Use mgard_x::Array.** ```mgard_x::Array``` is used for holding a managed array on GPU.
     +  For ***creating*** an array. ```mgard_x::Array::Array<N_dims, D_type>(std::vector<size_t> shape)``` creates an manged array on GPU with ```shape```.
     +  For ***loading data*** into an array. ```void mgard_x::Array::loadData(D_type *data, size_t ld = 0)``` copies ```data``` into the the managed array on GPU. ```data``` can be on either on CPU or GPU. An optional ```ld``` can be provided for specifying the size of the leading dimension.
     +  For ***accessing data from CPU*** ```D_type * mgard_x::Array::getDataHost()``` returns a CPU pointer of the array.
     +  For ***accessing data from GPU***```D_type * mgard_x::Array::getDataDevice(size_t &ld)``` returns a GPU pointer of the array with the leading dimension.
     +  For ***getting the shape*** of an array. ```std::vector<size_t> mgard_x::Array::getShape()``` returns the shape of the managed array.

   ***Note:*** ```mgard_x::Array``` will automatically release its internal CPU/GPU array when it goes out of scope.

* **Step 4: Query specifications of original data from compressed data** In case the data type/structure/shape are unknown when decompression, the following APIs can be use to infer those information

  + For **infering data type**: ```enum mgard_x::data_type mgard_x::infer_data_type(const void *compressed_data, size_t compressed_size)```
    + ```[In] compressed_data:``` Compressed data.
    + ```[In] compressed_size:``` Size of comrpessed data.
    + ```[Return] Data type```
  + For **infering data shape**: ```std::vector<mgard_x::SIZE> mgard_x::infer_shape(const void *compressed_data, size_t compressed_size)```
    + ```[In] compressed_data:``` Compressed data.
    + ```[In] compressed_size:``` Size of comrpessed data.
    + ```[Return] Data shape```
  + For **infering data structure**: ```enum mgard_x::data_structure infer_data_structure(const void *compressed_data, size_t compressed_size)```
    + ```[In] compressed_data:``` Compressed data.
    + ```[In] compressed_size:``` Size of comrpessed data.
    + ```[Return] Data structure```
  + For **infering data structure**: ```std::vector<T *> infer_coords(const void *compressed_data, size_t compressed_size)```
    + ```[In] compressed_data:``` Compressed data.
    + ```[In] compressed_size:``` Size of comrpessed data.
    + ```[Return] Coordinates```          
* **Step 4: Invoke compression/decompression.**:
  	+ For ***compression***: ```
			mgard_x::Array<1, unsigned char> mgard_x::compress(mgard_x::Handle<N_dims, D_type> &handle, mgard_x::Array<N_dims, D_type> in_array, mgard_x::error_bound_type type, D_type tol, D_type s)```
     	- ```[In] in_array ```: Input data to be compressed (its value will be altered during compression).
     	- ```[In] type ```: Error bound type. ```mgard_x::REL``` for relative error bound or ```mgard_x::ABS``` for absolute error bound. 
	  	- ```[In] tol```: Error bound.
	  	- ```[In] s```: Smoothness parameter.
	  	- ```[Return]```: Compressed data.
  	+ For ***decompression***: ```mgard_x::Array<N_dims, D_type> mgard_x::decompress(mgard_x::Handle<N_dims, D_type> &handle, mgard_x::Array<1, unsigned char> compressed_data)```    
  		- ```[In] compressed_data ```: Compressed data.
  		- ```[Return]```: Decompressed data.
	
## Performance optimization

* **Optimize for specific GPU architectures:** MGARD-GPU is pre-tuned for Volta and Turing GPUs. To enable this optimization, the following additional CMake options need to be enabled when configuring MGARD-GPU.
	+ For ***Volta*** GPUs: ```-DMGARD_ENABLE_CUDA_OPTIMIZE_VOLTA=ON```
	+ For ***Turing*** GPUs: ```-DMGARD_ENABLE_CUDA_OPTIMIZE_TURING=ON```
* **Optimize for GPUs on edge systems:** MGARD-GPU capable of using FMA instructions that can help improve the compression/decompression performance on consumer-class GPUs (GeForce GPUs on edge systems), where they have relative low throughput on some arithmetic operations. To enable this optimization, enable the following option when configuring MGARD-GPU. 
	+ ```-MGARD_ENABLE_CUDA_FMA=ON```
* **Optimize for fast CPU-GPU data transfer:** It is recommanded to use pinned memory on CPU for loading data into ```mgard_x::Array``` such that it can enable fast CPU-GPU data transfer. 
	+ To allocate pinned memory on CPU: ```mgard_x::cudaMallocHostHelper(void ** data_ptr, size_t size)```.
	+ To free pinned memory on CPU: ```mgard_x::cudaFreeHostHelper(void * data_ptr)```

## A simple example
The following code shows how to compress/decompress a 3D dataset with the low-level APIs. 

		#include <vector>
		#include <iostream>
		#include "mgard/compress.hpp"
		int main() 
		{
		  mgard_x::SIZE n1 = 10;
		  mgard_x::SIZE n2 = 20;
		  mgard_x::SIZE n3 = 30;
		
		  //prepare 
		  std::cout << "Preparing data...";
		  double * in_array_cpu;
		  mgard_x::cudaMallocHostHelper((void **)&in_array_cpu, sizeof(double)*n1*n2*n3);
		  //... load data into in_array_cpu
		  std::vector<mgard_x::SIZE> shape{ n1, n2, n3 };
		  mgard_x::Handle<3, double> handle(shape);
		  mgard_x::Array<3, double> in_array(shape);
		  in_array.loadData(in_array_cpu);
		  std::cout << "Done\n";
		
		  std::cout << "Compressing with MGARD-GPU...";
		  double tol = 0.01, s = 0;
		  mgard_x::Array<1, unsigned char> compressed_array = mgard_x::compress(handle, in_array, mgard_x::REL, tol, s);
		  mgard_x::SIZE compressed_size = compressed_array.getShape()[0]; //compressed size in number of bytes.          
		  unsigned char * compressed_array_cpu = compressed_array.getDataHost();
		  std::cout << "Done\n";
		
		  std::cout << "Decompressing with MGARD-GPU...";
		  // decompression
		  mgard_x::Array<3, double> decompressed_array = mgard_x::decompress(handle, compressed_array);
		  double * decompressed_array_cpu = decompressed_array.getDataHost();
		  std::cout << "Done\n";
		  
		  mgard_x::cudaFreeHostHelper(in_array_cpu);
	}

## A comprehensive example
* A comprehensive example about how to use MGARD-GPU is located in [here][example].

[example]:../examples/gpu-cuda/SimpleExample
