
# MGARD-X

MGARD-X is a portable implementation of the MGARD lossy compressor supporting various types of GPUs and CPUs.
## Supporting features
* **Data type:** Double and single precision floating-point data
* **Dimensions:** 1D-5D
* **Error-bound type:** L\_Inf error and L\_2 error
* **Error-bound mode:** Absoluate and relative
* **Data structure:** Uniform and non-uniform spaced Cartisan gird

## Portability
|Hardware|Portability|Tested processors|
|---|---|---|
|NVIDIA GPUs|Yes*|V100, RTX2080 Ti|
|AMD GPUs|Yes|MI-100, MI-250X|
|Intel GPUs|Yes|Gen9|
|Integrated GPUs|Yes|Gen9|
|x86 CPUs|Yes|Intel CPUs, AMD CPUs|
|Power CPUs|Yes|IMB Power9 CPUs|
|ARM CPUs|To be tested||


*LZ4 lossless compressor is only avaialble to choose on NVIDIA GPUs. Portable version is under development.

## Interfaces

* **Command Line Inteface:** Full-featured command line executable for compression/decompression. 
* **High-level APIs:** Encapsulated all GPU-related details for easiest integration with user's program. Also, compressing metadata is being handled internally.
* **Low-level APIs:** Users can fully control different steps of compression (the preprocessing step, GPU buffers allocations, GPU compression kernel invokation, GPU-CPU data copy) for more fexiable and higher performance compression. Compressing metadata needs to be managed by users.

  

## Software requirements
### For Serial CPU

* GCC 7.5.0+
* CMake 3.19+

### For Multi-core CPU

* GCC 7.5.0+
* OpenMP 3.1+
* CMake 3.19+

### For NVIDIA GPUs

* CUDA 11.0+
* CMake 3.19+

### For AMD GPUs

* ROCm 4.5.0+
* CMake 3.21+

### For Intel GPUs

* DPC++/SYCL 2022+
* CMake 3.21+

## Software dependencies 
* [NVCOMP v2.2.0][nvcomp] (for NVIDIA GPUs only)
* [ZSTD v1.5.0][zstd]
* [Protobuf v3.19.4][protobuf]

[nvcomp]: https://github.com/NVIDIA/nvcomp.git
[zstd]: https://github.com/facebook/zstd.git
[protobuf]: https://github.com/protocolbuffers/protobuf.git
## Configure and build
* **Option 1:** One-step configure and build scripts are available [here][build_scripts].
* **Option 2:** Manually confiugre and build with CMake

[build_scripts]:../build_scripts
	
|CMake options|Value|Default|Description|
|---|---|---|---|
|MGARD\_ENABLE\_SERIAL|ON/OFF|ON|Enable portable serial compression/decompression on CPUs|
|MGARD\_ENABLE\_OPENMP|ON/OFF|OFF|Enable portable multi-threaded compression/decompression on CPUs|
|MGARD\_ENABLE\_CUDA|ON/OFF|OFF|Enable portable GPU compression/decompression with CUDA on NVIDIA GPUs|
|MGARD\_ENABLE\_HIP|ON/OFF|OFF|Enable portable GPU compression/decompression with HIP on AMD GPUs|
|MGARD\_ENABLE\_SYCL|ON/OFF|OFF|Enable portable GPU compression/decompression with SYCL on Intel GPUs|
|MGARD\_ENABLE\_MULTI_DEVICE|ON/OFF|OFF|Enable multi-device (GPUs) compression/decompression|


## Control Errors
MGARD can bound errors in various types of norm:

* To bound ***L<sup>&infin;</sup> norm***, the ```s``` smoothness parameter needs to be set to `infinity`.
* To bound ***L<sup>2</sup> norm***, the ```s``` smoothness parameter needs to be set to `0`.
* To bound ***L<sup>s</sup> norm***, the ```s``` smoothness parameter needs to be set to `s`.

MGARD can bound error `tol` in two ways (using L<sup>&infin;</sup> norm as an example, `u` is original data, `u'` is decompressed data):

* ***Absolute*** error mode can guarantee ***| u - u' |<sub>&infin;</sub> < tol***
* ***Relative*** error mode can guarantee ***| u - u' |<sub>&infin;</sub> < tol * | u |<sub>&infin;</sub>***


## Using command line interface (CLI)
An executable ```mgard-x``` will be built after building the MGARD-X library. To use the ```mgard-x``` CLI, here are the options:

+ ```-z```: enable compression mode
    + ```-i <path>``` path to data file to be compressed
    + ```-c <path>``` path to compressed file
    + ```-t <s|d>``` data type (s: single | d:double)
    + ```-n <D>``` total number of dimensions
        + ```<n_1>``` slowest dimention
        + ```<n_2>``` 2nd slowest dimention
        + ...
        + ```<n_D>``` fastest dimention
    + ```-u <path>``` path to coordinate file (non-uniform only)
    + ```-m <abs|rel>``` error bound mode (abs: abolute | rel: relative)
    + ```-e <error>``` error bound
    + ```-s <smoothness>``` smoothness parameter
    + ```-r <0|1>``` internal data layout (0: Higher throughput | 1: Higher compression ratio)    
    + ```-l <0|1|2>``` choose lossless compressor (0:Huffman | 1:Huffman+LZ4 (NVIDIA GPU only) | 2:Huffman@ZSTD)
+ ```-x```: enable decompression mode
    + ```-c <path>``` path to compressed file
    + ```-d <path>``` path to decompressed file
+ ```-v``` enable verbose (0:None | 1: INFO | 2: TIMING | 3: ALL)
+ ```-d <auto|serial|openmp|cuda|hip|sycl>``` choose processor (auto: Auto select | serial: CPU | openmp: multi-threaded CPU | cuda: NVIDIA GPU | hip: AMD GPU | sycl: Intel GPU )
+ ```-g <G>``` number of devices (GPUs) to use

	
## For using both the high-level APIs and low-level API
* **Include the header file.**
    + Use ```mgard/compress_x.hpp``` for ***high-level*** compression/decompression APIs
    + Use ```mgard/compress_x_lowlevel.hpp``` for ***low-level*** compression/decompression APIs
* **Configure using ```mgard_x::Config```** Both high-level APIs and low-level APIs have an optional parameter for users to configure the compression/decomrpession process via ```mgard_x::Config``` class. To configure, create a ```mgard_x::Config``` object and configure its fields:
    + ```Config.dev_type```: sepcifying the processor for compression/decompression:
        + ```mgard_x::device_type::Auto```: Auto detect the best processor (***Default***)
        + ```mgard_x::device_type::SERIAL```: Use CPUs (serial)
        + ```mgard_x::device_type::OPENMP```: Use CPUs (multi-threaded)
        + ```mgard_x::device_type::CUDA```: Use NVIDIA GPUs
        + ```mgard_x::device_type::HIP ```: Use AMD GPUs
        + ```mgard_x::device_type::SYCL ```: Use Intel GPUs
        
    + ```Config.dev_id```: sepcifying a specific GPU to use in multi-GPU systems (***Default: 0***)
    + ```Config.num_dev```: sepcifying the number of GPU to use in multi-GPU systems (***Default: 1***)
    + ```Config.reorder```: sepcifying an internal data layout (0: Higher throughput | 1: Higher compression ratio) (***Default: 0***)
    + ```Config.lossless```: control the lossless compression used: 
        + ```mgard_x::lossless_type::Huffman```: Huffman compression (***Default***)
        + ```mgard_x::lossless_type::Huffman_LZ4```: Huffman and LZ4 compression 
        + ```mgard_x::lossless_type::Huffman_Zstd```: Huffman and ZSTD compression
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
    + ```[In][Optional] coords:``` The coordinates in each dimension (from slowest to fastest).
    + ```[in][Optional] config:``` For configuring the compression process (optional).
    + ```[in] output_pre_allocated:``` Indicate whether or not the output buffer is pre-allocated. If not, MGARD will allocate the output buffer.

* **For decompression:** ```void decompress(const void *compressed_data, size_t compressed_size, void *&decompressed_data, Config config)```
    + ```[In] compressed_data:``` Compressed data.
    + ```[In] compressed_size:``` Size of comrpessed data.
    + ```[Out] decompressed_data:``` Decompressed data.
    + ```[Out][Optional] shape:``` Shape of the decompressed data.
    + ```[Out][Optional] data_type:``` Data type of the decompressed data.
    + ```[In][Optional] config:``` For configuring the decompression process (optional).
    + ```[in] output_pre_allocated:``` Indicate whether or not the output buffer is pre-allocated. If not, MGARD will allocate the output buffer.

## Using low-level APIs
* **Step 1: Create Hierarchy**
An object ```mgard_x::Hierarchy``` needs to be created and initialized. This initializes the necessary environment for efficient compression. It only needs to be created once if the input shape is not changed. For example, compressing the same variable on different timesteps only needs the Hierarchy object to be created once. Also, the same Hierarchy object can be reused in between compression and decompression APIs.
    + ```mgard_x::Hierarchy<NumDims, DataType, Device_type>(std::vector<size_t> shape, std::vector<T*> coords, mgard_x::Config config)```.
        + ```[In] NumDims```: Total number of dimensions (1 - 5)
	+ ```[In] DataType```: Input data type (float or double)
        + ```[In] Device_type ```: The type of device used (mgard\_x::SERIAL, mgard\_x::OPENMP, mgard\_x::CUDA, mgard\_x::HIP, or mgard\_x::SYCL)
        + ```[In] shape```: Stores the size in each dimension (from slowest to fastest).
        + ```[In][Optional] coords```: The coordinates in each dimension (from slowest to fastest).
      	+ ```[In] config```: For configuring compression/decomrpession.
* **Step 2: Allocate workspace** The workspace needs to be pre-allocated by creating an ```mgard_x::CompressionLowLevelWorkspace``` object. Same as ```mgard_x::Hierarchy```, it only needs to be created once if the input shape is not changed and it can be reused in between compression and decompression APIs.
    + ```mgard_x::CompressionLowLevelWorkspace<NumDims, DataType, Device_type>(mgard_x::Hierarchy<NumDims, DataType, Device_type> &hierarchy)```.
* **Step 3: Use mgard_x::Array.** ```mgard_x::Array``` is used for holding a managed array on GPU or CPU.
    +  For ***creating*** an array. ```mgard_x::Array::Array<NumDims, DataType, Device_type>(std::vector<size_t> shape)``` creates an managed array on GPU or CPU with the shape of ```shape```.
    +  For ***loading data*** into an array. ```void mgard_x::Array::load(DataType *data, size_t ld = 0)``` copies ```data``` into the the managed array on the targeting processor. ```data``` can be on either on CPU or GPU. An optional ```ld``` can be provided for specifying the size of the leading dimension. Passing ```ld = 0``` indicates that the size of the leading dimension equals to the size of the fastest dimension.
    +  For ***accessing data from CPU*** ```DataType * mgard_x::Array::hostCopy(bool keep = false)``` returns a CPU pointer of the array. An optional flag ```keep``` can be use to indicates whether or not to keep memory allocation of host copy after the ```mgard_x::Array``` object is destructed.
    +  For ***accessing data from GPU***```DataType * mgard_x::Array::data(size_t &ld)``` returns a pointer of the array on the targeting processor.
    +  For ***getting the shape*** of an array. ```mgard_x::SIZE mgard_x::Array::shape(<dimension>)``` returns the size of a specified dimension of the managed array.

   ***Note:*** ```mgard_x::Array``` will automatically release its internal CPU/GPU array when it goes out of scope.
 
* **Step 4: Invoke compression/decompression.**:
    + For ***compression***: ```void mgard_x::compress(mgard_x::Hierarchy <NumDims, DataType, DeviceType> &hierarchy, mgard_x::Array<NumDims, DataType, DeviceType> in_array, mgard_x::error_bound_type type, DataType tol, DataType s, DataType &norm, mgard_x::Config config, 
CompressionLowLevelWorkspace<NumDims, DataType, DeviceType> &workspace, mgard_x::Array<1, unsigned char, DeviceType> &compressed_data)```
        + ```[In] hierarchy ```: Hierarchy object initilzied with the shape of the input data.
     	+ ```[In] in_array ```: Input data to be compressed (its value will be altered during compression).
     	+ ```[In] type ```: Error bound type. ```mgard_x::error_bound_type::REL``` for relative error bound or ```mgard_x::error_bound_type::ABS``` for absolute error bound.
        + ```[In] tol```: Error bound.
        + ```[In] s```: Smoothness parameter.
        + ```[Out] norm```: Norm of the original data (vaild only in relative error bound mode).
        + ```[In] config```: For configuring compression.
        + ```[In] workspace```: Pre-allocated workspace.
        + ```[Out] compressed_data```: Compressed data.
  	
    + For ***decompression***: ```void mgard_x::decompress(mgard_x::hierarchy <NumDims, DataType, DeviceType> &hierarchy, mgard_x::Array<1, unsigned char, DeviceType> compressed_data, enum mgard_x::error_bound_type type, DataType tol, DataType s, DataType norm, mgard_x::Config config, CompressionLowLevelWorkspace<NumDims, DataType, DeviceType> &workspace, mgard_x::Array<NumDims, DataType, DeviceType> &decompressed_data)```
        + ```[In] hierarchy ```: Hierarchy object initilzied with the shape of the original data.
        + ```[In] compressed_data ```: Compressed data.
        + ```[In] type ```: Error bound type. ```mgard_x::error_bound_type::REL``` for relative error bound or ```mgard_x::error_bound_type::ABS``` for absolute error bound.
        + ```[In] tol```: Error bound.
        + ```[In] s```: Smoothness parameter.
        + ```[In] norm```: Norm of the original data.
        + ```[In] config```: For configuring decomrpession.
        + ```[In] workspace```: Pre-allocated workspace.
        + ```[Out] decompressed_data```: Decompressed data.
    + ***Template parameters***
        - ```NumDims```: Number of dimentions of the original data (1 - 5).
        - ```DataType```: Data type of the original data:
            + ```float```
            + ```double```
        - ```DeviceType```: Processor used for compression/decompression:
            + ```mgard_x::SERIAL```
            + ```mgard_x::OPENMP```
            + ```mgard_x::CUDA```
            + ```mgard_x::HIP```
            + ```mgard_x::SYCL```
	
## Performance optimization
For achieving the best performance:

* **Specifiying the suitable GPU architecture(s)**: Use CMake configuration options to specifiying the suitable GPU architecture(s)
    + For NVIDIA GPUs, use ```-DCMAKE_CUDA_ARCHITECTURES=<arch>```
    + For AMD GPUs, use ```-DCMAKE_HIP_ARCHITECTURES=<arch>```
    + For Intel GPU, please specisify ```-fsycl-targets``` and ```-Xsycl-target-backend``` C++ compiler flags accrodingly
* **Auto Tuning**: each kernel in MGARD-X can be auto tuned for the current hardware architecture. After MGARD-X is built, an executable ```mgard-x-autotuner``` will be generated for auto tuning. ```mgard-x-autotuner``` can be used in following ways:
    + **Full automatic mode:** run ```mgard-x-autotuner```  without arguments with make MGARD-X auto tune all its kernels for all backends that are enabled.
    + **Tune for a specific backend:** run ```mgard-x-autotuner -d <serial|openmp|cuda|hip|sycl>```
    + **Tune for a specific shape of data on a specific backend :** run ```mgard-x-autotuner -d <auto|serial|openmp|cuda|hip|sycl> -n <ndim> [dim1] [dim2] ... [dimN]```.
    + ***Note:*** MGARD-X needs to be recompiled after auto tuning to make it effective.

## Example Code

* High-level APIs example code can be found in [here][high-level-example].
* Low-level APIs example code can be found in [here][low-level-example].

[high-level-example]:../examples/mgard-x/HighLevelAPIs
[low-level-example]:../examples/mgard-x/LowLevelAPIs
