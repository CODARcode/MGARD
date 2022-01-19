/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_RUNTIME_X_DATA_TYPES_H
#define MGARD_X_RUNTIME_X_DATA_TYPES_H

#include <stdint.h>

#if defined MGARDX_COMPILE_KOKKOS
#include "Kokkos_Core.hpp"
#endif  

namespace mgard_x {


#if defined(__CUDACC__) // NVCC
   #define MGARDX_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define MGARDX_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define MGARDX_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif


#ifdef MGARDX_COMPILE_SERIAL 
#define MGARDX_CONT __inline__
#define MGARDX_KERL 
#define MGARDX_EXEC __inline__
#define MGARDX_CONT_EXEC __inline__
#define MGARDX_MANAGED
#endif  


#ifdef MGARDX_COMPILE_CUDA
#define MGARDX_CONT __host__  __inline__
#define MGARDX_KERL __global__
#define MGARDX_EXEC __device__ __forceinline__
#define MGARDX_CONT_EXEC __host__ __device__ __forceinline__
#define MGARDX_MANAGED __managed__
#endif


#ifdef MGARDX_COMPILE_HIP
#define MGARDX_CONT __host__  __inline__
#define MGARDX_KERL __global__
#define MGARDX_EXEC __device__ __inline__
#define MGARDX_CONT_EXEC __host__ __device__ __inline__
#define MGARDX_MANAGED __managed__
#endif

#if defined MGARDX_COMPILE_KOKKOS
#define MGARDX_CONT __inline__
#define MGARDX_KERL 
#define MGARDX_EXEC __inline__
#define MGARDX_CONT_EXEC KOKKOS_INLINE_FUNCTION
#define MGARDX_MANAGED 
#endif  


#define MAX_GRID_X 2147483647
#define MAX_GRID_Y 65536
#define MAX_GRID_Z 65536
#define MGARDX_WARP_SIZE 32
#define MGARDX_MAX_NUM_WARPS_PER_TB 32
#define MGARDX_NUM_SMs 68

#define MGARDX_NUM_QUEUES 16
#define SIZE_MAX_VALUE 4294967295

#define COPY 0
#define ADD 1
#define SUBTRACT 2
    

class Device {};
class Serial: public Device {};
class CUDA: public Device {};
class HIP: public Device {};
class None: public Device {};

#if defined MGARDX_COMPILE_KOKKOS
using KOKKOS = Kokkos::DefaultExecutionSpace;
#else
using KOKKOS = None;
#endif

class DPCxx: public Device {};
class OpenMp: public Device {};




using IDX = unsigned long long int;
using LENGTH = unsigned long long int;
using SIZE = uint32_t;//unsigned int;
// using SIZE = int;
using DIM = uint32_t;
using QUANTIZED_INT = int32_t;
using QUANTIZED_UNSIGNED_INT = uint32_t;
using SERIALIZED_TYPE = unsigned char;
using Byte = unsigned char;
using OPTION = int8_t;
using THREAD_IDX = unsigned int;
}

#endif