/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_RUNTIME_X_DATA_TYPES_H
#define MGARD_X_RUNTIME_X_DATA_TYPES_H

// Types defined here are for RuntimeX

#if defined __CUDACC__
#define MGARDX_COMPILE_CUDA
#elif defined __HIPCC__
#define MGARDX_COMPILE_HIP
#elif defined SYCL_LANGUAGE_VERSION
#define MGARDX_COMPILE_SYCL
#else
#define MGARDX_COMPILE_SERIAL
#if MGARD_ENABLE_OPENMP
#define MGARDX_COMPILE_OPENMP
#endif
#endif

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

#if defined(MGARDX_COMPILE_SERIAL) || defined(MGARDX_COMPILE_OPENMP)
#define MGARDX_CONT __inline__
#define MGARDX_KERL
#define MGARDX_EXEC __inline__
#define MGARDX_CONT_EXEC __inline__
#define MGARDX_MANAGED
#endif

#ifdef MGARDX_COMPILE_CUDA
#define MGARDX_CONT __host__ __inline__
#define MGARDX_KERL __global__
#define MGARDX_EXEC __device__ __forceinline__
#define MGARDX_CONT_EXEC __host__ __device__ __forceinline__
#define MGARDX_MANAGED __managed__
#endif

#ifdef MGARDX_COMPILE_HIP
#define MGARDX_CONT __host__ __inline__
#define MGARDX_KERL __global__
#define MGARDX_EXEC __device__ __inline__
#define MGARDX_CONT_EXEC __host__ __device__ __inline__
#define MGARDX_MANAGED __managed__
#endif

#ifdef MGARDX_COMPILE_SYCL
#define MGARDX_CONT __inline__
#define MGARDX_KERL
#define MGARDX_EXEC __inline__
#define MGARDX_CONT_EXEC __inline__
#define MGARDX_MANAGED
#endif

#if defined MGARDX_COMPILE_KOKKOS
#define MGARDX_CONT __inline__
#define MGARDX_KERL
#define MGARDX_EXEC __inline__
#define MGARDX_CONT_EXEC KOKKOS_INLINE_FUNCTION
#define MGARDX_MANAGED
#endif

#define MGARDX_WARP_SIZE 32
#define MGARDX_MAX_NUM_WARPS_PER_TB 32
#define MGARDX_NUM_SMs 68

// 16 Async queues + 1 Sync queue
#define MGARDX_NUM_QUEUES 17
#define MGARDX_NUM_ASYNC_QUEUES 16
#define MGARDX_SYNCHRONIZED_QUEUE 16
#define SIZE_MAX_VALUE 4294967295

#define COPY 0
#define ADD 1
#define SUBTRACT 2

class Device {};
class SERIAL : public Device {};
class OPENMP : public Device {};
class CUDA : public Device {};
class HIP : public Device {};
class SYCL : public Device {};
class NONE : public Device {};

enum class device_type : uint8_t {
  AUTO,
  SERIAL,
  OPENMP,
  CUDA,
  HIP,
  SYCL,
  NONE
};

#if defined MGARDX_COMPILE_KOKKOS
using KOKKOS = Kokkos::DefaultExecutionSpace;
#else
using KOKKOS = NONE;
#endif

using IDX = uint64_t;
using ATOMIC_IDX = unsigned long long int;
using SIZE = uint64_t;
using DIM = uint8_t;
using QUANTIZED_INT = int64_t;
using QUANTIZED_UNSIGNED_INT = uint64_t;
using HUFFMAN_CODE = uint64_t;
using SERIALIZED_TYPE = unsigned char;
using Byte = unsigned char;
using OPTION = int8_t;
using THREAD_IDX = unsigned int;
} // namespace mgard_x

#endif