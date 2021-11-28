/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */




#include "RuntimeXPublic.h"


#ifndef MGARD_X_RUNTIME_X_H
#define MGARD_X_RUNTIME_X_H

namespace mgard_x {

#if defined(__CUDACC__) // NVCC
   #define MGARDm_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define MGARDm_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define MGARDm_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif


#define MGARDm_CONT __host__  __inline__
#define MGARDm_KERL __global__
#define MGARDm_EXEC __device__ __forceinline__
#define MGARDm_CONT_EXEC __host__ __device__ __forceinline__
#define MGARDm_COMPILE_EXEC __CUDACC__

#define MAX_GRID_X 2147483647
#define MAX_GRID_Y 65536
#define MAX_GRID_Z 65536
#define MGARDm_WARP_SIZE 32
#define MGARDm_MAX_NUM_WARPS_PER_TB 32
#define MGARDm_NUM_SMs 68

#define MGARDm_NUM_QUEUES 16

#define SIZE_MAX_VALUE 4294967295

#define COPY 0
#define ADD 1
#define SUBTRACT 2

// reduction operations
// #define SUM 0
// #define MAX 1

}

#endif





#include "AutoTuners/AutoTuner.h"
#include "Tasks/Task.h"
#include "DeviceAdapters/DeviceAdapter.h"

// #include "Utilities/CheckEndianess.hpp"
#include "Utilities/CheckShape.hpp"
#include "Utilities/OffsetCalculators.hpp"

#include "DataStructures/Array.hpp"
#include "DataStructures/SubArray.hpp"
#include "Utilities/SubArrayPrinter.hpp"

