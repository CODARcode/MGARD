/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */
#include "cuda/CommonInternal.h"
 
#include "cuda/MDR/BitplaneEncoder/PerBitBPEncoderGPU.hpp"

namespace mgard_x {
namespace MDR {

// #define KERNELS(D, T) template class PerBitEncoder<T, CUDA>;

// KERNELS(1, double)
// KERNELS(1, float)
// KERNELS(2, double)
// KERNELS(2, float)
// KERNELS(3, double)
// KERNELS(3, float)

// #undef KERNELS

}
} // namespace mgard_x