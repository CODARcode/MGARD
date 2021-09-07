/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_COMMON
#define MGRAD_CUDA_COMMON

#include <stdint.h>

namespace mgard_cuda {

enum error_bound_type { REL, ABS };
// enum data_type { Float, Double };
enum class data_type : uint8_t { Float, Double };

using IDX = unsigned long long int;
using LENGTH = unsigned long long int;
using SIZE = unsigned int;
// using SIZE = int;
using DIM = uint32_t;
using QUANTIZED_INT = int;
using SERIALIZED_TYPE = unsigned char;
using Byte = unsigned char;
using OPTION = uint32_t;
} // namespace mgard_cuda

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Array.h"
#include "ErrorCalculator.h"
#include "Handle.h"
#include "MemoryManagement.h"
#include "Message.h"

#endif
