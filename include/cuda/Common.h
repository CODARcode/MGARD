/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_COMMON
#define MGRAD_CUDA_COMMON

namespace mgard_cuda {
enum error_bound_type { REL, ABS };
}

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Array.h"
#include "Handle.h"
#include "MemoryManagement.h"
#include "Message.h"

#endif
