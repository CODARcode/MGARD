/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#include "cuda/CommonInternal.h"

#include "cuda/ParallelHuffman/Histogram.hpp"

namespace mgard_cuda {


template class Histogram<uint8_t, unsigned int, CUDA>;
template class Histogram<uint16_t, unsigned int, CUDA>;
template class Histogram<uint32_t, unsigned int, CUDA>;


} // namespace mgard_cuda