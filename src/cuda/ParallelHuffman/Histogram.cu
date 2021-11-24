/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include "cuda/CommonInternal.h"

#include "cuda/ParallelHuffman/Histogram.hpp"

namespace mgard_x {


template class Histogram<uint8_t, unsigned int, CUDA>;
template class Histogram<uint16_t, unsigned int, CUDA>;
template class Histogram<uint32_t, unsigned int, CUDA>;


} // namespace mgard_x