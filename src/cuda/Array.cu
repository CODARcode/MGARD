/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include <algorithm>

#include "cuda/CommonInternal.h"

#include "cuda/Array.hpp"

namespace mgard_x {

template class Array<1, double, CUDA>;
template class Array<1, float, CUDA>;
template class Array<2, double, CUDA>;
template class Array<2, float, CUDA>;
template class Array<3, double, CUDA>;
template class Array<3, float, CUDA>;
template class Array<4, double, CUDA>;
template class Array<4, float, CUDA>;
template class Array<5, double, CUDA>;
template class Array<5, float, CUDA>;

// template class Array<1, unsigned char, CUDA>;


template class Array<1, bool, CUDA>;

template class Array<1, uint8_t, CUDA>;
template class Array<1, uint16_t, CUDA>;
template class Array<1, uint32_t, CUDA>;
template class Array<1, uint64_t, CUDA>;

template class Array<2, uint8_t, CUDA>;
template class Array<2, uint16_t, CUDA>;
template class Array<2, uint32_t, CUDA>;
template class Array<2, uint64_t, CUDA>;

template class Array<1, unsigned long long, CUDA>;

// template class Array<1, QUANTIZED_INT, CUDA>;
// template class Array<2, QUANTIZED_INT, CUDA>;
// template class Array<3, QUANTIZED_INT, CUDA>;
// template class Array<4, QUANTIZED_INT, CUDA>;
// template class Array<5, QUANTIZED_INT, CUDA>;

} // namespace mgard_x