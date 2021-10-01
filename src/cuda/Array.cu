/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#include <algorithm>

#include "cuda/CommonInternal.h"

#include "cuda/Array.hpp"

namespace mgard_cuda {

template class Array<1, double>;
template class Array<1, float>;
template class Array<2, double>;
template class Array<2, float>;
template class Array<3, double>;
template class Array<3, float>;
template class Array<4, double>;
template class Array<4, float>;
template class Array<5, double>;
template class Array<5, float>;

// template class Array<1, unsigned char>;


template class Array<1, bool>;

template class Array<1, uint8_t>;
template class Array<1, uint16_t>;
template class Array<1, uint32_t>;
template class Array<1, uint64_t>;

template class Array<2, uint8_t>;
template class Array<2, uint16_t>;
template class Array<2, uint32_t>;
template class Array<2, uint64_t>;

template class Array<1, unsigned long long>;

// template class Array<1, QUANTIZED_INT>;
// template class Array<2, QUANTIZED_INT>;
// template class Array<3, QUANTIZED_INT>;
// template class Array<4, QUANTIZED_INT>;
// template class Array<5, QUANTIZED_INT>;

} // namespace mgard_cuda