/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: Jul 10, 2021
 */

#include <algorithm>

#include "cuda/CommonInternal.h"

#include "cuda/SubArray.hpp"

namespace mgard_cuda {



template class SubArray<1, double>;
template class SubArray<1, float>;
template class SubArray<2, double>;
template class SubArray<2, float>;
template class SubArray<3, double>;
template class SubArray<3, float>;
template class SubArray<4, double>;
template class SubArray<4, float>;
template class SubArray<5, double>;
template class SubArray<5, float>;

template class SubArray<1, bool>;

template class SubArray<1, uint8_t>;
template class SubArray<1, uint16_t>;
template class SubArray<1, uint32_t>;
template class SubArray<1, uint64_t>;

template class SubArray<2, uint8_t>;
template class SubArray<2, uint16_t>;
template class SubArray<2, uint32_t>;
template class SubArray<2, uint64_t>;

template class SubArray<1, unsigned long long>;


// template class SubArray<1, QUANTIZED_INT>;
// template class SubArray<2, QUANTIZED_INT>;
// template class SubArray<3, QUANTIZED_INT>;
// template class SubArray<4, QUANTIZED_INT>;
// template class SubArray<5, QUANTIZED_INT>;

} // namespace mgard_cuda