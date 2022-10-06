/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */
#include "mgard-x/RuntimeX/RuntimeX.h"
namespace mgard_x {

template void BeginAutoTuning<SERIAL>();
template void EndAutoTuning<SERIAL>();

} // namespace mgard_x