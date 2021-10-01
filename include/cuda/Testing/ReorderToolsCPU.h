#ifndef REORDERTOOLSCPU_H
#define REORDERTOOLSCPU_H

/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#include "TensorMeshHierarchy.hpp"
#include "shuffle.hpp"

namespace mgard {

template <std::size_t D, typename T>
void ReorderCPU(TensorMeshHierarchy<D, T> &hierarchy, T * input, T * output);
template <std::size_t D, typename T>
void ReverseReorderCPU(TensorMeshHierarchy<D, T> &hierarchy, T * input, T * output);

}

#endif