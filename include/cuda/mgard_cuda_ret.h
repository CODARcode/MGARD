/* 
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_RET
#define MGRAD_CUDA_RET

struct mgard_cuda_ret {
  int info;
  double time;
  mgard_cuda_ret() : info(0), time(0.0) {}
  mgard_cuda_ret(int info, double time) {
    this->info = info;
    this->time = time;
  }
};

#endif
