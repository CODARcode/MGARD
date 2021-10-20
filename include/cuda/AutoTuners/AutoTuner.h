/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGARD_CUDA_AUTOTUNER_H
#define MGARD_CUDA_AUTOTUNER_H

namespace mgard_cuda {


template <typename DeviceType>
class KernelConfigs {
public:
  MGARDm_CONT
  KernelConfigs(){};
};

template <typename DeviceType>
class AutoTuningTable {
public:
  MGARDm_CONT
  AutoTuningTable(){};
};


template <typename DeviceType>
class AutoTuner {
public:
  MGARDm_CONT
  AutoTuner(){};

  static KernelConfigs<DeviceType> kernelConfigs;
  static AutoTuningTable<DeviceType> autoTuningTable;
  static bool ProfileKenrles;
};
}

#include "AutoTunerCuda.h"

#endif