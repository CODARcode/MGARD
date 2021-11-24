/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_AUTOTUNER_H
#define MGARD_X_AUTOTUNER_H

namespace mgard_x {


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