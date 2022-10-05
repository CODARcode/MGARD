/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */
#include "MGARDXConfig.h"

#include "DataTypes.h"

#include "Utilities/Log.h"
#include "Utilities/Timer.hpp"

#ifndef MGARD_X_RUNTIME_X_PUBLIC_H
#define MGARD_X_RUNTIME_X_PUBLIC_H

namespace mgard_x {

template <typename DeviceType> bool deviceAvailable();
}

#endif
