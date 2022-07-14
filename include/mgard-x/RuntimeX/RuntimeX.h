/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "RuntimeXPublic.h"

#include "DataTypes.h"

#include "AutoTuners/AutoTuner.h"
#include "Tasks/Task.h"

// Serial backend should be always available
#include "DeviceAdapters/DeviceAdapterSerial.h"

#if MGARD_ENABLE_CUDA
#ifdef MGARDX_COMPILE_CUDA
#include "DeviceAdapters/DeviceAdapterCuda.h"
#endif
#endif

#if MGARD_ENABLE_HIP
#ifdef MGARDX_COMPILE_HIP
#include "DeviceAdapters/DeviceAdapterHip.h"
#endif
#endif

#if MGARD_ENABLE_SYCL
#ifdef MGARDX_COMPILE_SYCL
#include "DeviceAdapters/DeviceAdapterSycl.h"
#endif
#endif

#if RUNTIME_X_ENABLE_KOKKOS
#include "DeviceAdapters/DeviceAdapterKokkos.h"
#endif

#include "Utilities/CheckShape.hpp"
#include "Utilities/OffsetCalculators.hpp"
#include "Utilities/Serializer.hpp"

#include "DataStructures/Array.hpp"
#include "DataStructures/SubArray.hpp"
#include "DataStructures/SubArrayCopy.hpp"
#include "Utilities/SubArrayPrinter.hpp"
