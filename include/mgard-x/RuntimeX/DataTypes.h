/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_RUNTIME_X_DATA_TYPES_H
#define MGARD_X_RUNTIME_X_DATA_TYPES_H

#include <stdint.h>

namespace mgard_x {

class Device {};
class CUDA: public Device {};
class HIP: public Device {};
class DPCxx: public Device {};
class OpenMp: public Device {};
class Kokkos: public Device {};

using IDX = unsigned long long int;
using LENGTH = unsigned long long int;
using SIZE = uint32_t;//unsigned int;
// using SIZE = int;
using DIM = uint32_t;
using QUANTIZED_INT = int32_t;
using QUANTIZED_UNSIGNED_INT = uint32_t;
using SERIALIZED_TYPE = unsigned char;
using Byte = unsigned char;
using OPTION = int8_t;
}

#endif