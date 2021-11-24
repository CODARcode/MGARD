/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_COMMON
#define MGARD_X_COMMON

#include <stdint.h>
#include <algorithm>
#include <cstdio>

namespace mgard_x {

enum class processor_type:uint8_t { CPU, GPU_CUDA };

enum class error_bound_type:uint8_t { REL, ABS };
enum class norm_type:uint8_t { L_Inf, L_2 };
enum class lossless_type:uint8_t { CPU_Lossless, GPU_Huffman, GPU_Huffman_LZ4 };

enum class data_type:uint8_t { Float, Double };
enum class data_structure_type:uint8_t { Cartesian_Grid_Uniform, Cartesian_Grid_Non_Uniform};

enum class endiness_type:uint8_t { Little_Endian, Big_Endian };

enum class coordinate_location:uint8_t { Embedded, External };

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

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Array.h"
#include "Handle.h"
#include "Message.h"
#include "ErrorCalculator.h"
#include "MemoryManagement.h"

#endif
