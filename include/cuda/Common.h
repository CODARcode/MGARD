/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: September 27, 2021
 */

#ifndef MGRAD_CUDA_COMMON
#define MGRAD_CUDA_COMMON

#include <stdint.h>

namespace mgard_cuda {

enum class processor_type : uint8_t { CPU, GPU_CUDA };

enum class error_bound_type : uint8_t { REL, ABS };
enum class norm_type : uint8_t { L_Inf, L_2 };
enum class lossless_type : uint8_t {
  CPU_Lossless,
  GPU_Huffman,
  GPU_Huffman_LZ4
};

enum class data_type : uint8_t { Float, Double };
enum class data_structure_type : uint8_t {
  Cartesian_Grid_Uniform,
  Cartesian_Grid_Non_Uniform
};

enum class endiness_type : uint8_t { Little_Endian, Big_Endian };

enum class coordinate_location : uint8_t { Embedded, External };

class Device {};
class CUDA : public Device {};
class HIP : public Device {};
class DPCxx : public Device {};
class OpenMp : public Device {};
class Kokkos : public Device {};

using IDX = unsigned long long int;
using LENGTH = unsigned long long int;
using SIZE = unsigned int;
// using SIZE = int;
using DIM = uint32_t;
using QUANTIZED_INT = int;
using SERIALIZED_TYPE = unsigned char;
using Byte = unsigned char;
using OPTION = int8_t;
} // namespace mgard_cuda

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Array.h"
#include "ErrorCalculator.h"
#include "Handle.h"
#include "MemoryManagement.h"
#include "Message.h"

#endif
