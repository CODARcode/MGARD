/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_UTILITIES_TYPES
#define MGARD_X_UTILITIES_TYPES

// Types defined here are for compression

#include <algorithm>
#include <cstdio>
#include <stdint.h>

namespace mgard_x {

enum class decomposition_type : uint8_t { MultiDim, SingleDim, Hybrid };

enum class processor_type : uint8_t {
  CPU,
  GPU_CUDA,
  X_SERIAL,
  X_OPENMP,
  X_CUDA,
  X_HIP,
  X_SYCL
};

enum class error_bound_type : uint8_t { REL, ABS };
enum class norm_type : uint8_t { L_Inf, L_2 };
enum class lossless_type : uint8_t {
  Huffman,
  Huffman_LZ4,
  Huffman_Zstd,
  CPU_Lossless
};

enum class data_type : uint8_t { Float, Double };
enum class data_structure_type : uint8_t {
  Cartesian_Grid_Uniform,
  Cartesian_Grid_Non_Uniform
};

enum class endiness_type : uint8_t { Little_Endian, Big_Endian };

enum class coordinate_location : uint8_t { Embedded, External };

enum class domain_decomposition_type : uint8_t { MaxDim, TemporalDim, Block };

enum class operation_type : uint8_t { Compression, MDR };

enum class bitplane_encoding_type : uint8_t { GroupedBitplaneEncoding };

enum class compress_status_type : uint8_t {
  Success,
  Failure,
  OutputTooLargeFailure,
  NotSupportHigherNumberOfDimensionsFailure,
  NotSupportDataTypeFailure,
  BackendNotAvailableFailure
};

enum class compressor_type : uint8_t { MGARD, ZFP };

} // namespace mgard_x

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#endif
