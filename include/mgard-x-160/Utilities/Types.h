/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_UTILITIES_TYPES
#define MGARD_X_160_UTILITIES_TYPES

// Types defined here are for compression

#include <algorithm>
#include <cstdio>
#include <stdint.h>

namespace mgard_x_160 {

enum class decomposition_type : uint8_t { MultiDim, SingleDim, Hybrid };

// Edge length of the block-local transform block used by the hybrid
// (BlockMGARD) hierarchy. The 8 -> 5 coarsening in the block-local refactor
// and the ROI block grid are both built on this. Compile-time today; recorded
// in the file header so a future change stays readable.
constexpr uint64_t MGARDX_HYBRID_LOCAL_BLOCK_SIZE = 8;

// Coarse edge length one block-local level produces: 8 fine nodes coarsen to
// the 5 nodes {0, 2, 4, 6, 7} per dimension.
constexpr uint64_t MGARDX_HYBRID_LOCAL_COARSE_SIZE = 5;

// Number of coefficients one block-local block contributes per level:
// 8^D - 5^D (387 in 3D, 39 in 2D, 3 in 1D). The coefficient array is a
// row-major sequence of blocks, so `index / this` is the block index -- which
// is how the ROI per-block quantizers are addressed.
constexpr uint64_t hybrid_local_coeff_per_block(uint64_t num_dims) {
  uint64_t fine = 1, coarse = 1;
  for (uint64_t d = 0; d < num_dims; d++) {
    fine *= MGARDX_HYBRID_LOCAL_BLOCK_SIZE;
    coarse *= MGARDX_HYBRID_LOCAL_COARSE_SIZE;
  }
  return fine - coarse;
}

// Transform basis used by the (de)compose step, shared by the plain
// Compressor and the hybrid (BlockMGARD) HybridHierarchyCompressor via
// Config::projection_mode -- one setting controls both.
//
// - Auto (default): resolved from the error-bound norm at (de)compress time
//   -- Hierarchical under an L-infinity bound, Orthogonal otherwise. See
//   resolve_projection_mode in Utilities/ProjectionMode.h.
// - Orthogonal / Hierarchical: forced explicitly. Hierarchical still only
//   supports L-infinity error control; requesting it under any other bound
//   throws. Unlike kernel fusion, the resolved basis changes the coefficient
//   stream, so it is recorded in the compressed-file metadata for the hybrid
//   path.
enum class compression_projection_mode_type : uint8_t {
  Auto,
  Orthogonal,
  Hierarchical
};

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
  Huffman_LZ4, // Huffman, then the portable LZ4 backend (Lossless/LZ4/)
  Huffman_Zstd,
  CPU_Lossless,
  BlockDelta,
  // Portable LZ4 applied directly to the (raw signed) quantized integer stream,
  // with no Huffman entropy stage. Self-contained like BlockDelta.
  LZ4,
  // Zero-run-length encoding (RLE0) of the quantized stream, then a
  // byte-alphabet rANS entropy stage on the resulting (counts, symbols) blob.
  // Breaks Huffman's ~1-bit/symbol ratio floor. Self-contained like LZ4.
  ZeroRLE_Rans,
  // Symbol-alphabet rANS: same model as Huffman (outlier separation + dict_size
  // histogram) but rANS entropy coding instead of Huffman codes -> always <=
  // Huffman size (fractional bits), uncapped. Self-contained.
  SymbolRans
};

// Encoding variant for the BlockDelta lossless backend (mirrors cuSZp):
//   Fixed   - fixed-length encoding only (no delta)
//   Delta   - per-block delta + zigzag + fixed-length encoding
//   Outlier - Delta plus per-block outlier peeling (large values stored aside)
enum class block_delta_mode_type : uint8_t { Fixed, Delta, Outlier };

enum class data_type : uint8_t { Float, Double };
enum class data_structure_type : uint8_t {
  Cartesian_Grid_Uniform,
  Cartesian_Grid_Non_Uniform
};

enum class endiness_type : uint8_t { Little_Endian, Big_Endian };

enum class coordinate_location : uint8_t { Embedded, External };

enum class domain_decomposition_type : uint8_t { MaxDim, Block, Variable };

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

} // namespace mgard_x_160

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#endif
