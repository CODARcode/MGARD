#ifndef MGARD_X_CONFIG_HPP
#define MGARD_X_CONFIG_HPP

#include "../RuntimeX/RuntimeXPublic.h"
#include "../RuntimeX/Utilities/Log.h"
#include "../Utilities/Types.h"

namespace mgard_x {

struct Config {
  device_type dev_type;
  int dev_id;
  enum compressor_type compressor;
  enum domain_decomposition_type domain_decomposition;
  enum decomposition_type decomposition;
  double estimate_outlier_ratio;
  SIZE huff_dict_size;
  SIZE huff_block_size;
  SIZE block_delta_block_size;
  enum block_delta_mode_type block_delta_mode;
  SIZE lz4_block_size;
  int zstd_compress_level;
  bool normalize_coordinates;
  enum lossless_type lossless;
  int log_level;
  bool auto_pin_host_buffers;
  SIZE max_memory_footprint;
  SIZE total_num_bitplanes;
  SIZE block_size;
  SIZE domain_decomposition_dim;
  std::vector<SIZE> domain_decomposition_sizes;
  bool mdr_adaptive_resolution;
  bool adjust_shape;
  bool compress_with_dryrun;
  // Block-local levels of the hybrid hierarchy (L). Default 1.
  SIZE num_local_refactoring_level;
  // Global refactoring levels (M): caps the levels of every multilevel
  // hierarchy -- the plain multi-dimensional one and the global stage of the
  // hybrid one. Recorded in the file header. Default 3.
  SIZE num_global_refactoring_level;
  bool auto_cache_release;
  cpu_parallelization_mode cpu_mode;
  bool mdr_qoi_mode;
  int mdr_qoi_num_variables;
  std::vector<double> roi_tolerance_map;
  bool enable_roi;
  // Transform basis policy, shared by the plain Compressor and the hybrid
  // (BlockMGARD) HybridHierarchyCompressor. Auto (default) picks Hierarchical
  // under an L-infinity bound and Orthogonal otherwise; Orthogonal/
  // Hierarchical force a specific basis (Hierarchical still requires an
  // L-infinity bound). See compression_projection_mode_type in Types.h.
  enum compression_projection_mode_type projection_mode;
  // The hybrid (BlockMGARD) local stage fuses its decompose/recompose kernels
  // with quantization/dequantization so coefficients never round-trip through
  // global memory as T. These select the older separate-pass implementation,
  // which the test suite still exercises. Purely a performance choice: both
  // paths reconstruct identically, so a file compressed either way decompresses
  // either way and nothing about the choice is recorded in the file header.
  bool fuse_decompose_quantize;   // compression
  bool fuse_dequantize_recompose; // decompression

  Config();
  void apply();
};

} // namespace mgard_x

#endif
