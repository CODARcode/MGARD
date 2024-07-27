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
  SIZE lz4_block_size;
  int zstd_compress_level;
  bool normalize_coordinates;
  enum lossless_type lossless;
  int reorder;
  int log_level;
  bool auto_pin_host_buffers;
  SIZE max_larget_level;
  SIZE max_memory_footprint;
  SIZE total_num_bitplanes;
  SIZE block_size;
  SIZE temporal_dim;
  SIZE temporal_dim_size;
  SIZE domain_decomposition_dim;
  std::vector<SIZE> domain_decomposition_sizes;
  bool mdr_adaptive_resolution;
  bool adjust_shape;
  bool compress_with_dryrun;
  int num_local_refactoring_level;
  bool auto_cache_release;
  cpu_parallelization_mode cpu_mode;

  Config();
  void apply();
};

} // namespace mgard_x

#endif