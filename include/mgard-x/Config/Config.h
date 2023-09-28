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
  SIZE max_larget_level;
  bool prefetch;
  SIZE max_memory_footprint;
  SIZE total_num_bitplanes;
  SIZE block_size;
  SIZE temporal_dim;
  SIZE temporal_dim_size;
  bool mdr_adaptive_resolution;
  bool collect_uncertainty;
  bool adjust_shape;
  bool compress_with_dryrun;
  int num_local_refactoring_level;
  bool cache_compressor;

  Config();
  void apply();
};

} // namespace mgard_x

#endif