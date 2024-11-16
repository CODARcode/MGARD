/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include <limits>

#include "mgard-x/Config/Config.h"

namespace mgard_x {

Config::Config() {
  dev_type = device_type::AUTO;
  dev_id = 0;
  compressor = compressor_type::MGARD;
  domain_decomposition = domain_decomposition_type::MaxDim;
  decomposition = decomposition_type::MultiDim;
  estimate_outlier_ratio = 1.0;
  huff_dict_size = 8192;
  huff_block_size = 1024 * 20;
  lz4_block_size = 1 << 15;
  zstd_compress_level = 3;
  normalize_coordinates = false;
  lossless = lossless_type::Huffman;
  reorder = 0;
  prefetch = false;
  log_level = log::ERR;
  max_larget_level = std::numeric_limits<SIZE>::max(); // no limit
  auto_pin_host_buffers = true;
  max_memory_footprint = std::numeric_limits<SIZE>::max(); // no limit
  total_num_bitplanes = 32;
  block_size = 256;
  domain_decomposition_dim = 0;
  domain_decomposition_sizes = std::vector<SIZE>();
  mdr_adaptive_resolution = false;
  adjust_shape = false;
  compress_with_dryrun = false;
  num_local_refactoring_level = 1;
  auto_cache_release = false;
  cpu_mode = cpu_parallelization_mode::INTER_BLOCK;
}

void Config::apply() { log::level = log_level; }

} // namespace mgard_x
