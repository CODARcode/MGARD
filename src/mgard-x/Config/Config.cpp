/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "mgard-x/Config/Config.h"

namespace mgard_x {

Config::Config() {
  dev_type = device_type::AUTO;
  dev_id = 0;
  num_dev = 1;
  decomposition = decomposition_type::MultiDim;
  huff_dict_size = 8192;
  huff_block_size = 1024 * 20;
  lz4_block_size = 1 << 15;
  zstd_compress_level = 3;
#if MGARD_ENABLE_COORDINATE_NORMALIZATION
  normalize_coordinates = true;
#else
  normalize_coordinates = false;
#endif
  lossless = lossless_type::Huffman;
  reorder = 0;
  log_level = log::ERR;
  max_larget_level = 0;
  prefetch = false;
}

void Config::apply() { log::level = log_level; }

} // namespace mgard_x
