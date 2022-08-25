#ifndef MGARD_X_CONFIG_HPP
#define MGARD_X_CONFIG_HPP

#include "../RuntimeX/Utilities/Message.h"

namespace mgard_x {

struct Config {
  device_type dev_type;
  int dev_id;
  int num_dev;
  enum decomposition_type decomposition;
  SIZE huff_dict_size;
  SIZE huff_block_size;
  SIZE lz4_block_size;
  int zstd_compress_level;
  int uniform_coord_mode;
  enum lossless_type lossless;
  int reorder;
  int log_level;

  Config() {
    dev_type = device_type::AUTO;
    dev_id = 0;
    num_dev = 1;
    decomposition = decomposition_type::MultiDim;
    huff_dict_size = 8192;
    huff_block_size = 1024 * 20;
    lz4_block_size = 1 << 15;
    zstd_compress_level = 3;
    uniform_coord_mode = 1;
    lossless = lossless_type::Huffman;
    reorder = 0;
    log_level = log::ERR;
  }

  void apply() {
    log::level = log_level;
  }
};

}

#endif