#ifndef MGARD_X_CONFIG_HPP
#define MGARD_X_CONFIG_HPP

#include "../RuntimeX/RuntimeXPublic.h"
#include "../RuntimeX/Utilities/Message.h"
#include "../Utilities/Types.h"

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
  bool normalize_coordinates;
  enum lossless_type lossless;
  int reorder;
  int log_level;

  Config();
  void apply();
};

} // namespace mgard_x

#endif