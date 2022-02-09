/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_HANDLE
#define MGARD_X_HANDLE

#include "RuntimeX/RuntimeXPublic.h"
#include "Types.h"

namespace mgard_x {

struct Config {
  device_type dev_type;
  int dev_id;
  SIZE l_target;
  SIZE huff_dict_size;
  SIZE huff_block_size;
  SIZE lz4_block_size;
  int zstd_compress_level;
  bool reduce_memory_footprint;
  bool profile_kernels;
  bool sync_and_check_all_kernels;
  bool timing;
  int uniform_coord_mode;
  enum lossless_type lossless;
  double global_norm;
  bool reorder;

  Config() {
    dev_type = device_type::Auto;
    dev_id = 0;
    l_target = -1; // no limit
    huff_dict_size = 8192;
    //#ifdef MGARD_X_OPTIMIZE_TURING
    //    huff_block_size = 1024 * 30;
    //#endif
    //#ifdef MGARD_X_OPTIMIZE_VOLTA
    huff_block_size = 1024 * 20;
    //#endif
    lz4_block_size = 1 << 15;
    zstd_compress_level = 3;
    // reduce_memory_footprint = false;
    // profile_kernels = false;
    // sync_and_check_all_kernels = false;
    timing = false;
    uniform_coord_mode = 0;
    lossless = lossless_type::Huffman_LZ4;
    global_norm = 1;
    reorder = false;
  }
};

template <DIM D, typename T, typename DeviceType> struct Hierarchy {

  /* for general users */
  Hierarchy(std::vector<SIZE> shape, int uniform_coord_mode = 0);
  Hierarchy(std::vector<SIZE> shape, std::vector<T *> coords);

  /* for Internal use only */
  Hierarchy();
  Hierarchy(std::vector<SIZE> shape, DIM domain_decomposed_dim,
            SIZE domain_decomposed_size, int uniform_coord_mode = 0);
  Hierarchy(std::vector<SIZE> shape, DIM domain_decomposed_dim,
            SIZE domain_decomposed_size, std::vector<T *> coords);
  Hierarchy(const Hierarchy &hierarchy);
  // Hierarchy(std::vector<SIZE> shape, Config config, int uniform_coord_mode =
  // 0); Hierarchy(std::vector<SIZE> shape, std::vector<T *> coords, Config
  // config);

  ~Hierarchy();

  /* Refactoring env */
  // enum device_type dev_type;
  // int dev_id;
  SIZE l_target;
  DIM D_padded;
  std::vector<SIZE> shape_org;
  std::vector<SIZE> shape;
  std::vector<std::vector<SIZE>> dofs;
  std::vector<Array<1, SIZE, DeviceType>> shapes;
  Array<1, SIZE, DeviceType> ranges;
  std::vector<T *> coords_h; // do not copy this
  // std::vector<T *> coords_d;
  std::vector<Array<1, T, DeviceType>> coords;

  std::vector<std::vector<Array<1, T, DeviceType>>> dist_array;
  std::vector<std::vector<Array<1, T, DeviceType>>> ratio_array;

  Array<2, T, DeviceType> volumes_array;

  std::vector<std::vector<Array<1, T, DeviceType>>> am_array;
  std::vector<std::vector<Array<1, T, DeviceType>>> bm_array;

  LENGTH linearized_depth;
  LENGTH padded_linearized_depth;

  enum data_structure_type dstype;

  DIM *processed_n;
  DIM *unprocessed_n;

  Array<1, DIM, DeviceType> processed_dims[D];
  Array<1, DIM, DeviceType> unprocessed_dims[D];

  bool domain_decomposed = false;
  DIM domain_decomposed_dim;
  SIZE domain_decomposed_size;
  std::vector<Hierarchy<D, T, DeviceType>> hierarchy_chunck;

private:
  void padding_dimensions(std::vector<SIZE> &shape, std::vector<T *> &coords);
  std::vector<T *> create_uniform_coords(std::vector<SIZE> shape, int mode);
  void coord_to_dist(SIZE dof, T *coord, T *dist);
  void dist_to_ratio(SIZE dof, T *dist, T *ratio);
  void reduce_dist(SIZE dof, T *dist, T *dist2);
  void calc_am_bm(SIZE dof, T *dist, T *am, T *bm);
  void calc_volume(SIZE dof, T *dist, T *volume);
  size_t estimate_memory_usgae(std::vector<SIZE> shape);
  bool need_domain_decomposition(std::vector<SIZE> shape);
  void domain_decomposition_strategy(std::vector<SIZE> shape);
  void domain_decompose(std::vector<SIZE> shape, int uniform_coord_mode);
  void domain_decompose(std::vector<SIZE> shape, std::vector<T *> &coords);
  void init(std::vector<SIZE> shape, std::vector<T *> coords);
  void destroy();
  bool uniform_coords_created = false;
  bool initialized = false;
};

} // namespace mgard_x

#endif
