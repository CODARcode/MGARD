/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_HANDLE
#define MGARD_X_HANDLE

#include "Types.h"
#include "RuntimeX/RuntimeXPublic.h"

namespace mgard_x {

struct Config {
  device_type dev_type;
  int dev_id;
  SIZE l_target;
  SIZE huff_dict_size;
  SIZE huff_block_size;
  SIZE lz4_block_size;
  bool reduce_memory_footprint;
  bool profile_kernels;
  bool sync_and_check_all_kernels;
  bool timing;
  int uniform_coord_mode;
  enum lossless_type lossless;

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
    reduce_memory_footprint = false;
    profile_kernels = false;
    sync_and_check_all_kernels = false;
    timing = false;
    uniform_coord_mode = 0;
    lossless = lossless_type::GPU_Huffman;
  }
};

template <DIM D, typename T, typename DeviceType> struct Handle {

  /* for Internal use only */
  Handle();

  /* for general users */
  Handle(std::vector<SIZE> shape);
  Handle(std::vector<SIZE> shape, std::vector<T *> coords);
  Handle(std::vector<SIZE> shape, Config config);
  Handle(std::vector<SIZE> shape, std::vector<T *> coords, Config config);
  ~Handle();


  /* Refactoring env */
  enum device_type dev_type;
  int dev_id;
  SIZE l_target;
  DIM D_padded;
  std::vector<SIZE> shape_org;
  std::vector<SIZE> shape;
  std::vector<std::vector<SIZE>> dofs;
  std::vector<Array<1, SIZE, DeviceType>> shapes;
  SIZE * ranges_h;
  Array<1, SIZE, DeviceType> ranges;
  std::vector<T *> coords_h;
  std::vector<T *> coords_d;
  std::vector<Array<1, T, DeviceType>> coords;

  std::vector<std::vector<Array<1, T, DeviceType>>> dist_array;
  std::vector<std::vector<Array<1, T, DeviceType>>> ratio_array;

  Array<2, T, DeviceType> volumes_array;

  std::vector<std::vector<Array<1, T, DeviceType>>> am_array;
  std::vector<std::vector<Array<1, T, DeviceType>>> bm_array;

  LENGTH linearized_depth;
  LENGTH padded_linearized_depth;

  enum data_structure_type dstype;
  SIZE huff_dict_size;
  SIZE huff_block_size;
  SIZE lz4_block_size;
  enum lossless_type lossless;

  bool reduce_memory_footprint;
  bool profile_kernels;
  bool sync_and_check_all_kernels;
  bool timing;

  DIM *processed_n;
  DIM *unprocessed_n;

  Array<1, DIM, DeviceType> processed_dims[D];
  Array<1, DIM, DeviceType> unprocessed_dims[D];

private:
  void padding_dimensions(std::vector<SIZE> &shape, std::vector<T *> &coords);
  std::vector<T *> create_uniform_coords(std::vector<SIZE> shape, int mode);
  void coord_to_dist(SIZE dof, T * coord, T * dist);
  void dist_to_ratio(SIZE dof, T * dist, T * ratio);
  void reduce_dist(SIZE dof, T * dist, T * dist2);
  void calc_am_bm(SIZE dof, T * dist, T * am, T * bm);
  void calc_volume(SIZE dof, T * dist, T * volume);
  void init(std::vector<SIZE> shape, std::vector<T *> coords, Config config);
  void destroy();
  bool uniform_coords_created = false;
  bool initialized = false;
};




} // namespace mgard_x

#endif
