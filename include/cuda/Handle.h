/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_HANDLE
#define MGARD_X_HANDLE

#include "Common.h"

namespace mgard_x {

struct Config {
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

template <DIM D, typename T> struct Handle {

  /* for Internal use only */
  Handle();

  /* for general users */
  Handle(std::vector<SIZE> shape);
  Handle(std::vector<SIZE> shape, std::vector<T *> coords);
  Handle(std::vector<SIZE> shape, Config config);
  Handle(std::vector<SIZE> shape, std::vector<T *> coords, Config config);
  ~Handle();

  void allocate_workspace();
  void free_workspace();
  void *get(int i);
  void sync(int i);
  void sync_all();

  /* CUDA env */
  void *queues;
  int num_of_queues;
  int dev_id = 0;

  /* Refactoring env */
  SIZE l_target;
  DIM D_padded;
  std::vector<SIZE> shape_org;
  std::vector<SIZE> shape;
  std::vector<std::vector<SIZE>> dofs;
  std::vector<SIZE *> shapes_h;
  std::vector<SIZE *> shapes_d;
  std::vector<Array<1, SIZE, CUDA>> shapes;
  SIZE * ranges_h;
  SIZE * ranges_d;
  Array<1, SIZE, CUDA> ranges;
  std::vector<T *> coords_h;
  std::vector<T *> coords_d;
  std::vector<std::vector<T *>> dist;
  std::vector<std::vector<T *>> ratio;

  std::vector<std::vector<Array<1, T, CUDA>>> dist_array;
  std::vector<std::vector<Array<1, T, CUDA>>> ratio_array;

  T * volumes;
  SIZE ldvolumes;
  Array<2, T, CUDA> volumes_array;
  std::vector<std::vector<T *>> am;
  std::vector<std::vector<T *>> bm;
  LENGTH linearized_depth;
  LENGTH padded_linearized_depth;

  enum data_structure_type dstype;
  T *quantizers;
  SIZE huff_dict_size;
  SIZE huff_block_size;
  SIZE lz4_block_size;
  enum lossless_type lossless;

  bool reduce_memory_footprint;
  bool profile_kernels;
  bool sync_and_check_all_kernels;
  bool timing;

  DIM *processed_n;
  DIM **processed_dims_h;
  DIM **processed_dims_d;

  DIM *unprocessed_n;
  DIM **unprocessed_dims_h;
  DIM **unprocessed_dims_d;

  Array<1, DIM, CUDA> processed_dims[D];
  Array<1, DIM, CUDA> unprocessed_dims[D];


  T *dw;
  SIZE lddw1, lddw2;
  std::vector<SIZE> ldws_h;
  SIZE *ldws_d;

  T *db;
  SIZE lddb1, lddb2;
  std::vector<SIZE> ldbs_h;
  SIZE *ldbs_d;

  int ***auto_tuning_cc;
  int ***auto_tuning_mr1, ***auto_tuning_ts1;
  int ***auto_tuning_mr2, ***auto_tuning_ts2;
  int ***auto_tuning_mr3, ***auto_tuning_ts3;
  int arch, precision;

private:
  void padding_dimensions(std::vector<SIZE> &shape, std::vector<T *> &coords);

  void create_queues();
  void destroy_queues();

  std::vector<T *> create_uniform_coords(std::vector<SIZE> shape, int mode);
  bool uniform_coords_created = false;

  int num_arch = 3;
  int num_precision = 2;
  int num_range = 9;
  void init_auto_tuning_table();
  void destroy_auto_tuning_table();
  bool auto_tuning_table_created = false;

  void coord_to_dist(SIZE dof, T * coord, T * dist);
  void dist_to_ratio(SIZE dof, T * dist, T * ratio);
  void reduce_dist(SIZE dof, T * dist, T * dist2);
  void calc_am_bm(SIZE dof, T * dist, T * am, T * bm);
  void calc_volume(SIZE dof, T * dist, T * volume);
  void init(std::vector<SIZE> shape, std::vector<T *> coords, Config config);

  void destroy();
  bool initialized = false;
};

} // namespace mgard_x

#endif
