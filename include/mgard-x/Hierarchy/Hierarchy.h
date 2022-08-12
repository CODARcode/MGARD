/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_HIERARCHY_H
#define MGARD_X_HIERARCHY_H

#include "../RuntimeX/RuntimeXPublic.h"
#include "../Utilities/Types.h"

namespace mgard_x {

struct Config {
  device_type dev_type;
  int dev_id;
  enum decomposition_type decomposition;
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
  int reorder;

  Config() {
    dev_type = device_type::AUTO;
    dev_id = 0;
    decomposition = decomposition_type::MultiDim;
    l_target = -1; // no limit
    huff_dict_size = 8192;
    huff_block_size = 1024 * 20;
    lz4_block_size = 1 << 15;
    zstd_compress_level = 3;
    timing = false;
    uniform_coord_mode = 1;
    lossless = lossless_type::Huffman;
    global_norm = 1;
    reorder = 0;
  }
};

template <DIM D, typename T, typename DeviceType> struct Hierarchy {

  /* for general users */
  Hierarchy(std::vector<SIZE> shape, int uniform_coord_mode = 1,
            SIZE target_level = 0);
  Hierarchy(std::vector<SIZE> shape, std::vector<T *> coords,
            SIZE target_level = 0);

  /* for Internal use only */
  Hierarchy();
  Hierarchy(std::vector<SIZE> shape, DIM domain_decomposed_dim,
            SIZE domain_decomposed_size, int uniform_coord_mode = 1);
  Hierarchy(std::vector<SIZE> shape, DIM domain_decomposed_dim,
            SIZE domain_decomposed_size, std::vector<T *> coords);
  Hierarchy(const Hierarchy &hierarchy);

  SIZE total_num_elems();
  SIZE linearized_width();
  SIZE l_target();
  std::vector<SIZE> level_shape(SIZE level);
  SIZE level_shape(SIZE level, DIM dim);
  Array<1, SIZE, DeviceType> &level_shape_array(SIZE level);
  Array<1, T, DeviceType> &dist(SIZE level, DIM dim);
  Array<1, T, DeviceType> &ratio(SIZE level, DIM dim);
  Array<1, T, DeviceType> &am(SIZE level, DIM dim);
  Array<1, T, DeviceType> &bm(SIZE level, DIM dim);
  Array<1, DIM, DeviceType> &processed(SIZE idx, DIM &processed_n);
  Array<1, DIM, DeviceType> &unprocessed(SIZE idx, DIM &processed_n);
  Array<2, SIZE, DeviceType> &level_ranges();
  Array<3, T, DeviceType> &level_volumes();

  ~Hierarchy();

  /* Refactoring env */

  // For domain decomposition
  bool domain_decomposed = false;
  DIM domain_decomposed_dim;
  SIZE domain_decomposed_size;
  std::vector<Hierarchy<D, T, DeviceType>> hierarchy_chunck;

private:
  // Shape of the finest grid
  std::vector<SIZE> shape;
  // Pre-computed varaibles
  SIZE _total_num_elems;
  SIZE _linearized_width;
  // For out-of-bound returns
  Array<1, T, DeviceType> dummy_array;
  // Target level: number of times of decomposition
  // Total number of levels = l_target+1
  SIZE _l_target;
  // Shape of grid of each level
  std::vector<std::vector<SIZE>> _level_shape;
  std::vector<Array<1, SIZE, DeviceType>> _level_shape_array;
  // Coordinates
  std::vector<Array<1, T, DeviceType>> _coords_org;
  // Pre-computed cell distance array
  std::vector<std::vector<Array<1, T, DeviceType>>> _dist_array;
  // Pre-computed neighbor cell distance ratio array
  std::vector<std::vector<Array<1, T, DeviceType>>> _ratio_array;
  // Pre-computed coefficients for solving tri-diag system
  std::vector<std::vector<Array<1, T, DeviceType>>> _am_array;
  std::vector<std::vector<Array<1, T, DeviceType>>> _bm_array;
  // Pre-computed markers for processing high dimensional data
  DIM _processed_n[D];
  DIM _unprocessed_n[D];
  Array<1, DIM, DeviceType> _processed_dims[D];
  Array<1, DIM, DeviceType> _unprocessed_dims[D];
  // Pre-computed range array for fast quantization
  Array<2, SIZE, DeviceType> _level_ranges;
  // Pre-computed volume array for fast quantization
  Array<3, T, DeviceType> _level_volumes;
  // Indicating if it is uniform or non-uniform grid
  enum data_structure_type dstype;

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
  void init(std::vector<SIZE> shape, std::vector<T *> coords,
            SIZE target_level = 0);
  void destroy();
  bool uniform_coords_created = false;
  bool initialized = false;
};

} // namespace mgard_x

#endif
