/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_HANDLE
#define MGARD_X_HANDLE

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
  // Hierarchy(std::vector<SIZE> shape, Config config, int uniform_coord_mode =
  // 0); Hierarchy(std::vector<SIZE> shape, std::vector<T *> coords, Config
  // config);

  ~Hierarchy();

  /* Refactoring env */
  // enum device_type dev_type;
  // int dev_id;
  /// Target number of levels in the hierarchy.
  SIZE l_target;
  /// A padded version of the number of dimensions. This will always be as big
  /// as `D` or larger. It will also always be 3 or larger. If `D` is even,
  /// one is added to make `D_padded` odd.
  DIM D_padded;
  /// The size of each dimension. The length of this array is `D`.
  std::vector<SIZE> shape_org;
  /// The size of each dimension in reverse (k,j,i) order. The length of this
  /// array is `D`. This is the same as `shape_org` except in reverse order.
  std::vector<SIZE> shape;
  /// The degrees of freedom (i.e., number of values) along each dimension for
  /// each level. the length of the outer vector is `D`, and the length of
  /// each inner vector is `l_target`.
  std::vector<std::vector<SIZE>> dofs;
  /// For each level, the shape of the data at that point in the hierarchy.
  /// The outer vector is of size `l_target + 1`. Each inner array is of
  /// size `D_padded`.
  std::vector<Array<1, SIZE, DeviceType>> shapes;
  /// Same as `shapes` but used on the host instead of the device.
  std::vector<std::vector<SIZE>> shapes2;
  /// The degrees of freedom along each dimension for each level. This array
  /// has the same information as `dofs` with the following exceptions:
  /// 1. The 2D data is flattened into a single array so that it can be
  ///    accessed on the device.
  /// 2. The level order is reversed.
  /// 3. The levels are padded with a 0 value at the beginning.
  std::vector<std::vector<SIZE>> shapes_vec;
  Array<1, SIZE, DeviceType> ranges;
  /// An array of coordinates along each dimension. The vector is of size
  /// `D`. Each array should be the size of the associated dimension.
  std::vector<T *> coords_h; // do not copy this
  // std::vector<T *> coords_d;
  /// An array of coordinates along each dimension. The vector is of size
  /// `D`. Each array should be the size of the associated dimension.
  std::vector<Array<1, T, DeviceType>> coords;

  DIM D_pad;
  std::vector<SIZE> shape_org_padded;
  std::vector<std::vector<SIZE>> level_shape;
  std::vector<Array<1, SIZE, DeviceType>> level_shape_array;
  Array<1, SIZE, DeviceType> ranges_org;
  std::vector<T *> coords_h_org;
  std::vector<Array<1, T, DeviceType>> coords_org;
  std::vector<std::vector<Array<1, T, DeviceType>>> dist_org_array;
  std::vector<std::vector<Array<1, T, DeviceType>>> ratio_org_array;
  Array<2, T, DeviceType> volumes_org_array;
  std::vector<std::vector<Array<1, T, DeviceType>>> am_org_array;
  std::vector<std::vector<Array<1, T, DeviceType>>> bm_org_array;


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
  void init(std::vector<SIZE> shape, std::vector<T *> coords,
            SIZE target_level = 0);
  void destroy();
  bool uniform_coords_created = false;
  bool initialized = false;
};

} // namespace mgard_x

#endif
