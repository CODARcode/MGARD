/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#include "../Config/Config.h"
#include "../RuntimeX/RuntimeX.h"
#include "../Utilities/Types.h"

#ifndef MGARD_X_HIERARCHY_H
#define MGARD_X_HIERARCHY_H

namespace mgard_x {

template <DIM D, typename T, typename DeviceType> struct Hierarchy {

  /* for general users */
  Hierarchy(std::vector<SIZE> shape, Config config);
  Hierarchy(std::vector<SIZE> shape, std::vector<T *> coords, Config config);

  /* for Internal use only */
  Hierarchy();
  Hierarchy(const Hierarchy &hierarchy);

  SIZE total_num_elems();
  SIZE level_num_elems(SIZE level);
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
  Array<2, int, DeviceType> &level_marks();
  Array<3, T, DeviceType> &level_volumes(bool reciprocal);
  data_structure_type data_structure();
  bool is_initialized();
  bool can_reuse(std::vector<SIZE> shape);
  size_t EstimateMemoryFootprint(std::vector<SIZE> shape);

  ~Hierarchy();

private:
  // Shape of the finest grid
  std::vector<SIZE> shape;
  // Pre-computed varaibles
  SIZE _total_num_elems;
  std::vector<SIZE> _level_num_elems;
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
  Array<2, int, DeviceType> _level_marks;
  // Pre-computed volume array for fast quantization
  Array<3, T, DeviceType> _level_volumes;
  Array<3, T, DeviceType> _level_volumes_reciprocal;
  // Indicating if it is uniform or non-uniform grid
  enum data_structure_type dstype;

  std::vector<T *> create_uniform_coords(std::vector<SIZE> shape,
                                         bool normalize_coordinates);
  void coord_to_dist(SIZE dof, T *coord, T *dist);
  void dist_to_ratio(SIZE dof, T *dist, T *ratio);
  void reduce_dist(SIZE dof, T *dist, T *dist2);
  void calc_am_bm(SIZE dof, T *dist, T *am, T *bm);
  void calc_volume(SIZE dof, T *dist, T *volume, bool reciprocal);
  void init(std::vector<SIZE> shape, std::vector<T *> coords,
            SIZE target_level = 0);
  void destroy();
  bool uniform_coords_created = false;
  bool initialized = false;
};

} // namespace mgard_x

#endif
