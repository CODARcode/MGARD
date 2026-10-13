/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#include "../Config/Config.h"
#include "../RuntimeX/RuntimeX.h"
#include "../Utilities/Types.h"

#ifndef MGARD_X_160_HIERARCHY_H
#define MGARD_X_160_HIERARCHY_H

namespace mgard_x_160 {

template <DIM D, typename T, typename DeviceType> struct Hierarchy {

  /* for general users */
  Hierarchy(std::vector<SIZE> shape, Config config);
  Hierarchy(std::vector<SIZE> shape, std::vector<T *> coords, Config config);

  /* for Internal use only */
  Hierarchy();
  Hierarchy(const Hierarchy &hierarchy);
  // Moves hand over the device arrays (a moved-from Array is left empty), so
  // e.g. inserting into a hierarchy cache does not deep-copy every array.
  Hierarchy(Hierarchy &&hierarchy) = default;
  Hierarchy &operator=(const Hierarchy &hierarchy) = default;
  Hierarchy &operator=(Hierarchy &&hierarchy) = default;

  SIZE total_num_elems();
  SIZE level_num_elems(SIZE level);
  std::vector<SIZE> level_num_elems();
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
  Array<1, T, DeviceType> &level_volumes(bool reciprocal);
  Array<2, SIZE, DeviceType> &level_volume_offsets();
  data_structure_type data_structure();
  bool is_initialized();
  bool can_reuse(std::vector<SIZE> shape);
  // Whether a cached hierarchy is still valid for `config`. A plain hierarchy
  // does not depend on it; HybridHierarchy hides this to compare its level
  // counts.
  bool matches_config(Config config) { return true; }
  size_t EstimateMemoryFootprint(std::vector<SIZE> shape);
  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape, Config config);

  ~Hierarchy();

protected:
  // Describe only the input grid: one level equal to `shape`, so l_target(),
  // level_shape() and total_num_elems() work, but no device arrays (coords,
  // dist/ratio, am/bm, volumes, ranges/marks) are built. For hierarchies such
  // as HybridHierarchy whose decomposition does not use them.
  void init_shape_only(std::vector<SIZE> shape, data_structure_type dstype);

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
  // Pre-computed volume array for fast quantization. Stored per level: the
  // level_shape[l][d] volumes of each (level l, dim d) are packed back to
  // back starting at _level_volume_offsets(l, d), so the total is
  // sum_l sum_d level_shape[l][d] rather than (l_target + 1) * D * max_width
  // (which in 1D is ~log2(N) * N).
  Array<1, T, DeviceType> _level_volumes;
  Array<1, T, DeviceType> _level_volumes_reciprocal;
  Array<2, SIZE, DeviceType> _level_volume_offsets;
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

} // namespace mgard_x_160

#endif
