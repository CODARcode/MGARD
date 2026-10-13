/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: October 1, 2026
 */

#ifndef MGARD_X_160_HYBRID_HIERARCHY_H
#define MGARD_X_160_HYBRID_HIERARCHY_H

#include "../RuntimeX/Utilities/Exceptions.h"
#include "Hierarchy.h"

namespace mgard_x_160 {

// Hierarchy of the hybrid (BlockMGARD) decomposition: L block-local levels on
// the input grid, then M global multilevel levels on the coarsest local grid
// (or on the input grid when L == 0).
//
// The block-local stage needs nothing from a Hierarchy beyond the input shape
// and the padded per-level block grids, so the base Hierarchy only describes
// the input grid (one level, no device arrays -- see init_shape_only). The
// only full Hierarchy is the global one, built here once, on the grid the
// global stage actually decomposes, and only when M > 0.
template <DIM D, typename T, typename DeviceType>
class HybridHierarchy : public Hierarchy<D, T, DeviceType> {
public:
  HybridHierarchy() {}

  HybridHierarchy(std::vector<SIZE> shape, Config config) {
    init(shape, config, data_structure_type::Cartesian_Grid_Uniform);
  }

  // Coordinates are accepted for parity with Hierarchy (DomainDecomposer
  // builds hierarchies either way) but not used: the block-local transform
  // assumes uniform spacing and the global stage is built on a uniform grid.
  // The grid is still marked non-uniform, so the compressor cache treats it
  // the same as a non-uniform Hierarchy.
  HybridHierarchy(std::vector<SIZE> shape, std::vector<T *> coords,
                  Config config) {
    init(shape, config, data_structure_type::Cartesian_Grid_Non_Uniform);
  }

  // Shape of the input data.
  std::vector<SIZE> input_shape() { return this->level_shape(0); }

  SIZE num_local_levels() { return L; }
  SIZE num_global_levels() { return M; }

  // Block-local level l (0 = finest) operates on local_fine_shape(l): the
  // previous grid padded up to whole blocks. It produces local_coarse_shape(l),
  // MGARDX_HYBRID_LOCAL_COARSE_SIZE nodes per block per dimension.
  std::vector<SIZE> local_fine_shape(SIZE l) { return fine_shapes[l]; }
  std::vector<SIZE> local_coarse_shape(SIZE l) { return coarse_shapes[l]; }

  // Grid the global stage decomposes: the coarsest local grid, or the input
  // grid when there is no local stage.
  std::vector<SIZE> global_shape() {
    return L > 0 ? coarse_shapes[L - 1] : input_shape();
  }

  // Hierarchy of the global stage. Only exists when M > 0.
  Hierarchy<D, T, DeviceType> &global_hierarchy() {
    if (M == 0) {
      throw ProcessingException(
          "HybridHierarchy: no global hierarchy without global levels.");
    }
    return global;
  }

  // Config the global stage is built and run with: defaults, limited to M
  // levels.
  static Config global_config(SIZE M) {
    Config config;
    config.num_global_refactoring_level = M;
    return config;
  }

  bool matches_config(Config config) {
    return L == config.num_local_refactoring_level &&
           M == config.num_global_refactoring_level;
  }

  // The hybrid components take their level counts from the config and their
  // grids from this hierarchy, so the two have to agree.
  void check_config(Config config) {
    if (!matches_config(config)) {
      throw ProcessingException(
          "HybridHierarchy was built for " + std::to_string(L) + " local / " +
          std::to_string(M) + " global levels, but config has " +
          std::to_string(config.num_local_refactoring_level) + " / " +
          std::to_string(config.num_global_refactoring_level) + ".");
    }
  }

  // Padded block grids of the L local levels of an input of `shape`.
  static void compute_local_shapes(std::vector<SIZE> shape, SIZE L,
                                   std::vector<std::vector<SIZE>> &fine,
                                   std::vector<std::vector<SIZE>> &coarse) {
    fine.clear();
    coarse.clear();
    std::vector<SIZE> curr = shape;
    for (SIZE l = 0; l < L; l++) {
      std::vector<SIZE> f(D), c(D);
      for (DIM d = 0; d < D; d++) {
        SIZE num_blocks = (curr[d] - 1) / MGARDX_HYBRID_LOCAL_BLOCK_SIZE + 1;
        f[d] = num_blocks * MGARDX_HYBRID_LOCAL_BLOCK_SIZE;
        c[d] = num_blocks * MGARDX_HYBRID_LOCAL_COARSE_SIZE;
      }
      fine.push_back(f);
      coarse.push_back(c);
      curr = c;
    }
  }

  // Device memory of a HybridHierarchy for `shape`: only the global
  // hierarchy, if any (the base part allocates nothing). Uses the Hierarchy
  // estimate, which assumes all levels, so it is an upper bound for M levels.
  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape,
                                        Config config) {
    SIZE L = config.num_local_refactoring_level;
    SIZE M = config.num_global_refactoring_level;
    if (M == 0) {
      return 0;
    }
    std::vector<SIZE> global_input = shape;
    if (L > 0) {
      std::vector<std::vector<SIZE>> fine, coarse;
      compute_local_shapes(shape, L, fine, coarse);
      global_input = coarse[L - 1];
    }
    return Hierarchy<D, T, DeviceType>::EstimateMemoryFootprint(
        global_input, global_config(M));
  }

  // Same, for this hierarchy's own level counts.
  size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    Config config;
    config.num_local_refactoring_level = L;
    config.num_global_refactoring_level = M;
    return EstimateMemoryFootprint(shape, config);
  }

private:
  void init(std::vector<SIZE> shape, Config config,
            data_structure_type dstype) {
    L = config.num_local_refactoring_level;
    M = config.num_global_refactoring_level;
    if (L == 0 && M == 0) {
      throw ProcessingException(
          "HybridHierarchy: local and global levels cannot both be zero.");
    }
    this->init_shape_only(shape, dstype);
    compute_local_shapes(shape, L, fine_shapes, coarse_shapes);
    if (M > 0) {
      global = Hierarchy<D, T, DeviceType>(global_shape(), global_config(M));
    }
  }

  SIZE L = 0;
  SIZE M = 0;
  std::vector<std::vector<SIZE>> fine_shapes;
  std::vector<std::vector<SIZE>> coarse_shapes;
  Hierarchy<D, T, DeviceType> global;
};

} // namespace mgard_x_160

#endif
