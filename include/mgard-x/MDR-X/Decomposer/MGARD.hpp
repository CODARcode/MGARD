#ifndef _MDR_MGARD_DECOMPOSER_HPP
#define _MDR_MGARD_DECOMPOSER_HPP

#include "../../DataRefactoring/DataRefactoring.hpp"
#include "DecomposerInterface.hpp"
#include "decompose.hpp"
#include "recompose.hpp"
#include <cstring>
namespace MDR {
// MGARD decomposer with orthogonal basis
template <typename T>
class MGARDOrthoganalDecomposer : public concepts::DecomposerInterface<T> {
public:
  MGARDOrthoganalDecomposer() {}
  void decompose(T *data, const std::vector<uint32_t> &dimensions,
                 uint32_t target_level) const {
    MGARD::Decomposer<T> decomposer;
    std::vector<size_t> dims(dimensions.size());
    for (int i = 0; i < dims.size(); i++) {
      dims[i] = dimensions[i];
    }
    decomposer.decompose(data, dims, target_level);
  }
  void recompose(T *data, const std::vector<uint32_t> &dimensions,
                 uint32_t target_level) const {
    MGARD::Recomposer<T> recomposer;
    std::vector<size_t> dims(dimensions.size());
    for (int i = 0; i < dims.size(); i++) {
      dims[i] = dimensions[i];
    }
    recomposer.recompose(data, dims, target_level);
  }
  void print() const {
    std::cout << "MGARD orthogonal decomposer" << std::endl;
  }
};
// MGARD decomposer with hierarchical basis
template <typename T>
class MGARDHierarchicalDecomposer : public concepts::DecomposerInterface<T> {
public:
  MGARDHierarchicalDecomposer() {}
  void decompose(T *data, const std::vector<uint32_t> &dimensions,
                 uint32_t target_level) const {
    MGARD::Decomposer<T> decomposer;
    std::vector<size_t> dims(dimensions.size());
    for (int i = 0; i < dims.size(); i++) {
      dims[i] = dimensions[i];
    }
    decomposer.decompose(data, dims, target_level, true);
  }
  void recompose(T *data, const std::vector<uint32_t> &dimensions,
                 uint32_t target_level) const {
    MGARD::Recomposer<T> recomposer;
    std::vector<size_t> dims(dimensions.size());
    for (int i = 0; i < dims.size(); i++) {
      dims[i] = dimensions[i];
    }
    recomposer.recompose(data, dims, target_level, true);
  }
  void print() const {
    std::cout << "MGARD hierarchical decomposer" << std::endl;
  }
};
} // namespace MDR

namespace mgard_x {
namespace MDR {
// MGARD decomposer with orthogonal basis
template <DIM D, typename T, typename DeviceType>
class MGARDOrthoganalDecomposer
    : public concepts::DecomposerInterface<D, T, DeviceType> {
public:
  MGARDOrthoganalDecomposer(Hierarchy<D, T, DeviceType> &hierarchy)
      : hierarchy(hierarchy) {
        workspace = DataRefactoringWorkspace<D, T, DeviceType>(hierarchy);
      }
  void decompose(SubArray<D, T, DeviceType> v, SIZE target_level,
                 int queue_idx) const {
    mgard_x::decompose<D, T, DeviceType>(
        hierarchy, v, workspace.refactoring_w_subarray,
        workspace.refactoring_b_subarray, 0, queue_idx);
    mgard_x::DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  }
  void recompose(SubArray<D, T, DeviceType> v, SIZE target_level,
                 int queue_idx) const {
    mgard_x::recompose<D, T, DeviceType>(
        hierarchy, v, workspace.refactoring_w_subarray,
        workspace.refactoring_b_subarray, target_level, queue_idx);
    mgard_x::DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  }
  void print() const {
    std::cout << "MGARD orthogonal decomposer" << std::endl;
  }

private:
  Hierarchy<D, T, DeviceType> &hierarchy;
  DataRefactoringWorkspace<D, T, DeviceType> workspace;
};

} // namespace MDR
} // namespace mgard_x
#endif
