#ifndef _MDR_MGARD_DECOMPOSER_HPP
#define _MDR_MGARD_DECOMPOSER_HPP

#include "../../DataRefactoring/DataRefactor.hpp"
#include "DecomposerInterface.hpp"
// #include "decompose.hpp"
// #include "recompose.hpp"
#include <cstring>
// namespace MDR {
// // MGARD decomposer with orthogonal basis
// template <typename T>
// class MGARDOrthoganalDecomposer : public concepts::DecomposerInterface<T> {
// public:
//   MGARDOrthoganalDecomposer() {}
//   void decompose(T *data, const std::vector<uint32_t> &dimensions,
//                  uint32_t target_level) const {
//     MGARD::Decomposer<T> decomposer;
//     std::vector<size_t> dims(dimensions.size());
//     for (int i = 0; i < dims.size(); i++) {
//       dims[i] = dimensions[i];
//     }
//     decomposer.decompose(data, dims, target_level);
//   }
//   void recompose(T *data, const std::vector<uint32_t> &dimensions,
//                  uint32_t target_level) const {
//     MGARD::Recomposer<T> recomposer;
//     std::vector<size_t> dims(dimensions.size());
//     for (int i = 0; i < dims.size(); i++) {
//       dims[i] = dimensions[i];
//     }
//     recomposer.recompose(data, dims, target_level);
//   }
//   void print() const {
//     std::cout << "MGARD orthogonal decomposer" << std::endl;
//   }
// };
// // MGARD decomposer with hierarchical basis
// template <typename T>
// class MGARDHierarchicalDecomposer : public concepts::DecomposerInterface<T> {
// public:
//   MGARDHierarchicalDecomposer() {}
//   void decompose(T *data, const std::vector<uint32_t> &dimensions,
//                  uint32_t target_level) const {
//     MGARD::Decomposer<T> decomposer;
//     std::vector<size_t> dims(dimensions.size());
//     for (int i = 0; i < dims.size(); i++) {
//       dims[i] = dimensions[i];
//     }
//     decomposer.decompose(data, dims, target_level, true);
//   }
//   void recompose(T *data, const std::vector<uint32_t> &dimensions,
//                  uint32_t target_level) const {
//     MGARD::Recomposer<T> recomposer;
//     std::vector<size_t> dims(dimensions.size());
//     for (int i = 0; i < dims.size(); i++) {
//       dims[i] = dimensions[i];
//     }
//     recomposer.recompose(data, dims, target_level, true);
//   }
//   void print() const {
//     std::cout << "MGARD hierarchical decomposer" << std::endl;
//   }
// };
// } // namespace MDR

namespace mgard_x {
namespace MDR {
// MGARD decomposer with orthogonal basis
template <DIM D, typename T, typename DeviceType>
class MGARDOrthoganalDecomposer
    : public concepts::DecomposerInterface<D, T, DeviceType> {
public:
  MGARDOrthoganalDecomposer(Hierarchy<D, T, DeviceType> hierarchy)
      : hierarchy(hierarchy), refactor(this->hierarchy, Config()) {}
  static size_t EstimateMemoryFootprint(std::vector<SIZE> shape) {
    size_t size = 0;
    Hierarchy<D, T, DeviceType> hierarchy;
    size += hierarchy.EstimateMemoryFootprint(shape);
    size += data_refactoring::DataRefactor<
        D, T, DeviceType>::EstimateMemoryFootprint(shape);
    return size;
  }
  void decompose(Array<D, T, DeviceType> &v, int start_level, int stop_level,
                 int queue_idx) {
    refactor.Decompose(v, start_level, stop_level, queue_idx);
  }
  void recompose(Array<D, T, DeviceType> &v, int start_level, int stop_level,
                 int queue_idx) {
    refactor.Recompose(v, start_level, stop_level, queue_idx);
  }
  void print() const {
    std::cout << "MGARD orthogonal decomposer" << std::endl;
  }

private:
  Hierarchy<D, T, DeviceType> hierarchy;
  data_refactoring::DataRefactor<D, T, DeviceType> refactor;
};

} // namespace MDR
} // namespace mgard_x
#endif
