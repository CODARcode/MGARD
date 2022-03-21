#ifndef _MDR_MGARD_DECOMPOSER_HPP
#define _MDR_MGARD_DECOMPOSER_HPP

#include "DecomposerInterface.hpp"
#include "decompose.hpp"
#include "recompose.hpp"

namespace MDR {
// MGARD decomposer with orthogonal basis
template <class T>
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
template <class T>
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
#endif
