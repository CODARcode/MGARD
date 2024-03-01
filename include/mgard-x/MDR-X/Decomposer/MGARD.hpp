#ifndef _MDR_MGARD_DECOMPOSER_HPP
#define _MDR_MGARD_DECOMPOSER_HPP

#include "../../DataRefactoring/DataRefactor.hpp"
#include "DecomposerInterface.hpp"

#include <cstring>

namespace mgard_x {
namespace MDR {
// MGARD decomposer with orthogonal basis
template <DIM D, typename T, typename DeviceType>
class MGARDOrthoganalDecomposer
    : public concepts::DecomposerInterface<D, T, DeviceType> {
public:
  MGARDOrthoganalDecomposer() : initialized(false) {}
  MGARDOrthoganalDecomposer(Hierarchy<D, T, DeviceType> &hierarchy,
                            Config config) {
    Adapt(hierarchy, config, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  void Adapt(Hierarchy<D, T, DeviceType> &hierarchy, Config config,
             int queue_idx) {
    this->initialized = true;
    this->hierarchy = &hierarchy;
    refactor.Adapt(hierarchy, config, queue_idx);
  }
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
  bool initialized;
  Hierarchy<D, T, DeviceType> *hierarchy;
  data_refactoring::DataRefactor<D, T, DeviceType> refactor;
};

} // namespace MDR
} // namespace mgard_x
#endif
