#ifndef _MDR_DECOMPOSER_INTERFACE_HPP
#define _MDR_DECOMPOSER_INTERFACE_HPP
#include "../../RuntimeX/RuntimeX.h"

namespace MDR {
namespace concepts {

// inplace data decomposer: de-correlates and overwrites original data
template <typename T> class DecomposerInterface {
public:
  virtual ~DecomposerInterface() = default;

  virtual void decompose(T *data, const std::vector<uint32_t> &dimensions,
                         uint32_t target_level) const = 0;

  virtual void recompose(T *data, const std::vector<uint32_t> &dimensions,
                         uint32_t target_level) const = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR

namespace mgard_x {
namespace MDR {
namespace concepts {

// inplace data decomposer: de-correlates and overwrites original data
template <DIM D, typename T>
class DecomposerInterface {
public:
  virtual ~DecomposerInterface() = default;

  virtual void decompose(SubArray<D, T, CUDA> v, SIZE target_level, int queue_idx) const = 0;

  virtual void recompose(SubArray<D, T, CUDA> v, SIZE target_level, int queue_idx) const = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR
} // namespace mgard_x

#endif
