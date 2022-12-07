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
template <DIM D, typename T, typename DeviceType> class DecomposerInterface {
public:
  virtual ~DecomposerInterface() = default;

  virtual void decompose(Array<D, T, DeviceType> &v, int start_level,
                         int stop_level, int queue_idx) = 0;

  virtual void recompose(Array<D, T, DeviceType> &v, int start_level,
                         int stop_level, int queue_idx) = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR
} // namespace mgard_x

#endif
