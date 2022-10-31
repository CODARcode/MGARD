#ifndef _MDR_INTERLEAVER_INTERFACE_HPP
#define _MDR_INTERLEAVER_INTERFACE_HPP

#include "../../RuntimeX/RuntimeX.h"
namespace MDR {
namespace concepts {

// level data interleaver: interleave level coefficients
template <typename T> class InterleaverInterface {
public:
  virtual ~InterleaverInterface() = default;

  virtual void interleave(T const *data, const std::vector<uint32_t> &dims,
                          const std::vector<uint32_t> &dims_fine,
                          const std::vector<uint32_t> &dims_coasre,
                          T *buffer) const = 0;

  virtual void reposition(T const *buffer, const std::vector<uint32_t> &dims,
                          const std::vector<uint32_t> &dims_fine,
                          const std::vector<uint32_t> &dims_coasre,
                          T *data) const = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR

namespace mgard_x {
namespace MDR {
namespace concepts {

// level data interleaver: interleave level coefficients
template <DIM D, typename T, typename DeviceType> class InterleaverInterface {
public:
  virtual ~InterleaverInterface() = default;

  virtual void interleave(SubArray<D, T, DeviceType> decomposed_data,
                          SubArray<1, T, DeviceType> *levels_decomposed_data,
                          SIZE num_levels, int queue_idx) = 0;

  virtual void reposition(SubArray<1, T, DeviceType> *levels_decomposed_data,
                          SubArray<D, T, DeviceType> decomposed_data,
                          SIZE num_levels, int queue_idx) = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR
} // namespace mgard_x

#endif
