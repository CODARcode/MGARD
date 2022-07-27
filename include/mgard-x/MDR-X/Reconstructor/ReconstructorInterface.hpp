#ifndef _MDR_RECONSTRUCTOR_INTERFACE_HPP
#define _MDR_RECONSTRUCTOR_INTERFACE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace MDR {
namespace concepts {

// reconstructor: a general interface for scientific data reconstructor
template <typename T> class ReconstructorInterface {
public:
  virtual ~ReconstructorInterface() = default;

  virtual T *reconstruct(double tolerance) = 0;

  virtual T *progressive_reconstruct(double tolerance) = 0;

  virtual void load_metadata() = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR

namespace mgard_x {
namespace MDR {
namespace concepts {

// reconstructor: a general interface for scientific data reconstructor
template <DIM D, typename T_data, typename T_bitplane, typename DeviceType>
class ReconstructorInterface {
public:
  virtual ~ReconstructorInterface() = default;

  virtual void reconstruct(double tolerance) = 0;

  virtual Array<D, T_data, DeviceType> progressive_reconstruct(double tolerance) = 0;

  virtual void load_metadata() = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR
} // namespace mgard_x

#endif
