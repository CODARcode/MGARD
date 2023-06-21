/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_ENCODE_FIXED_LEN_TEMPLATE_HPP
#define MGARD_X_ENCODE_FIXED_LEN_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x {

// Jieyang: this kernel rely on whole grid sychronized execution
// For example, adding
//   if (thread == 0) {
//      __nanosleep(1e9);
//   }
// will cause incorrect results

template <typename Q, typename H, typename DeviceType>
class EncodeFixedLenFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT EncodeFixedLenFunctor() {}
  MGARDX_CONT EncodeFixedLenFunctor(SubArray<1, Q, DeviceType> data,
                                    SubArray<1, H, DeviceType> hcoded,
                                    SubArray<1, H, DeviceType> codebook)
      : data(data), hcoded(hcoded), codebook(codebook) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    unsigned int gid = (FunctorBase<DeviceType>::GetBlockIdX() *
                        FunctorBase<DeviceType>::GetBlockDimX()) +
                       FunctorBase<DeviceType>::GetThreadIdX();
    if (gid >= data.shape(0))
      return;
    *hcoded(gid) = *codebook(*data(gid)); // try to exploit cache?
  }

  MGARDX_EXEC void Operation2() {}

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, Q, DeviceType> data;
  SubArray<1, H, DeviceType> hcoded;
  SubArray<1, H, DeviceType> codebook;
};

template <typename Q, typename H, typename DeviceType>
class EncodeFixedLenKernel : public Kernel {
public:
  constexpr static DIM NumDim = 1;
  using DataType = H;
  constexpr static std::string_view Name = "encode";
  MGARDX_CONT
  EncodeFixedLenKernel(SubArray<1, Q, DeviceType> data,
                       SubArray<1, H, DeviceType> hcoded,
                       SubArray<1, H, DeviceType> codebook)
      : data(data), hcoded(hcoded), codebook(codebook) {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<EncodeFixedLenFunctor<Q, H, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = EncodeFixedLenFunctor<Q, H, DeviceType>;
    FunctorType functor(data, hcoded, codebook);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = 1;
    gridy = 1;
    gridx = (data.shape(0) - 1) / tbx + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, Q, DeviceType> data;
  SubArray<1, H, DeviceType> hcoded;
  SubArray<1, H, DeviceType> codebook;
};

} // namespace mgard_x

#endif