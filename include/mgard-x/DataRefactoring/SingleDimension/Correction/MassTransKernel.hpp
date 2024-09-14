/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_SINGLE_DIMENSION_MASSTRANS_KERNEL_TEMPLATE
#define MGARD_X_SINGLE_DIMENSION_MASSTRANS_KERNEL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

#include "../../MultiDimension/Correction/LPKFunctor.h"

namespace mgard_x {

namespace data_refactoring {

namespace single_dimension {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class SingleDimensionMassTransFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT SingleDimensionMassTransFunctor() {}
  MGARDX_CONT SingleDimensionMassTransFunctor(DIM current_dim,
                                              SubArray<1, T, DeviceType> dist,
                                              SubArray<1, T, DeviceType> ratio,
                                              SubArray<D, T, DeviceType> coeff,
                                              SubArray<D, T, DeviceType> v)
      : current_dim(current_dim), dist(dist), ratio(ratio), coeff(coeff), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE v_idx[D];

    SIZE firstD = div_roundup(v.shape(D - 1), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    v_idx[D - 1] =
        (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    bidx /= firstD;
    if (D >= 2)
      v_idx[D - 2] = FunctorBase<DeviceType>::GetBlockIdY() *
                         FunctorBase<DeviceType>::GetBlockDimY() +
                     FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      v_idx[D - 3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                         FunctorBase<DeviceType>::GetBlockDimZ() +
                     FunctorBase<DeviceType>::GetThreadIdZ();

    for (int d = D - 4; d >= 0; d--) {
      v_idx[d] = bidx % v.shape(d);
      bidx /= v.shape(d);
    }

    bool in_range = true;
    for (int d = D - 1; d >= 0; d--) {
      if (v_idx[d] >= v.shape(d))
        in_range = false;
    }

    if (in_range) {

      const T a = 0.0;
      const T c = 0.0;
      const T e = 0.0;
      T b = 0.0;
      T d = 0.0;

      if (v_idx[current_dim] > 0 &&
          v_idx[current_dim] < coeff.shape(current_dim)) {
        v_idx[current_dim]--;
        b = coeff[v_idx];
        v_idx[current_dim]++;
      }

      if (v_idx[current_dim] < coeff.shape(current_dim)) {
        // v_idx[current_dim] ++;
        d = coeff[v_idx];
        // v_idx[current_dim] --;
      }

      T h1 = 0, h2 = 0, h3 = 0, h4 = 0;
      T r1 = 0, r2 = 0, r3 = 0, r4 = 0;

      if (v_idx[current_dim] > 0 &&
          v_idx[current_dim] * 2 <
              coeff.shape(current_dim) + v.shape(current_dim) - 1) {
        h1 = *dist(v_idx[current_dim] * 2 - 2);
        h2 = *dist(v_idx[current_dim] * 2 - 1);
        r1 = *ratio(v_idx[current_dim] * 2 - 2);
        r2 = *ratio(v_idx[current_dim] * 2 - 1);
      }
      if (v_idx[current_dim] * 2 <
          coeff.shape(current_dim) + v.shape(current_dim) - 1) {
        h3 = *dist(v_idx[current_dim] * 2);
        h4 = *dist(v_idx[current_dim] * 2 + 1);
        r3 = *ratio(v_idx[current_dim] * 2);
        r4 = 1 - r3;
      }

      // printf("v_idx = [%d %d] %f %f %f %f %f f_sm_h %f %f %f %f f_sm_r %f %f
      // %f %f, out: %f\n",
      //   v_idx[1], v_idx[0], a,b,c,d,e, h1,h2,h3,h4, r1,r2,r3,r4,
      //   mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));

      v[v_idx] = mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4);
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  // functor parameters
  DIM current_dim;
  SubArray<1, T, DeviceType> dist;
  SubArray<1, T, DeviceType> ratio;
  SubArray<D, T, DeviceType> coeff;
  SubArray<D, T, DeviceType> v;
};

template <DIM D, typename T, typename DeviceType>
class SingleDimensionMassTransKernel : public Kernel {
public:
  constexpr static DIM NumDim = D;
  using DataType = T;
  constexpr static std::string_view Name = "sdmtk";
  MGARDX_CONT
  SingleDimensionMassTransKernel(DIM current_dim,
                                 SubArray<1, T, DeviceType> dist,
                                 SubArray<1, T, DeviceType> ratio,
                                 SubArray<D, T, DeviceType> coeff,
                                 SubArray<D, T, DeviceType> v)
      : current_dim(current_dim), dist(dist), ratio(ratio), coeff(coeff), v(v) {
  }

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<SingleDimensionMassTransFunctor<D, T, R, C, F, DeviceType>>
  GenTask(int queue_idx) {

    using FunctorType =
        SingleDimensionMassTransFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(current_dim, dist, ratio, coeff, v);

    SIZE nr = 1, nc = 1, nf = 1;
    if (D >= 3)
      nr = v.shape(D - 3);
    if (D >= 2)
      nc = v.shape(D - 2);
    nf = v.shape(D - 1);

    SIZE total_thread_z = nr;
    SIZE total_thread_y = nc;
    SIZE total_thread_x = nf;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();

    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((double)total_thread_z / tbz);
    gridy = ceil((double)total_thread_y / tby);
    gridx = ceil((double)total_thread_x / tbx);

    for (int d = D - 4; d >= 0; d--) {
      gridx *= coeff.shape(d);
    }

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  // functor parameters
  DIM current_dim;
  SubArray<1, T, DeviceType> dist;
  SubArray<1, T, DeviceType> ratio;
  SubArray<D, T, DeviceType> coeff;
  SubArray<D, T, DeviceType> v;
};

} // namespace single_dimension

} // namespace data_refactoring

} // namespace mgard_x

#endif