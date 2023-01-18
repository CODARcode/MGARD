/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_SINGLE_DIMENSION_COEFFICIENT_KERNEL_TEMPLATE
#define MGARD_X_SINGLE_DIMENSION_COEFFICIENT_KERNEL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

#include "../../MultiDimension/Coefficient/GPKFunctor.h"

#define DECOMPOSE 0
#define RECOMPOSE 1

namespace mgard_x {

namespace data_refactoring {

namespace single_dimension {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, OPTION OP,
          typename DeviceType>
class SingleDimensionCoefficientFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT SingleDimensionCoefficientFunctor() {}
  MGARDX_CONT SingleDimensionCoefficientFunctor(
      DIM current_dim, SubArray<1, T, DeviceType> ratio,
      SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
      SubArray<D, T, DeviceType> coeff)
      : current_dim(current_dim), ratio(ratio), v(v), coarse(coarse),
        coeff(coeff) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE v_left_idx[D];
    SIZE v_middle_idx[D];
    SIZE v_right_idx[D];
    SIZE coeff_idx[D];
    SIZE corase_idx[D];

    SIZE firstD = div_roundup(coeff.shape(D - 1), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    coeff_idx[D - 1] =
        (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    bidx /= firstD;
    if (D >= 2)
      coeff_idx[D - 2] = FunctorBase<DeviceType>::GetBlockIdY() *
                             FunctorBase<DeviceType>::GetBlockDimY() +
                         FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      coeff_idx[D - 3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                             FunctorBase<DeviceType>::GetBlockDimZ() +
                         FunctorBase<DeviceType>::GetThreadIdZ();

    for (int d = D - 4; d >= 0; d--) {
      coeff_idx[d] = bidx % coeff.shape(d);
      bidx /= coeff.shape(d);
    }

    bool in_range = true;
    for (int d = D - 1; d >= 0; d--) {
      if (coeff_idx[d] >= coeff.shape(d))
        in_range = false;
    }

    if (in_range) {
      for (int d = D - 1; d >= 0; d--) {
        if (d != current_dim) {
          v_left_idx[d] = coeff_idx[d];
          v_middle_idx[d] = coeff_idx[d];
          v_right_idx[d] = coeff_idx[d];
          corase_idx[d] = coeff_idx[d];
        } else {
          v_left_idx[d] = coeff_idx[d] * 2;
          v_middle_idx[d] = coeff_idx[d] * 2 + 1;
          v_right_idx[d] = coeff_idx[d] * 2 + 2;
          corase_idx[d] = coeff_idx[d];
        }
      }

      if (OP == DECOMPOSE) {
        coeff[coeff_idx] =
            v[v_middle_idx] - lerp(v[v_left_idx], v[v_right_idx],
                                   *ratio(v_left_idx[current_dim]));
        // if (coeff_idx[current_dim] == 1) {
        //   printf("left: %f, right: %f, middle: %f, ratio: %f, coeff: %f\n",
        //         *v(v_left_idx), *v(v_right_idx), *v(v_middle_idx),
        //         *ratio(v_left_idx[current_dim]), *coeff(coeff_idx));
        // }
        coarse[corase_idx] = v[v_left_idx];
        if (coeff_idx[current_dim] == coeff.shape(current_dim) - 1) {
          corase_idx[current_dim]++;
          coarse[corase_idx] = v[v_right_idx];
          if (v.shape(current_dim) % 2 == 0) {
            v_right_idx[current_dim]++;
            corase_idx[current_dim]++;
            coarse[corase_idx] = v[v_right_idx];
          }
        }
      } else if (OP == RECOMPOSE) {
        T left = coarse[corase_idx];
        corase_idx[current_dim]++;
        T right = coarse[corase_idx];
        corase_idx[current_dim]--;

        v[v_left_idx] = left;
        if (coeff_idx[current_dim] == coeff.shape(current_dim) - 1) {
          corase_idx[current_dim]++;
          v[v_right_idx] = right;
          if (v.shape(current_dim) % 2 == 0) {
            v_right_idx[current_dim]++;
            corase_idx[current_dim]++;
            v[v_right_idx] = coarse[corase_idx];
            v_right_idx[current_dim]--;
            corase_idx[current_dim]--;
          }
          corase_idx[current_dim]--;
        }

        v[v_middle_idx] = coeff[coeff_idx] +
                          lerp(left, right, *ratio(v_left_idx[current_dim]));
        // if (coeff_idx[current_dim] == 1) {
        // printf("left: %f, right: %f, middle: %f (%f), ratio: %f, coeff:
        // %f\n",
        //       *v(v_left_idx), *v(v_right_idx), *v(v_middle_idx),
        //       *coeff(coeff_idx) + lerp(*v(v_left_idx), *v(v_right_idx),
        //       *ratio(v_left_idx[current_dim])),
        //       *ratio(v_left_idx[current_dim]), *coeff(coeff_idx));
        // }
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  // functor parameters
  DIM current_dim;
  SubArray<1, T, DeviceType> ratio;
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<D, T, DeviceType> coeff;
};

template <DIM D, typename T, OPTION OP, typename DeviceType>
class SingleDimensionCoefficientKernel : public Kernel {
public:
  constexpr static DIM NumDim = D;
  using DataType = T;
  constexpr static std::string_view Name = "sdck";
  MGARDX_CONT
  SingleDimensionCoefficientKernel(DIM current_dim,
                                   SubArray<1, T, DeviceType> ratio,
                                   SubArray<D, T, DeviceType> v,
                                   SubArray<D, T, DeviceType> coarse,
                                   SubArray<D, T, DeviceType> coeff)
      : current_dim(current_dim), ratio(ratio), v(v), coarse(coarse),
        coeff(coeff) {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT
      Task<SingleDimensionCoefficientFunctor<D, T, R, C, F, OP, DeviceType>>
      GenTask(int queue_idx) {

    using FunctorType =
        SingleDimensionCoefficientFunctor<D, T, R, C, F, OP, DeviceType>;
    FunctorType functor(current_dim, ratio, v, coarse, coeff);

    SIZE nr = 1, nc = 1, nf = 1;
    if (D >= 3)
      nr = coeff.shape(D - 3);
    if (D >= 2)
      nc = coeff.shape(D - 2);
    nf = coeff.shape(D - 1);

    SIZE total_thread_z = nr;
    SIZE total_thread_y = nc;
    SIZE total_thread_x = nf;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();

    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);

    for (DIM d = 3; d < D; d++) {
      gridx *= coeff.shape(D - (d + 1));
    }

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  // functor parameters
  DIM current_dim;
  SubArray<1, T, DeviceType> ratio;
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<D, T, DeviceType> coeff;
};

} // namespace single_dimension

} // namespace data_refactoring

} // namespace mgard_x

#endif
