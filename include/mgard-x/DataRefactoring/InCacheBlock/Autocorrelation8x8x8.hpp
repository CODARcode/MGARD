/*
 * Copyright 2023, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: Jan. 15, 2023
 */

#ifndef MGARD_X_AUTOCORRELATION_8x8x8_KERNEL_TEMPLATE
#define MGARD_X_AUTOCORRELATION_8x8x8_KERNEL_TEMPLATE

#include "../../RuntimeX/RuntimeX.h"

// #include "IndexTable3x3x3.hpp"
// #include "IndexTable5x5x5.hpp"
// #include "IndexTable8x8x8.hpp"

#define DECOMPOSE 0
#define RECOMPOSE 1

namespace mgard_x {

namespace data_refactoring {

namespace in_cache_block {

/*

v           x           y           z           c                total
8*8*8(512)  5*8*8(320)  5*5*8(200)  5*5*5(125)  0                1157
5*5*5(125)  3*5*5(75)   3*3*5(45)   3*3*3(27)   8*8*8-5*5*5(387) 659
3*3*3(27)   2*3*3(18)   2*2*3(12)   2*2*2(8)    8*8*8-3*3*3(485) 550

 v(512)  x(320)  y(200)  z(125)
c8(512)  v(125)  x( 75)  y( 45)  z(27)
c8(512) c5( 98)  x( 18)  y( 12)  z( 8)
c8(512) c5( 98)  c3(19)  c2( 8)
*/

template <DIM D, typename T, SIZE Z, SIZE Y, SIZE X, OPTION OP,
          typename DeviceType>
class Autocorrelation8x8x8Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Autocorrelation8x8x8Functor() {}
  MGARDX_CONT
  Autocorrelation8x8x8Functor(SubArray<D, T, DeviceType> v,
                              SubArray<D, T, DeviceType> autocorrelation_x,
                              SubArray<D, T, DeviceType> autocorrelation_y,
                              SubArray<D, T, DeviceType> autocorrelation_z,
                              int lag)
      : v(v), autocorrelation_x(autocorrelation_x),
        autocorrelation_y(autocorrelation_y),
        autocorrelation_z(autocorrelation_z), lag(lag) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void initialize_sm_8x8x8() {
    sm_v = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_x = sm_v + 8 * 8 * 8;
    sm_y = sm_x + 5 * 8 * 8;
    sm_z = sm_y + 5 * 5 * 8;
  }

  MGARDX_EXEC T Mean(T *data, SIZE ld, SIZE n) {
    T sum = 0;
    for (SIZE i = 0; i < n; i++) {
      sum += data[i * ld];
    }
    return sum / n;
  }

  MGARDX_EXEC T Autocorrelation(T *data, SIZE ld, SIZE n, int lag) {
    T mean = Mean(data, ld, n);
    T cov = 0;
    T sdv = 0;
    for (SIZE i = 0; i < n; i++) {
      T diff1 = data[i * ld] - mean;
      T diff2 = data[((i + lag) % n) * ld] - mean;
      cov += diff1 * diff2;
      sdv += diff1 * diff1;
    }
    return cov / sdv;
  }

  // Interpolation
  MGARDX_EXEC void Operation1() {
    initialize_sm_8x8x8();
    x = FunctorBase<DeviceType>::GetThreadIdX();
    y = FunctorBase<DeviceType>::GetThreadIdY();
    z = FunctorBase<DeviceType>::GetThreadIdZ();
    x_tb = FunctorBase<DeviceType>::GetBlockIdX();
    y_tb = FunctorBase<DeviceType>::GetBlockIdY();
    z_tb = FunctorBase<DeviceType>::GetBlockIdZ();
    x_gl = X * x_tb + x;
    y_gl = Y * y_tb + y;
    z_gl = Z * z_tb + z;

    tid = z * X * Y + y * X + x;
    bid = z_tb * FunctorBase<DeviceType>::GetGridDimX() *
              FunctorBase<DeviceType>::GetGridDimY() +
          y_tb * FunctorBase<DeviceType>::GetGridDimX() + x_tb;
    if (z == 0 && y == 0 && x == 0)
      sm_v[zero_const_offset] = (T)0;

    offset = get_idx(ld1, ld2, z, y, x);
    sm_v[offset] = 0.0;
    // Removing this check can speed up
    if (z_gl < v.shape(D - 3) && y_gl < v.shape(D - 2) &&
        x_gl < v.shape(D - 1)) {
      sm_v[offset] = *v(z_gl, y_gl, x_gl);
    }
    if (tid == 0) {

      T ac_x = 0, ac;
      // printf("ac_x: ");
      for (int z_id = 0; z_id < 8; z_id++) {
        for (int y_id = 0; y_id < 8; y_id++) {
          ac = Autocorrelation(&sm_v[y_id * 8 + z_id * 8 * 8], 1, 8, lag);
          // printf("%f ", ac);
          ac_x += ac;
        }
      }
      // printf("\n");
      ac_x /= 64;
      // printf("ac_x: %f\n", ac_x);

      T ac_y = 0;
      // printf("ac_x: ");
      for (int z_id = 0; z_id < 8; z_id++) {
        for (int x_id = 0; x_id < 8; x_id++) {
          ac = Autocorrelation(&sm_v[x_id + z_id * 8 * 8], 8, 8, lag);
          ac_y += ac;
          // printf("%f ", ac);
        }
      }
      // printf("\n");
      ac_y /= 64;
      // printf("ac_y: %f\n", ac_y);

      T ac_z = 0;
      // printf("ac_x: ");
      for (int y_id = 0; y_id < 8; y_id++) {
        for (int x_id = 0; x_id < 8; x_id++) {
          ac = Autocorrelation(&sm_v[x_id + y_id * 8], 8 * 8, 8, lag);
          ac_z += ac;
          // printf("%f ", ac);
        }
      }
      // printf("\n");
      ac_z /= 64;
      // printf("ac_z: %f\n", ac_z);

      *autocorrelation_x(z_tb, y_tb, x_tb) = ac_x;
      *autocorrelation_y(z_tb, y_tb, x_tb) = ac_y;
      *autocorrelation_z(z_tb, y_tb, x_tb) = ac_z;
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = (Z * Y * X) + Z * Y * (X / 2 + 1) +
                  Z * (Y / 2 + 1) * (X / 2 + 1) +
                  (Z / 2 + 1) * (Y / 2 + 1) * (X / 2 + 1) + 1;
    return size * sizeof(T);
  }

private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> autocorrelation_x;
  SubArray<D, T, DeviceType> autocorrelation_y;
  SubArray<D, T, DeviceType> autocorrelation_z;
  int lag;
  T *sm_v, *sm_x, *sm_y, *sm_z, *sm_c8, *sm_c5, *sm_c3, *sm_c2;
  int ld1 = X;
  int ld2 = Y;
  int z, y, x, z_tb, y_tb, x_tb, z_gl, y_gl, x_gl;
  int tid, bid, op_tid;
  T left, right, middle;
  int offset;
  int zero_const_offset = (Z * Y * X) + Z * Y * (X / 2 + 1) +
                          Z * (Y / 2 + 1) * (X / 2 + 1) +
                          (Z / 2 + 1) * (Y / 2 + 1) * (X / 2 + 1);
  // #ifdef MGARDX_COMPILE_CUDA
  // clock_t start, end;
  // #endif
};

template <DIM D, typename T, OPTION OP, typename DeviceType>
class Autocorrelation8x8x8Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lwpk";
  MGARDX_CONT
  Autocorrelation8x8x8Kernel(SubArray<D, T, DeviceType> v,
                             SubArray<D, T, DeviceType> autocorrelation_x,
                             SubArray<D, T, DeviceType> autocorrelation_y,
                             SubArray<D, T, DeviceType> autocorrelation_z,
                             int lag)
      : v(v), autocorrelation_x(autocorrelation_x),
        autocorrelation_y(autocorrelation_y),
        autocorrelation_z(autocorrelation_z), lag(lag) {}

  MGARDX_CONT Task<Autocorrelation8x8x8Functor<D, T, 8, 8, 8, OP, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType =
        Autocorrelation8x8x8Functor<D, T, 8, 8, 8, OP, DeviceType>;
    FunctorType functor(v, autocorrelation_x, autocorrelation_y,
                        autocorrelation_z, lag);

    SIZE total_thread_z = v.shape(D - 3);
    SIZE total_thread_y = v.shape(D - 2);
    SIZE total_thread_x = v.shape(D - 1);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 8;
    tby = 8;
    tbx = 8;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> autocorrelation_x;
  SubArray<D, T, DeviceType> autocorrelation_y;
  SubArray<D, T, DeviceType> autocorrelation_z;
  int lag;
};

} // namespace in_cache_block

} // namespace data_refactoring

} // namespace mgard_x

#endif