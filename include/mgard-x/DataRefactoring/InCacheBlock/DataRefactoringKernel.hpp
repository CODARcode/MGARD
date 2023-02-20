/*
 * Copyright 2023, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: Jan. 15, 2023
 */

#ifndef MGARD_X_IN_CACHE_BLOCK_DATA_REFACTORING_KERNEL_TEMPLATE
#define MGARD_X_IN_CACHE_BLOCK_DATA_REFACTORING_KERNEL_TEMPLATE

#include "../../RuntimeX/RuntimeX.h"

#include "../MultiDimension/Correction/LPKFunctor.h"

#include "../MultiDimension/Correction/IPKFunctor.h"

#include "IndexTable3x3x3.hpp"
#include "IndexTable5x5x5.hpp"
#include "IndexTable8x8x8.hpp"

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
class DataRefactoringFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DataRefactoringFunctor() {}
  MGARDX_CONT DataRefactoringFunctor(SubArray<D, T, DeviceType> v) : v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void initialize_sm_8x8x8() {
    sm_v = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_x = sm_v + 8 * 8 * 8;
    sm_y = sm_x + 5 * 8 * 8;
    sm_z = sm_y + 5 * 5 * 8;
  }

  MGARDX_EXEC void initialize_sm_5x5x5() {
    sm_c8 = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_v = sm_c8 + 8 * 8 * 8;
    sm_x = sm_v + 5 * 5 * 5;
    sm_y = sm_x + 5 * 5 * 3;
    sm_z = sm_y + 5 * 3 * 3;
  }

  MGARDX_EXEC void initialize_sm_3x3x3() {
    sm_c8 = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_c5 = sm_c8 + 8 * 8 * 8;
    sm_v = sm_c5 + 5 * 5 * 5;
    sm_x = sm_v + 3 * 3 * 3;
    sm_y = sm_x + 3 * 3 * 2;
    sm_z = sm_y + 3 * 2 * 2;
  }

  MGARDX_EXEC void initialize_sm_2x2x2() {
    sm_c8 = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_c5 = sm_c8 + 8 * 8 * 8;
    sm_c3 = sm_c5 + 5 * 5 * 5;
    sm_c2 = sm_c3 + 3 * 3 * 3;
  }

  MGARDX_EXEC void Operation1() {
    initialize_sm_8x8x8();
    x = FunctorBase<DeviceType>::GetThreadIdX();
    y = FunctorBase<DeviceType>::GetThreadIdY();
    z = FunctorBase<DeviceType>::GetThreadIdZ();
    x_gl = FunctorBase<DeviceType>::GetBlockDimX() *
               FunctorBase<DeviceType>::GetBlockIdX() +
           x;
    y_gl = FunctorBase<DeviceType>::GetBlockDimY() *
               FunctorBase<DeviceType>::GetBlockIdY() +
           y;
    z_gl = FunctorBase<DeviceType>::GetBlockDimZ() *
               FunctorBase<DeviceType>::GetBlockIdZ() +
           z;
    tid = z * X * Y + y * X + x;
    if (z == 0 && y == 0 && x == 0)
      sm_v[zero_const_offset] = (T)0;

    offset = get_idx(ld1, ld2, z, y, x);
    sm_v[offset] = *v(z_gl, y_gl, x_gl);
    op_tid = tid;
    if (tid < 225) {
      left = sm_v[Coeff1D_L_Offset_8x8x8(op_tid)];
      right = sm_v[Coeff1D_R_Offset_8x8x8(op_tid)];
      middle = sm_v[Coeff1D_M_Offset_8x8x8(op_tid)];
      // printf("l %f, r %f, m %f\n", left, right, middle);
      middle = middle - (left + right) * (T)0.5;
      sm_v[Coeff1D_M_Offset_8x8x8(op_tid)] = middle;
    } else if (tid >= 256 && tid < 256 + 135) {
      op_tid -= 256;
      T c00 = sm_v[Coeff2D_LL_Offset_8x8x8(op_tid)];
      T c02 = sm_v[Coeff2D_LR_Offset_8x8x8(op_tid)];
      T c20 = sm_v[Coeff2D_RL_Offset_8x8x8(op_tid)];
      T c22 = sm_v[Coeff2D_RR_Offset_8x8x8(op_tid)];
      T c11 = sm_v[Coeff2D_MM_Offset_8x8x8(op_tid)];
      c11 -= (c00 + c02 + c20 + c22) / 4;
      sm_v[Coeff2D_MM_Offset_8x8x8(op_tid)] = c11;
    } else if (tid >= 416 && tid < 416 + 27) {
      op_tid -= 416;
      T c000 = sm_v[Coeff3D_LLL_Offset_8x8x8(op_tid)];
      T c002 = sm_v[Coeff3D_LLR_Offset_8x8x8(op_tid)];
      T c020 = sm_v[Coeff3D_LRL_Offset_8x8x8(op_tid)];
      T c022 = sm_v[Coeff3D_LRR_Offset_8x8x8(op_tid)];
      T c200 = sm_v[Coeff3D_RLL_Offset_8x8x8(op_tid)];
      T c202 = sm_v[Coeff3D_RLR_Offset_8x8x8(op_tid)];
      T c220 = sm_v[Coeff3D_RRL_Offset_8x8x8(op_tid)];
      T c222 = sm_v[Coeff3D_RRR_Offset_8x8x8(op_tid)];
      T c111 = sm_v[Coeff3D_MMM_Offset_8x8x8(op_tid)];
      c111 -= (c000 + c002 + c020 + c022 + c200 + c202 + c220 + c222) / 8;
      sm_v[Coeff3D_MMM_Offset_8x8x8(op_tid)] = c111;
    }
  }

  MGARDX_EXEC void Operation2() {

    // __syncthreads();
    // if (tid == 0) {
    //   for (int i = 0; i < 8; i++) {
    //     printf("sm[i = %d]\n", i);
    //     for (int j = 0; j < 8; j++) {
    //       for (int k = 0; k < 8; k++) {
    //         printf("%.6f ", sm_v[get_idx(8, 8, i, j, k)]);
    //       }
    //       printf("\n");
    //     }
    //     printf("\n");
    //   }
    // }

    // __syncthreads();

    if (tid < 320) {
      int const *index = MassTrans_X_Offset_8x8x8(tid);
      float const *dist = MassTrans_X_Dist_8x8x8(tid % 5);
      T a = sm_v[index[0]];
      T b = sm_v[index[1]];
      T c = sm_v[index[2]];
      T d = sm_v[index[3]];
      T e = sm_v[index[4]];
      T h1 = dist[0], h2 = dist[0], h3 = dist[3], h4 = dist[3];

      // int dist_case = MassTrans_X_DistCase_8x8x8(tid);
      // if (dist_case == 0) {
      //   h1 = 0; h2 = 0; h3 = 1; h4 = 1;
      // } else if (dist_case == 1) {
      //   h1 = 1; h2 = 1; h3 = 1; h4 = 1;
      // } else if (dist_case == 2) {
      //   h1 = 1; h2 = 1; h3 = 0.5; h4 = 0.5;
      // } else if (dist_case == 3) {
      //   h1 = 0.5; h2 = 0.5; h3 = 0; h4 = 0;
      // }
      sm_x[index[5]] = mass_trans(a, b, c, d, e, h1, h2, h3, h4, (T)0.0, (T)0.0,
                                  (T)0.0, (T)0.0);
    }
    // __syncthreads();
    // if (tid == 5) {
    //   for (int i = 0; i < 8; i++) {
    //     printf("sm[i = %d]\n", i);
    //     for (int j = 0; j < 8; j++) {
    //       for (int k = 0; k < 5; k++) {
    //         printf("%.6f ", sm_x[get_idx(5, 8, i, j, k)]);
    //       }
    //       printf("\n");
    //     }
    //     printf("\n");
    //   }
    // }

    // __syncthreads();
  }

  MGARDX_EXEC void Operation3() {
    if (tid < 200) {
      int const *index = MassTrans_Y_Offset_8x8x8(tid);
      T a = sm_x[index[0]];
      T b = sm_x[index[1]];
      T c = sm_x[index[2]];
      T d = sm_x[index[3]];
      T e = sm_x[index[4]];
      float const *dist = MassTrans_Y_Dist_8x8x8(tid);
      T h1 = dist[0], h2 = dist[0], h3 = dist[3], h4 = dist[3];
      // int dist_case = MassTrans_Y_DistCase_8x8x8(tid);
      // if (dist_case == 0) {
      //   h1 = 0; h2 = 0; h3 = 1; h4 = 1;
      // } else if (dist_case == 1) {
      //   h1 = 1; h2 = 1; h3 = 1; h4 = 1;
      // } else if (dist_case == 2) {
      //   h1 = 1; h2 = 1; h3 = 0.5; h4 = 0.5;
      // } else if (dist_case == 3) {
      //   h1 = 0.5; h2 = 0.5; h3 = 0; h4 = 0;
      // }
      sm_y[index[5]] = mass_trans(a, b, c, d, e, h1, h2, h3, h4, (T)0.0, (T)0.0,
                                  (T)0.0, (T)0.0);
    }

    // __syncthreads();
    // if (tid == 0) {
    //   for (int i = 0; i < 8; i++) {
    //     printf("sm[i = %d]\n", i);
    //     for (int j = 0; j < 5; j++) {
    //       for (int k = 0; k < 5; k++) {
    //         printf("%.6f ", sm_y[get_idx(5, 5, i, j, k)]);
    //       }
    //       printf("\n");
    //     }
    //     printf("\n");
    //   }
    // }

    // __syncthreads();
  }

  MGARDX_EXEC void Operation4() {
    if (tid < 125) {
      int const *index = MassTrans_Z_Offset_8x8x8(tid);
      T a = sm_y[index[0]];
      T b = sm_y[index[1]];
      T c = sm_y[index[2]];
      T d = sm_y[index[3]];
      T e = sm_y[index[4]];
      float const *dist = MassTrans_Z_Dist_8x8x8(tid);
      T h1 = dist[0], h2 = dist[0], h3 = dist[3], h4 = dist[3];
      // int dist_case = MassTrans_Y_DistCase_8x8x8(tid);
      // if (dist_case == 0) {
      //   h1 = 0; h2 = 0; h3 = 1; h4 = 1;
      // } else if (dist_case == 1) {
      //   h1 = 1; h2 = 1; h3 = 1; h4 = 1;
      // } else if (dist_case == 2) {
      //   h1 = 1; h2 = 1; h3 = 0.5; h4 = 0.5;
      // } else if (dist_case == 3) {
      //   h1 = 0.5; h2 = 0.5; h3 = 0; h4 = 0;
      // }
      sm_z[index[5]] = mass_trans(a, b, c, d, e, h1, h2, h3, h4, (T)0.0, (T)0.0,
                                  (T)0.0, (T)0.0);
    }

    // __syncthreads();
    // if (tid == 0) {
    //   for (int i = 0; i < 5; i++) {
    //     printf("sm[i = %d]\n", i);
    //     for (int j = 0; j < 5; j++) {
    //       for (int k = 0; k < 5; k++) {
    //         printf("%.6f ", sm_z[get_idx(5, 5, i, j, k)]);
    //       }
    //       printf("\n");
    //     }
    //     printf("\n");
    //   }
    // }

    // __syncthreads();
  }

  MGARDX_EXEC void Operation5() {
    if (tid < 25) {
      int const *index = TriDiag_X_Offset_8x8x8(tid);
      T a = sm_z[index[0]];
      T b = sm_z[index[1]];
      T c = sm_z[index[2]];
      T d = sm_z[index[3]];
      T e = sm_z[index[4]];
      T am = 0;
      T bm = 0;

      a = tridiag_forward2((T)0.0, (T)0.000000, (T)1.000000, a);
      b = tridiag_forward2((T)a, (T)0.333333, (T)0.666667, b);
      c = tridiag_forward2((T)b, (T)0.333333, (T)1.166667, c);
      d = tridiag_forward2((T)c, (T)0.333333, (T)1.238095, d);
      e = tridiag_forward2((T)d, (T)0.166667, (T)0.910256, e);

      // tridiag_backward2((T)0.0, am, bm, e);
      // tridiag_backward2(e, am, bm, d);
      // tridiag_backward2(d, am, bm, c);
      // tridiag_backward2(c, am, bm, b);
      // tridiag_backward2(b, am, bm, a);
      sm_z[index[0]] = a;
      sm_z[index[1]] = b;
      sm_z[index[2]] = c;
      sm_z[index[3]] = d;
      sm_z[index[4]] = e;
    }

    // __syncthreads();
    // if (tid == 0) {
    //   for (int i = 0; i < 5; i++) {
    //     printf("sm[i = %d]\n", i);
    //     for (int j = 0; j < 5; j++) {
    //       for (int k = 0; k < 5; k++) {
    //         printf("%.6f ", sm_z[get_idx(5, 5, i, j, k)]);
    //       }
    //       printf("\n");
    //     }
    //     printf("\n");
    //   }
    // }

    // __syncthreads();
  }

  // MGARDX_EXEC void Operation6() {
  //   if (tid < 25) {
  //     int const *index = TriDiag_Y_Offset_8x8x8(tid);
  //     T a = sm_z[index[0]];
  //     T b = sm_z[index[1]];
  //     T c = sm_z[index[2]];
  //     T d = sm_z[index[3]];
  //     T e = sm_z[index[4]];
  //     T am = 0;
  //     T bm = 0;
  //     tridiag_forward2((T)0.0, am, bm, a);
  //     tridiag_forward2(a, am, bm, b);
  //     tridiag_forward2(b, am, bm, c);
  //     tridiag_forward2(c, am, bm, d);
  //     tridiag_forward2(d, am, bm, e);

  //     tridiag_backward2((T)0.0, am, bm, e);
  //     tridiag_backward2(e, am, bm, d);
  //     tridiag_backward2(d, am, bm, c);
  //     tridiag_backward2(c, am, bm, b);
  //     tridiag_backward2(b, am, bm, a);

  //     sm_z[index[0]] = a;
  //     sm_z[index[1]] = b;
  //     sm_z[index[2]] = c;
  //     sm_z[index[3]] = d;
  //     sm_z[index[4]] = e;
  //   }
  // }

  // MGARDX_EXEC void Operation7() {
  //   if (tid < 25) {
  //     int const *index = TriDiag_Z_Offset_8x8x8(tid);
  //     T a = sm_z[index[0]];
  //     T b = sm_z[index[1]];
  //     T c = sm_z[index[2]];
  //     T d = sm_z[index[3]];
  //     T e = sm_z[index[4]];
  //     T am = 0;
  //     T bm = 0;
  //     tridiag_forward2((T)0.0, am, bm, a);
  //     tridiag_forward2(a, am, bm, b);
  //     tridiag_forward2(b, am, bm, c);
  //     tridiag_forward2(c, am, bm, d);
  //     tridiag_forward2(d, am, bm, e);

  //     tridiag_backward2((T)0.0, am, bm, e);
  //     tridiag_backward2(e, am, bm, d);
  //     tridiag_backward2(d, am, bm, c);
  //     tridiag_backward2(c, am, bm, b);
  //     tridiag_backward2(b, am, bm, a);

  //     sm_z[index[0]] = a;
  //     sm_z[index[1]] = b;
  //     sm_z[index[2]] = c;
  //     sm_z[index[3]] = d;
  //     sm_z[index[4]] = e;
  //   }
  //   *v(z_gl, y_gl, x_gl) = sm_v[offset];
  // }

  // MGARDX_EXEC void Operation8() {
  //   T coarse, correction;
  //   if (tid < 125) {
  //     coarse = sm_v[Coarse_Offset_8x8x8(tid)];
  //     correction = sm_z[tid];
  //   }
  //   initialize_sm_3x3x3();
  //   if (tid < 125) {
  //     sm_v[tid] = coarse + correction;
  //   }
  // }

  // MGARDX_EXEC void Operation9() {
  //   op_tid = tid;
  //   if (tid < 54) {
  //     left = sm_v[Coeff1D_L_Offset_5x5x5(op_tid)];
  //     right = sm_v[Coeff1D_M_Offset_5x5x5(op_tid)];
  //     middle = sm_v[Coeff1D_R_Offset_5x5x5(op_tid)];
  //     middle = middle - (left + right) * (T)0.5;
  //     sm_v[Coeff1D_M_Offset_5x5x5(op_tid)] = middle;
  //   } else if (tid >= 64 && tid < 64 + 36) {
  //     op_tid -= 64;
  //     T c00 = sm_v[Coeff2D_LL_Offset_5x5x5(op_tid)];
  //     T c02 = sm_v[Coeff2D_LR_Offset_5x5x5(op_tid)];
  //     T c20 = sm_v[Coeff2D_RL_Offset_5x5x5(op_tid)];
  //     T c22 = sm_v[Coeff2D_RR_Offset_5x5x5(op_tid)];
  //     T c11 = sm_v[Coeff2D_MM_Offset_5x5x5(op_tid)];
  //     c11 -= (c00 + c02 + c20 + c22) / 4;
  //     sm_v[Coeff2D_MM_Offset_5x5x5(op_tid)] = c11;
  //   }

  //   else if (tid >= 128 && tid < 128 + 8) {
  //     op_tid -= 128;
  //     T c000 = sm_v[Coeff3D_LLL_Offset_5x5x5(op_tid)];
  //     T c002 = sm_v[Coeff3D_LLR_Offset_5x5x5(op_tid)];
  //     T c020 = sm_v[Coeff3D_LRL_Offset_5x5x5(op_tid)];
  //     T c022 = sm_v[Coeff3D_LRR_Offset_5x5x5(op_tid)];
  //     T c200 = sm_v[Coeff3D_RLL_Offset_5x5x5(op_tid)];
  //     T c202 = sm_v[Coeff3D_RLR_Offset_5x5x5(op_tid)];
  //     T c220 = sm_v[Coeff3D_RRL_Offset_5x5x5(op_tid)];
  //     T c222 = sm_v[Coeff3D_RRR_Offset_5x5x5(op_tid)];
  //     T c111 = sm_v[Coeff3D_MMM_Offset_5x5x5(op_tid)];
  //     c111 -= (c000 + c002 + c020 + c022 + c200 + c202 + c220 + c222) / 8;
  //     sm_v[Coeff3D_MMM_Offset_5x5x5(tid)] = c111;
  //   }
  // }

  // MGARDX_EXEC void Operation10() {
  //   if (tid < 75) {
  //     int const *index = MassTrans_X_Offset_5x5x5(tid);
  //     T a = sm_v[index[0]];
  //     T b = sm_v[index[1]];
  //     T c = sm_v[index[2]];
  //     T d = sm_v[index[3]];
  //     T e = sm_v[index[4]];
  //     const T h1 = 1.0 / 6.0;
  //     const T h2 = 1.0 / 6.0;
  //     const T h3 = 1.0 / 6.0;
  //     const T h4 = 1.0 / 6.0;
  //     const T r1 = 0.5;
  //     const T r4 = 0.5;
  //     T tb = a * h1 + b * (h1 + h2) + c * h2;
  //     T tc = b * h2 + c * (h2 + h3) + d * h3;
  //     T td = c * h3 + d * (h3 + h4) + e * h4;
  //     sm_x[index[5]] += tb * r1 + td * r4;
  //   }
  // }

  // MGARDX_EXEC void Operation11() {
  //   if (tid < 45) {
  //     int const *index = MassTrans_Y_Offset_5x5x5(tid);
  //     T a = sm_v[index[0]];
  //     T b = sm_v[index[1]];
  //     T c = sm_v[index[2]];
  //     T d = sm_v[index[3]];
  //     T e = sm_v[index[4]];
  //     const T h1 = 1.0 / 6.0;
  //     const T h2 = 1.0 / 6.0;
  //     const T h3 = 1.0 / 6.0;
  //     const T h4 = 1.0 / 6.0;
  //     const T r1 = 0.5;
  //     const T r4 = 0.5;
  //     T tb = a * h1 + b * (h1 + h2) + c * h2;
  //     T tc = b * h2 + c * (h2 + h3) + d * h3;
  //     T td = c * h3 + d * (h3 + h4) + e * h4;
  //     sm_x[index[5]] += tb * r1 + td * r4;
  //   }
  // }

  // MGARDX_EXEC void Operation12() {
  //   if (tid < 27) {
  //     int const *index = MassTrans_Z_Offset_5x5x5(tid);
  //     T a = sm_v[index[0]];
  //     T b = sm_v[index[1]];
  //     T c = sm_v[index[2]];
  //     T d = sm_v[index[3]];
  //     T e = sm_v[index[4]];
  //     const T h1 = 1.0 / 6.0;
  //     const T h2 = 1.0 / 6.0;
  //     const T h3 = 1.0 / 6.0;
  //     const T h4 = 1.0 / 6.0;
  //     const T r1 = 0.5;
  //     const T r4 = 0.5;
  //     T tb = a * h1 + b * (h1 + h2) + c * h2;
  //     T tc = b * h2 + c * (h2 + h3) + d * h3;
  //     T td = c * h3 + d * (h3 + h4) + e * h4;
  //     sm_x[index[5]] += tb * r1 + td * r4;
  //   }
  // }

  // MGARDX_EXEC void Operation13() {
  //   if (tid < 9) {
  //     int const *index = TriDiag_X_Offset_5x5x5(tid);
  //     T a = sm_z[index[0]];
  //     T b = sm_z[index[1]];
  //     T c = sm_z[index[2]];
  //     T d = sm_z[index[3]];
  //     T e = sm_z[index[4]];
  //     T am = 0;
  //     T bm = 0;
  //     tridiag_forward2((T)0.0, am, bm, a);
  //     tridiag_forward2(a, am, bm, b);
  //     tridiag_forward2(b, am, bm, c);
  //     tridiag_forward2(c, am, bm, d);
  //     tridiag_forward2(d, am, bm, e);

  //     tridiag_backward2((T)0.0, am, bm, e);
  //     tridiag_backward2(e, am, bm, d);
  //     tridiag_backward2(d, am, bm, c);
  //     tridiag_backward2(c, am, bm, b);
  //     tridiag_backward2(b, am, bm, a);

  //     sm_z[index[0]] = a;
  //     sm_z[index[1]] = b;
  //     sm_z[index[2]] = c;
  //     sm_z[index[3]] = d;
  //     sm_z[index[4]] = e;
  //   }
  //   if (tid < 9) {
  //     int const *index = TriDiag_Y_Offset_5x5x5(tid);
  //     T a = sm_z[index[0]];
  //     T b = sm_z[index[1]];
  //     T c = sm_z[index[2]];
  //     T d = sm_z[index[3]];
  //     T e = sm_z[index[4]];
  //     T am = 0;
  //     T bm = 0;
  //     tridiag_forward2((T)0.0, am, bm, a);
  //     tridiag_forward2(a, am, bm, b);
  //     tridiag_forward2(b, am, bm, c);
  //     tridiag_forward2(c, am, bm, d);
  //     tridiag_forward2(d, am, bm, e);

  //     tridiag_backward2((T)0.0, am, bm, e);
  //     tridiag_backward2(e, am, bm, d);
  //     tridiag_backward2(d, am, bm, c);
  //     tridiag_backward2(c, am, bm, b);
  //     tridiag_backward2(b, am, bm, a);

  //     sm_z[index[0]] = a;
  //     sm_z[index[1]] = b;
  //     sm_z[index[2]] = c;
  //     sm_z[index[3]] = d;
  //     sm_z[index[4]] = e;
  //   }
  //   if (tid < 9) {
  //     int const *index = TriDiag_Z_Offset_5x5x5(tid);
  //     T a = sm_z[index[0]];
  //     T b = sm_z[index[1]];
  //     T c = sm_z[index[2]];
  //     T d = sm_z[index[3]];
  //     T e = sm_z[index[4]];
  //     T am = 0;
  //     T bm = 0;
  //     tridiag_forward2((T)0.0, am, bm, a);
  //     tridiag_forward2(a, am, bm, b);
  //     tridiag_forward2(b, am, bm, c);
  //     tridiag_forward2(c, am, bm, d);
  //     tridiag_forward2(d, am, bm, e);

  //     tridiag_backward2((T)0.0, am, bm, e);
  //     tridiag_backward2(e, am, bm, d);
  //     tridiag_backward2(d, am, bm, c);
  //     tridiag_backward2(c, am, bm, b);
  //     tridiag_backward2(b, am, bm, a);

  //     sm_z[index[0]] = a;
  //     sm_z[index[1]] = b;
  //     sm_z[index[2]] = c;
  //     sm_z[index[3]] = d;
  //     sm_z[index[4]] = e;
  //   }
  // }

  // MGARDX_EXEC void Operation14() {
  //   T coarse, correction;
  //   if (tid < 27) {
  //     coarse = sm_v[Coarse_Offset_5x5x5(tid)];
  //     correction = sm_z[tid];
  //   }
  //   initialize_sm_3x3x3();
  //   if (tid < 27) {
  //     sm_v[tid] = coarse + correction;
  //   }
  // }

  // MGARDX_EXEC void Operation15() {
  //   op_tid = tid;
  //   if (tid < 12) {
  //     left = sm_v[Coeff1D_L_Offset_3x3x3(op_tid)];
  //     right = sm_v[Coeff1D_M_Offset_3x3x3(op_tid)];
  //     middle = sm_v[Coeff1D_R_Offset_3x3x3(op_tid)];
  //     middle = middle - (left + right) * (T)0.5;
  //     sm_v[Coeff1D_M_Offset_3x3x3(op_tid)] = middle;
  //   } else if (tid >= 32 && tid < 32 + 6) {
  //     op_tid -= 32;
  //     T c00 = sm_v[Coeff2D_LL_Offset_3x3x3(op_tid)];
  //     T c02 = sm_v[Coeff2D_LR_Offset_3x3x3(op_tid)];
  //     T c20 = sm_v[Coeff2D_RL_Offset_3x3x3(op_tid)];
  //     T c22 = sm_v[Coeff2D_RR_Offset_3x3x3(op_tid)];
  //     T c11 = sm_v[Coeff2D_MM_Offset_3x3x3(op_tid)];
  //     c11 -= (c00 + c02 + c20 + c22) / 4;
  //     sm_v[Coeff2D_MM_Offset_3x3x3(op_tid)] = c11;
  //   }

  //   else if (tid >= 64 && tid < 64 + 1) {
  //     op_tid -= 64;
  //     T c000 = sm_v[Coeff3D_LLL_Offset_3x3x3(op_tid)];
  //     T c002 = sm_v[Coeff3D_LLR_Offset_3x3x3(op_tid)];
  //     T c020 = sm_v[Coeff3D_LRL_Offset_3x3x3(op_tid)];
  //     T c022 = sm_v[Coeff3D_LRR_Offset_3x3x3(op_tid)];
  //     T c200 = sm_v[Coeff3D_RLL_Offset_3x3x3(op_tid)];
  //     T c202 = sm_v[Coeff3D_RLR_Offset_3x3x3(op_tid)];
  //     T c220 = sm_v[Coeff3D_RRL_Offset_3x3x3(op_tid)];
  //     T c222 = sm_v[Coeff3D_RRR_Offset_3x3x3(op_tid)];
  //     T c111 = sm_v[Coeff3D_MMM_Offset_3x3x3(op_tid)];
  //     c111 -= (c000 + c002 + c020 + c022 + c200 + c202 + c220 + c222) / 8;
  //     sm_v[Coeff3D_MMM_Offset_3x3x3(tid)] = c111;
  //   }
  // }

  // MGARDX_EXEC void Operation16() {
  //   if (tid < 18) {
  //     int const *index = MassTrans_X_Offset_3x3x3(tid);
  //     T a = sm_v[index[0]];
  //     T b = sm_v[index[1]];
  //     T c = sm_v[index[2]];
  //     T d = sm_v[index[3]];
  //     T e = sm_v[index[4]];
  //     const T h1 = 1.0 / 6.0;
  //     const T h2 = 1.0 / 6.0;
  //     const T h3 = 1.0 / 6.0;
  //     const T h4 = 1.0 / 6.0;
  //     const T r1 = 0.5;
  //     const T r4 = 0.5;
  //     T tb = a * h1 + b * (h1 + h2) + c * h2;
  //     T tc = b * h2 + c * (h2 + h3) + d * h3;
  //     T td = c * h3 + d * (h3 + h4) + e * h4;
  //     sm_x[index[5]] += tb * r1 + td * r4;
  //   }
  //   if (tid < 12) {
  //     int const *index = MassTrans_Y_Offset_3x3x3(tid);
  //     T a = sm_v[index[0]];
  //     T b = sm_v[index[1]];
  //     T c = sm_v[index[2]];
  //     T d = sm_v[index[3]];
  //     T e = sm_v[index[4]];
  //     const T h1 = 1.0 / 6.0;
  //     const T h2 = 1.0 / 6.0;
  //     const T h3 = 1.0 / 6.0;
  //     const T h4 = 1.0 / 6.0;
  //     const T r1 = 0.5;
  //     const T r4 = 0.5;
  //     T tb = a * h1 + b * (h1 + h2) + c * h2;
  //     T tc = b * h2 + c * (h2 + h3) + d * h3;
  //     T td = c * h3 + d * (h3 + h4) + e * h4;
  //     sm_x[index[5]] += tb * r1 + td * r4;
  //   }
  //   if (tid < 8) {
  //     int const *index = MassTrans_Z_Offset_3x3x3(tid);
  //     T a = sm_v[index[0]];
  //     T b = sm_v[index[1]];
  //     T c = sm_v[index[2]];
  //     T d = sm_v[index[3]];
  //     T e = sm_v[index[4]];
  //     const T h1 = 1.0 / 6.0;
  //     const T h2 = 1.0 / 6.0;
  //     const T h3 = 1.0 / 6.0;
  //     const T h4 = 1.0 / 6.0;
  //     const T r1 = 0.5;
  //     const T r4 = 0.5;
  //     T tb = a * h1 + b * (h1 + h2) + c * h2;
  //     T tc = b * h2 + c * (h2 + h3) + d * h3;
  //     T td = c * h3 + d * (h3 + h4) + e * h4;
  //     sm_x[index[5]] += tb * r1 + td * r4;
  //   }
  //   if (tid < 4) {
  //     int const *index = TriDiag_X_Offset_3x3x3(tid);
  //     T a = sm_z[index[0]];
  //     T b = sm_z[index[1]];
  //     T c = sm_z[index[2]];
  //     T d = sm_z[index[3]];
  //     T e = sm_z[index[4]];
  //     T am = 0;
  //     T bm = 0;
  //     tridiag_forward2((T)0.0, am, bm, a);
  //     tridiag_forward2(a, am, bm, b);
  //     tridiag_forward2(b, am, bm, c);
  //     tridiag_forward2(c, am, bm, d);
  //     tridiag_forward2(d, am, bm, e);

  //     tridiag_backward2((T)0.0, am, bm, e);
  //     tridiag_backward2(e, am, bm, d);
  //     tridiag_backward2(d, am, bm, c);
  //     tridiag_backward2(c, am, bm, b);
  //     tridiag_backward2(b, am, bm, a);

  //     sm_z[index[0]] = a;
  //     sm_z[index[1]] = b;
  //     sm_z[index[2]] = c;
  //     sm_z[index[3]] = d;
  //     sm_z[index[4]] = e;
  //   }
  //   if (tid < 4) {
  //     int const *index = TriDiag_Y_Offset_3x3x3(tid);
  //     T a = sm_z[index[0]];
  //     T b = sm_z[index[1]];
  //     T c = sm_z[index[2]];
  //     T d = sm_z[index[3]];
  //     T e = sm_z[index[4]];
  //     T am = 0;
  //     T bm = 0;
  //     tridiag_forward2((T)0.0, am, bm, a);
  //     tridiag_forward2(a, am, bm, b);
  //     tridiag_forward2(b, am, bm, c);
  //     tridiag_forward2(c, am, bm, d);
  //     tridiag_forward2(d, am, bm, e);

  //     tridiag_backward2((T)0.0, am, bm, e);
  //     tridiag_backward2(e, am, bm, d);
  //     tridiag_backward2(d, am, bm, c);
  //     tridiag_backward2(c, am, bm, b);
  //     tridiag_backward2(b, am, bm, a);

  //     sm_z[index[0]] = a;
  //     sm_z[index[1]] = b;
  //     sm_z[index[2]] = c;
  //     sm_z[index[3]] = d;
  //     sm_z[index[4]] = e;
  //   }
  //   if (tid < 4) {
  //     int const *index = TriDiag_Z_Offset_3x3x3(tid);
  //     T a = sm_z[index[0]];
  //     T b = sm_z[index[1]];
  //     T c = sm_z[index[2]];
  //     T d = sm_z[index[3]];
  //     T e = sm_z[index[4]];
  //     T am = 0;
  //     T bm = 0;
  //     tridiag_forward2((T)0.0, am, bm, a);
  //     tridiag_forward2(a, am, bm, b);
  //     tridiag_forward2(b, am, bm, c);
  //     tridiag_forward2(c, am, bm, d);
  //     tridiag_forward2(d, am, bm, e);

  //     tridiag_backward2((T)0.0, am, bm, e);
  //     tridiag_backward2(e, am, bm, d);
  //     tridiag_backward2(d, am, bm, c);
  //     tridiag_backward2(c, am, bm, b);
  //     tridiag_backward2(b, am, bm, a);

  //     sm_z[index[0]] = a;
  //     sm_z[index[1]] = b;
  //     sm_z[index[2]] = c;
  //     sm_z[index[3]] = d;
  //     sm_z[index[4]] = e;
  //   }
  // }

  // MGARDX_EXEC void Operation17() {
  //   T coarse, correction;
  //   if (tid < 8) {
  //     coarse = sm_v[Coarse_Offset_3x3x3(tid)];
  //     correction = sm_z[tid];
  //   }
  //   initialize_sm_2x2x2();
  //   if (tid < 8) {
  //     sm_c2[tid] = coarse + correction;
  //   }
  // }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = (Z * Y * X) + Z * Y * (X / 2 + 1) +
                  Z * (Y / 2 + 1) * (X / 2 + 1) +
                  (Z / 2 + 1) * (Y / 2 + 1) * (X / 2 + 1) + 1;
    return size * sizeof(T);
  }

private:
  SubArray<D, T, DeviceType> v;
  T *sm_v, *sm_x, *sm_y, *sm_z, *sm_c8, *sm_c5, *sm_c3, *sm_c2;
  int ld1 = X;
  int ld2 = Y;
  int z, y, x, z_gl, y_gl, x_gl;
  int tid, op_tid;
  T left, right, middle;
  int offset;
  int zero_const_offset = (Z * Y * X) + Z * Y * (X / 2 + 1) +
                          Z * (Y / 2 + 1) * (X / 2 + 1) +
                          (Z / 2 + 1) * (Y / 2 + 1) * (X / 2 + 1);
};

template <DIM D, typename T, SIZE Z, SIZE Y, SIZE X, OPTION OP,
          typename DeviceType>
class DataRefactoringKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lwpk";
  MGARDX_CONT
  DataRefactoringKernel(SubArray<D, T, DeviceType> v) : v(v) {}

  MGARDX_CONT Task<DataRefactoringFunctor<D, T, Z, Y, X, OP, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = DataRefactoringFunctor<D, T, Z, Y, X, OP, DeviceType>;
    FunctorType functor(v);

    SIZE total_thread_z = 1;
    SIZE total_thread_y = 1;
    SIZE total_thread_x = 1;
    if (D >= 3)
      total_thread_z = v.shape(D - 3);
    if (D >= 2)
      total_thread_y = v.shape(D - 2);
    total_thread_x = v.shape(D - 1);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = Z;
    tby = Y;
    tbx = X;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (int d = D - 4; d >= 0; d--) {
      gridx *= v.shape(d);
    }
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<D, T, DeviceType> v;
};

} // namespace in_cache_block

} // namespace data_refactoring

} // namespace mgard_x

#endif