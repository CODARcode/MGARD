/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_LINEAR_PROCESSING_KERNEL_TEMPLATE
#define MGARD_X_LINEAR_PROCESSING_KERNEL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"
#include "LPKFunctor.h"

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class Lpk1ReoFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT Lpk1ReoFunctor() {}
  MGARDX_CONT Lpk1ReoFunctor(
      SubArray<1, SIZE, DeviceType> shape,
      SubArray<1, SIZE, DeviceType> shape_c, DIM processed_n,
      SubArray<1, DIM, DeviceType> processed_dims, DIM curr_dim_r,
      DIM curr_dim_c, DIM curr_dim_f, SubArray<1, T, DeviceType> dist_f,
      SubArray<1, T, DeviceType> ratio_f, SubArray<D, T, DeviceType> v1,
      SubArray<D, T, DeviceType> v2, SubArray<D, T, DeviceType> w)
      : shape(shape), shape_c(shape_c), processed_n(processed_n),
        processed_dims(processed_dims), curr_dim_r(curr_dim_r),
        curr_dim_c(curr_dim_c), curr_dim_f(curr_dim_f), dist_f(dist_f),
        ratio_f(ratio_f), v1(v1), v2(v2), w(w) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                (FunctorBase<DeviceType>::GetBlockDimX() *
                 FunctorBase<DeviceType>::GetBlockDimY())) +
               (FunctorBase<DeviceType>::GetThreadIdY() *
                FunctorBase<DeviceType>::GetBlockDimX()) +
               FunctorBase<DeviceType>::GetThreadIdX();

    T *sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F * 2 + 3;
    ldsm2 = C;
    v_sm = sm;
    sm += ldsm1 * ldsm2 * R;

    dist_f_sm = sm;
    sm += ldsm1;
    ratio_f_sm = sm;
    sm += ldsm1;

    SIZE *sm_size = (SIZE *)sm;
    shape_sm = sm_size;
    sm_size += D;
    shape_c_sm = sm_size;
    sm_size += D;
    sm = (T *)sm_size;

    DIM *sm_dim = (DIM *)sm;
    processed_dims_sm = sm_dim;
    sm_dim += D;
    sm = (T *)sm_dim;

    if (threadId < D) {
      shape_sm[threadId] = *shape(threadId);
      shape_c_sm[threadId] = *shape_c(threadId);
    }
    if (threadId < processed_n) {
      processed_dims_sm[threadId] = *processed_dims(threadId);
    }
  }

  MGARDX_EXEC void Operation2() {
    // __syncthreads();

    for (DIM d = 0; d < D; d++)
      idx[d] = 0;

    nr = shape_sm[curr_dim_r];
    nc = shape_sm[curr_dim_c];
    nf = shape_sm[curr_dim_f];
    nf_c = shape_c_sm[curr_dim_f];

    zero_other = true;
    PADDING = (nf % 2 == 0);

    bidx = FunctorBase<DeviceType>::GetBlockIdX();
    if (nf_c % 2 == 1) {
      firstD = div_roundup(nf_c, FunctorBase<DeviceType>::GetBlockDimX());
    } else {
      firstD = div_roundup(nf_c, FunctorBase<DeviceType>::GetBlockDimX());
    }
    blockId = bidx % firstD;
    bidx /= firstD;

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
        SIZE t = shape_sm[d];
        for (DIM k = 0; k < processed_n; k++) {
          if (d == *processed_dims(k)) {
            t = shape_c_sm[d];
          }
        }
        idx[d] = bidx % t;
        bidx /= t;
        if (idx[d] >= shape_c_sm[d])
          zero_other = false;
      }
    }

    zero_r = shape_c_sm[curr_dim_r];
    zero_c = shape_c_sm[curr_dim_c];
    zero_f = shape_c_sm[curr_dim_f];

    if (D < 3) {
      nr = 1;
      zero_r = 1;
    }
    if (D < 2) {
      nc = 1;
      zero_c = 1;
    }

    v1.offset(idx);
    v2.offset(idx);
    w.offset(idx);

    // if (debug2) {
    //   printf("idx: %d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
    //   printf("other_offset_v: %llu\n", other_offset_v);
    //   printf("other_offset_w: %llu\n", other_offset_w);
    // }
    r_gl = FunctorBase<DeviceType>::GetBlockIdZ() *
               FunctorBase<DeviceType>::GetBlockDimZ() +
           FunctorBase<DeviceType>::GetThreadIdZ();
    c_gl = FunctorBase<DeviceType>::GetBlockIdY() *
               FunctorBase<DeviceType>::GetBlockDimY() +
           FunctorBase<DeviceType>::GetThreadIdY();
    f_gl = blockId * FunctorBase<DeviceType>::GetBlockDimX() +
           FunctorBase<DeviceType>::GetThreadIdX();

    r_sm = FunctorBase<DeviceType>::GetThreadIdZ();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();

    actual_F = F;
    if (nf_c - blockId * FunctorBase<DeviceType>::GetBlockDimX() < F) {
      actual_F = nf_c - blockId * FunctorBase<DeviceType>::GetBlockDimX();
    }

    // if (nf_c % 2 == 1){
    //   if(nf_c-1 - blockId * FunctorBase<DeviceType>::GetBlockDimX() < F) {
    //   actual_F = nf_c - 1 - blockId *
    //   FunctorBase<DeviceType>::GetBlockDimX(); }
    // } else {
    //   if(nf_c - blockId * FunctorBase<DeviceType>::GetBlockDimX() < F) {
    //   actual_F = nf_c - blockId * FunctorBase<DeviceType>::GetBlockDimX(); }
    // }

    // if (debug) printf("actual_F %d\n", actual_F);

    if (r_gl < nr && c_gl < nc && f_gl < nf_c) {
      if (zero_other && r_gl < zero_r && c_gl < zero_c && f_gl < zero_f) {
        // if (debug) printf("load left vsm[%d]: 0.0\n", f_sm * 2 + 2);
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 2)] = 0.0;
      } else {
        // if (debug) printf("load left vsm[%d]<-dv1[%d, %d, %d]: %f\n", f_sm *
        // 2
        // + 2, r_gl, c_gl, f_gl, *v1( r_gl, c_gl, f_gl));
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 2)] =
            *v1(r_gl, c_gl, f_gl);
      }

      if (f_sm == actual_F - 1) {
        if (zero_other && r_gl < zero_r && c_gl < zero_c && f_gl < zero_f) {
          // if (debug) printf("load left+1 vsm[%d]: 0.0\n", actual_F * 2 + 2);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 2)] = 0.0;
        } else {
          if (f_gl + 1 < nf_c) {
            // if (debug) printf("load left+1 vsm[%d]: %f\n", actual_F * 2 + 2,
            // *v1( r_gl, c_gl, f_gl + 1));
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 2)] =
                *v1(r_gl, c_gl, f_gl + 1);
          } else {
            // if (debug) printf("load left+1 vsm[%d]: 0.0\n", actual_F * 2 +
            // 2);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 2)] = 0.0;
          }
        }
      }

      if (f_sm == 0) {
        // left
        if (zero_other && r_gl < zero_r && c_gl < zero_c && f_gl < zero_f) {
          // coarse (-1)
          // if (debug) printf("load left-1 vsm[0]: 0.0\n");
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 0.0;
        } else {
          if (f_gl >= 1) {
            // other (-1)
            // if (debug) printf("load left-1 vsm[0]: %f\n", dv1[get_idx(lddv11,
            // lddv12, r_gl, c_gl, f_gl-1)]);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
                *v1(r_gl, c_gl, f_gl - 1);
          } else {
            // other (-1)
            // if (debug) printf("load left-1 vsm[0]: 0.0\n");
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 0.0;
          }
        }
      }

      // right
      if (!PADDING) { // other = nf_c - 1
        if (nf_c % 2 != 0) {
          if (f_gl >= 1 &&
              f_gl < nf_c) { // shift for better memory access pattern
            // if (debug) printf("load right vsm[%d]: %f <- %d %d %d\n", f_sm *
            // 2
            // + 1, *v2( r_gl, c_gl, f_gl - 1), r_gl,
            // c_gl, f_gl - 1);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)] =
                *v2(r_gl, c_gl, f_gl - 1);
          } else {
            // if (debug) printf("load right vsm[%d]: 0\n", f_sm * 2 + 1);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)] = 0.0;
          }
        } else { // nf_c % 2 == 0, do not shift
          if (f_gl < nf_c - 1) {
            // if (debug) printf("load right vsm[%d]: %f <- %d %d %d\n", f_sm *
            // 2
            // + 3, *v2( r_gl, c_gl, f_gl), r_gl, c_gl,
            // f_gl);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 3)] =
                *v2(r_gl, c_gl, f_gl);
          } else {
            // if (debug) printf("load right vsm[%d]: 0\n", f_sm * 2 + 3);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 3)] = 0.0;
          }
        }
      } else { // PADDING other = nf_c - 2
        if (nf_c % 2 != 0) {
          if (f_gl >= 1 &&
              f_gl < nf_c - 1) { // shift for better memory access pattern
            // if (debug) printf("load right vsm[%d]: %f <- %d %d %d\n", f_sm *
            // 2
            // + 1, *v2( r_gl, c_gl, f_gl - 1), r_gl,
            // c_gl, f_gl - 1);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)] =
                *v2(r_gl, c_gl, f_gl - 1);
          } else {
            // if (debug) printf("load right vsm[%d]: 0\n", f_sm * 2 + 1);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)] = 0.0;
          }
        } else {                 // nf_c % 2 == 0
          if (f_gl < nf_c - 2) { // do not shift
            // if (debug) printf("load right vsm[%d]: %f <- %d %d %d\n", f_sm *
            // 2
            // + 3, *v2( r_gl, c_gl, f_gl), r_gl, c_gl,
            // f_gl);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 3)] =
                *v2(r_gl, c_gl, f_gl);
          } else {
            // if (debug) printf("load right vsm[%d]: 0\n", f_sm * 2 + 3);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 3)] = 0.0;
          }
        }
      }

      if (f_sm == actual_F - 1) {
        // right (+1)
        if (!PADDING) {
          if (nf_c % 2 != 0) {
            if (f_gl < nf_c - 1) {
              // if (debug) printf("load right+1 vsm[%d]: %f <- %d %d %d\n",
              // actual_F * 2 + 1, *v2( r_gl, c_gl, f_gl),
              // r_gl, c_gl, f_gl);
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 1)] =
                  *v2(r_gl, c_gl, f_gl);
            } else {
              // if (debug) printf("load right+1 vsm[%d]: 0.0\n", actual_F * 2 +
              // 1);
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 1)] = 0.0;
            }
          } else { // nf_c % 2 == 0
            if (f_gl >= actual_F) {
              // if (debug) printf("load right-1 vsm[1]: %f <- %d %d %d\n",
              // *v2( r_gl, c_gl, f_gl - actual_F), r_gl,
              // c_gl, f_gl - actual_F);
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 1)] =
                  *v2(r_gl, c_gl, f_gl - actual_F);
            } else {
              // if (debug) printf("load right-1 vsm[1]: 0.0\n");
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 1)] = 0.0;
            }
          }
        } else {
          if (nf_c % 2 != 0) {
            if (f_gl < nf_c - 2) {
              // if (debug) printf("actual_F(%d), load right+1 vsm[%d]: %f <- %d
              // %d %d\n", actual_F, actual_F * 2 + 1, *v2( r_gl, c_gl, f_gl),
              // r_gl, c_gl, f_gl);
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 1)] =
                  *v2(r_gl, c_gl, f_gl);
            } else {
              // if (debug) printf("load right+1 vsm[%d]: 0.0\n", actual_F * 2 +
              // 1);
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 1)] = 0.0;
            }
          } else { // nf_c % 2 == 0
            if (f_gl >= actual_F && f_gl - actual_F < nf_c - 2) {
              // if (debug) printf("load right-1 vsm[1]: %f <- %d %d %d\n",
              // *v2( r_gl, c_gl, f_gl - actual_F), r_gl,
              // c_gl, f_gl - actual_F);
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 1)] =
                  *v2(r_gl, c_gl, f_gl - actual_F);
            } else {
              // if (debug) printf("load right-1 vsm[1]: 0.0\n");
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 1)] = 0.0;
            }
          }
        }
      }
    }

    // if (debug)  printf("actual_F: %d\n", actual_F);
    if (r_sm == 0 && c_sm == 0 && f_sm < actual_F) {
      // if (debug) printf("blockId * F * 2 + f_sm = %d\n", blockId * F * 2 +
      // f_sm);
      if (blockId * F * 2 + f_sm < nf) { // padding: num of dist == nf,
                                         // non-padding: non of dist == nf - 1
        // if (debug) printf("load dist/ratio1[%d]: %f <- %d\n", 2 + f_sm,
        // *dist_f(blockId * F * 2 + f_sm), blockId * F * 2 + f_sm);
        dist_f_sm[2 + f_sm] = *dist_f(blockId * F * 2 + f_sm);
        ratio_f_sm[2 + f_sm] = *ratio_f(blockId * F * 2 + f_sm);
      } else {
        // if (debug) printf("load dist/ratio1[%d]: 0.0\n", 2 + f_sm);
        dist_f_sm[2 + f_sm] = 0.0;
        ratio_f_sm[2 + f_sm] = 0.0;
      }

      if (blockId * F * 2 + actual_F + f_sm < nf) {
        // if (debug) printf("load dist/ratio2[%d]: %f <- %d\n", 2 + actual_F +
        // f_sm, *dist_f(blockId * F * 2 + actual_F + f_sm), blockId * F * 2 +
        // actual_F + f_sm);
        dist_f_sm[2 + actual_F + f_sm] =
            *dist_f(blockId * F * 2 + actual_F + f_sm);
        ratio_f_sm[2 + actual_F + f_sm] =
            *ratio_f(blockId * F * 2 + actual_F + f_sm);
      } else {
        // if (debug) printf("load dist/ratio2[%d]: 0.0\n", 2 + actual_F +
        // f_sm);
        dist_f_sm[2 + actual_F + f_sm] = 0.0;
        ratio_f_sm[2 + actual_F + f_sm] = 0.0;
      }
    }

    if (blockId > 0) {
      if (f_sm < 2) {
        // dist_f_sm[f_sm] = *dist_f(f_gl - 2);
        // ratio_f_sm[f_sm] = *ratio_f(f_gl - 2);
        // if (debug) printf("load dist/ratio-1[%d]: %f <- %d\n", f_sm,
        // *dist_f(blockId * F * 2 + f_sm - 2), blockId * F * 2 + f_sm - 2);
        dist_f_sm[f_sm] = *dist_f(blockId * F * 2 + f_sm - 2);
        ratio_f_sm[f_sm] = *ratio_f(blockId * F * 2 + f_sm - 2);
      }
    } else {
      if (f_sm < 2) {
        // if (debug) printf("load dist/ratio-1[%d]: 0.0 <- %d\n", f_sm);
        dist_f_sm[f_sm] = 0.0;
        ratio_f_sm[f_sm] = 0.0;
      }
    }
  }

  MGARDX_EXEC void Operation3() {

    // __syncthreads();

    if (r_gl < nr && c_gl < nc && f_gl < nf_c) {
      T h1 = dist_f_sm[f_sm * 2];
      T h2 = dist_f_sm[f_sm * 2 + 1];
      T h3 = dist_f_sm[f_sm * 2 + 2];
      T h4 = dist_f_sm[f_sm * 2 + 3];
      T r1 = ratio_f_sm[f_sm * 2];
      T r2 = ratio_f_sm[f_sm * 2 + 1];
      T r3 = ratio_f_sm[f_sm * 2 + 2];
      T r4 = 1 - r3;
      T a = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2)];
      T b = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)];
      T c = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 2)];
      T d = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 3)];
      T e = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 4)];

      // bool debug = false;
      // if (idx[3] == 0) debug = false;
      // if (debug) {
      //   printf("f_sm(%d) %f %f %f %f %f f_sm_h %f %f %f %f f_sm_r %f %f %f
      //   %f, out: %f\n",f_sm, a,b,c,d,e, h1,h2,h3,h4,r1,r2,r3,r4,
      //   mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));
      // }

      // T tb = a * h1/6 + b * (h1+h2)/3 + c * h2/6;
      // T tc = b * h2/6 + c * (h2+h3)/3 + d * h3/6;
      // T td = c * h3/6 + d * (h3+h4)/3 + e * h4/6;

      // if (debug) printf("f_sm(%d) tb tc td tc: %f %f %f %f\n", f_sm, tb, tc,
      // td, tc+tb * r1 + td * r4);

      // tc += tb * r1 + td * r4;

      *w(r_gl, c_gl, f_gl) =
          mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4);

      // if (debug) printf("store[%d %d %d] %f \n", r_gl, c_gl, f_gl,
      //           mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));

      // printf("test block %d F %d nf %d\n", blockId, F, nf);
      // if (f_gl+1 == nf_c-1) {

      //     // T te = h4 * d + 2 * h4 * e;
      //     //printf("f_sm(%d) mm-e: %f\n", f_sm, te);
      //     // te += td * r3;
      //     *w( r_gl, c_gl, f_gl+1) =
      //       mass_trans(c, d, e, (T)0.0, (T)0.0, h1, h2, (T)0.0, (T)0.0, r1,
      //       r2, (T)0.0, (T)0.0);
      // }
    }

    v1.reset_offset();
    v2.reset_offset();
    w.reset_offset();
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size = (R * C * (F * 2 + 3) + (F * 2 + 3) * 2) * sizeof(T);
    size += (D * 4) * sizeof(SIZE);
    size += (D * 1) * sizeof(DIM);
    return size;
  }

private:
  // functor parameters
  SubArray<1, SIZE, DeviceType> shape;
  SubArray<1, SIZE, DeviceType> shape_c;
  DIM processed_n;
  SubArray<1, DIM, DeviceType> processed_dims;
  DIM curr_dim_r, curr_dim_c, curr_dim_f;
  SubArray<1, T, DeviceType> dist_f;
  SubArray<1, T, DeviceType> ratio_f;
  SubArray<D, T, DeviceType> v1, v2, w;

  // thread local variables

  LENGTH threadId;

  T *v_sm;
  SIZE ldsm1, ldsm2;
  T *dist_f_sm;
  T *ratio_f_sm;

  SIZE *shape_sm;
  SIZE *shape_c_sm;
  DIM *processed_dims_sm;

  SIZE idx[D];
  SIZE nr, nc, nf, nf_c;
  bool zero_other;
  bool PADDING;

  SIZE bidx;
  SIZE firstD;
  SIZE blockId;

  SIZE zero_r;
  SIZE zero_c;
  SIZE zero_f;

  SIZE r_gl;
  SIZE c_gl;
  SIZE f_gl;

  SIZE r_sm;
  SIZE c_sm;
  SIZE f_sm;

  SIZE actual_F;
};

template <DIM D, typename T, typename DeviceType>
class Lpk1Reo : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  Lpk1Reo() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Lpk1ReoFunctor<D, T, R, C, F, DeviceType>>
  GenTask(SubArray<1, SIZE, DeviceType> shape,
          SubArray<1, SIZE, DeviceType> shape_c, DIM processed_n,
          SubArray<1, DIM, DeviceType> processed_dims, DIM curr_dim_r,
          DIM curr_dim_c, DIM curr_dim_f, SubArray<1, T, DeviceType> dist_f,
          SubArray<1, T, DeviceType> ratio_f, SubArray<D, T, DeviceType> v1,
          SubArray<D, T, DeviceType> v2, SubArray<D, T, DeviceType> w,
          int queue_idx) {
    using FunctorType = Lpk1ReoFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(shape, shape_c, processed_n, processed_dims, curr_dim_r,
                        curr_dim_c, curr_dim_f, dist_f, ratio_f, v1, v2, w);

    SIZE nr = shape.dataHost()[curr_dim_r];
    SIZE nc = shape.dataHost()[curr_dim_c];
    SIZE nf = shape.dataHost()[curr_dim_f];
    SIZE nf_c = shape_c.dataHost()[curr_dim_f];

    SIZE total_thread_z = nr;
    SIZE total_thread_y = nc;
    SIZE total_thread_x = nf_c;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
        SIZE t = shape.dataHost()[d];
        for (DIM k = 0; k < processed_n; k++) {
          if (d == processed_dims.dataHost()[k]) {
            t = shape_c.dataHost()[d];
          }
        }
        gridx *= t;
      }
    }

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "Lpk1Reo");
  }

  MGARDX_CONT
  void Execute(SubArray<1, SIZE, DeviceType> shape,
               SubArray<1, SIZE, DeviceType> shape_c, DIM processed_n,
               SubArray<1, DIM, DeviceType> processed_dims, DIM curr_dim_r,
               DIM curr_dim_c, DIM curr_dim_f,
               SubArray<1, T, DeviceType> dist_f,
               SubArray<1, T, DeviceType> ratio_f,
               SubArray<D, T, DeviceType> v1, SubArray<D, T, DeviceType> v2,
               SubArray<D, T, DeviceType> w, int queue_idx) {
    int range_l = std::min(6, (int)std::log2(shape.dataHost()[curr_dim_f]) - 1);
    int arch = DeviceRuntime<DeviceType>::GetArchitectureGeneration();
    int prec = TypeToIdx<T>();
    // int config =
    // AutoTuner<DeviceType>::autoTuningTable.auto_tuning_mr1[arch][prec][range_l];
    int config = AutoTuner<DeviceType>::autoTuningTable.lpk1_nd[prec][range_l];

    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;

#define LPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = LPK_CONFIG[D - 1][CONFIG][0];                                \
    const int C = LPK_CONFIG[D - 1][CONFIG][1];                                \
    const int F = LPK_CONFIG[D - 1][CONFIG][2];                                \
    using FunctorType = Lpk1ReoFunctor<D, T, R, C, F, DeviceType>;             \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<R, C, F>(                                          \
        shape, shape_c, processed_n, processed_dims, curr_dim_r, curr_dim_c,   \
        curr_dim_f, dist_f, ratio_f, v1, v2, w, queue_idx);                    \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ExecutionReturn ret = adapter.Execute(task);                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (min_time > ret.execution_time) {                                     \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    LPK(0)
    LPK(1)
    LPK(2)
    LPK(3)
    LPK(4)
    LPK(5)
    LPK(6)
#undef LPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("lpk1_nd", prec, range_l, min_config);
    }
  }
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class Lpk2ReoFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT Lpk2ReoFunctor() {}
  MGARDX_CONT Lpk2ReoFunctor(
      SubArray<1, SIZE, DeviceType> shape,
      SubArray<1, SIZE, DeviceType> shape_c, DIM processed_n,
      SubArray<1, DIM, DeviceType> processed_dims, DIM curr_dim_r,
      DIM curr_dim_c, DIM curr_dim_f, SubArray<1, T, DeviceType> dist_c,
      SubArray<1, T, DeviceType> ratio_c, SubArray<D, T, DeviceType> v1,
      SubArray<D, T, DeviceType> v2, SubArray<D, T, DeviceType> w)
      : shape(shape), shape_c(shape_c), processed_n(processed_n),
        processed_dims(processed_dims), curr_dim_r(curr_dim_r),
        curr_dim_c(curr_dim_c), curr_dim_f(curr_dim_f), dist_c(dist_c),
        ratio_c(ratio_c), v1(v1), v2(v2), w(w) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                (FunctorBase<DeviceType>::GetBlockDimX() *
                 FunctorBase<DeviceType>::GetBlockDimY())) +
               (FunctorBase<DeviceType>::GetThreadIdY() *
                FunctorBase<DeviceType>::GetBlockDimX()) +
               FunctorBase<DeviceType>::GetThreadIdX();

    T *sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F;
    ldsm2 = C * 2 + 3;
    v_sm = sm;
    sm += ldsm1 * ldsm2 * R;

    dist_c_sm = sm;
    sm += ldsm2;
    ratio_c_sm = sm;
    sm += ldsm2;

    SIZE *sm_size = (SIZE *)sm;
    shape_sm = sm_size;
    sm_size += D;
    shape_c_sm = sm_size;
    sm_size += D;
    sm = (T *)sm_size;
    DIM *sm_dim = (DIM *)sm;
    processed_dims_sm = sm_dim;
    sm_dim += D;
    sm = (T *)sm_dim;

    if (threadId < D) {
      shape_sm[threadId] = *shape(threadId);
      shape_c_sm[threadId] = *shape_c(threadId);
    }
    if (threadId < processed_n) {
      processed_dims_sm[threadId] = *processed_dims(threadId);
    }
  }

  MGARDX_EXEC void Operation2() {
    // __syncthreads();

    for (DIM d = 0; d < D; d++)
      idx[d] = 0;

    nr = shape_sm[curr_dim_r];
    nc = shape_sm[curr_dim_c];
    nf_c = shape_c_sm[curr_dim_f];
    nc_c = shape_c_sm[curr_dim_c];
    PADDING = (nc % 2 == 0);

    if (D < 3) {
      nr = 1;
    }

    bidx = FunctorBase<DeviceType>::GetBlockIdX();
    firstD = div_roundup(nf_c, FunctorBase<DeviceType>::GetBlockDimX());
    blockId_f = bidx % firstD;
    bidx /= firstD;

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
        SIZE t = shape_sm[d];
        for (DIM k = 0; k < processed_n; k++) {
          if (d == processed_dims_sm[k]) {
            t = shape_c_sm[d];
          }
        }
        idx[d] = bidx % t;
        bidx /= t;
      }
    }

    v1.offset(idx);
    v2.offset(idx);
    w.offset(idx);

    // LENGTH other_offset_v = get_idx<D>(ldvs_sm, idx);
    // LENGTH other_offset_w = get_idx<D>(ldws_sm, idx);

    // dv1 = dv1 + other_offset_v;
    // dv2 = dv2 + other_offset_v;
    // dw = dw + other_offset_w;

    // if (debug2) {
    //   printf("idx: %d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
    //   printf("other_offset_v: %llu\n", other_offset_v);
    //   printf("other_offset_w: %llu\n", other_offset_w);
    // }

    r_gl = FunctorBase<DeviceType>::GetBlockIdZ() *
               FunctorBase<DeviceType>::GetBlockDimZ() +
           FunctorBase<DeviceType>::GetThreadIdZ();
    c_gl = FunctorBase<DeviceType>::GetBlockIdY() *
               FunctorBase<DeviceType>::GetBlockDimY() +
           FunctorBase<DeviceType>::GetThreadIdY();
    f_gl = blockId_f * FunctorBase<DeviceType>::GetBlockDimX() +
           FunctorBase<DeviceType>::GetThreadIdX();

    blockId = FunctorBase<DeviceType>::GetBlockIdY();

    r_sm = FunctorBase<DeviceType>::GetThreadIdZ();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();

    actual_C = C;
    if (nc_c - FunctorBase<DeviceType>::GetBlockIdY() *
                   FunctorBase<DeviceType>::GetBlockDimY() <
        C) {
      actual_C = nc_c - FunctorBase<DeviceType>::GetBlockIdY() *
                            FunctorBase<DeviceType>::GetBlockDimY();
    }

    // if (nc_c % 2 == 1){
    //   if(nc_c-1 - FunctorBase<DeviceType>::GetBlockIdY() *
    //   FunctorBase<DeviceType>::GetBlockDimY() < C) { actual_C = nc_c - 1 -
    //   FunctorBase<DeviceType>::GetBlockIdY() *
    //   FunctorBase<DeviceType>::GetBlockDimY(); }
    // } else {
    //   if(nc_c - FunctorBase<DeviceType>::GetBlockIdY() *
    //   FunctorBase<DeviceType>::GetBlockDimY() < C) { actual_C = nc_c -
    //   FunctorBase<DeviceType>::GetBlockIdY() *
    //   FunctorBase<DeviceType>::GetBlockDimY(); }
    // }

    // bool debug = false;
    // if (idx[3] == 0 && r_gl == 0 ) debug = false;

    // if (debug) printf("actual_C %d\n", actual_C);

    if (r_gl < nr && c_gl < nc_c && f_gl < nf_c) {
      // if (debug) printf("load up vsm[%d]: %f <- %d %d %d\n", c_sm * 2 + 2,
      // *v1( r_gl, c_gl, f_gl), r_gl, c_gl, f_gl);
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 2, f_sm)] =
          *v1(r_gl, c_gl, f_gl);

      if (c_sm == actual_C - 1) {
        if (c_gl + 1 < nc_c) {
          // if (debug) printf("load up+1 vsm[%d]: %f <- %d %d %d\n", actual_C *
          // 2
          // + 2, dv1[get_idx(lddv11, lddv12, r_gl, blockId * C + actual_C,
          // f_gl)], r_gl, blockId * C + actual_C, f_gl);
          // c_gl+1 == blockId * C + C
          v_sm[get_idx(ldsm1, ldsm2, r_sm, actual_C * 2 + 2, f_sm)] =
              *v1(r_gl, c_gl + 1, f_gl);
        } else {
          // if (debug) printf("load up+1 vsm[%d]: 0.0\n", actual_C * 2 + 2);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, actual_C * 2 + 2, f_sm)] = 0.0;
        }
      }

      if (c_sm == 0) {
        if (c_gl >= 1) {
          // if (debug) printf("load up-1 vsm[0]: %f <- %d %d %d\n",
          // *v1( r_gl, c_gl-1, f_gl), r_gl, c_gl-1,
          // f_gl);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
              *v1(r_gl, c_gl - 1, f_gl);
        } else {
          // if (debug) printf("load up-1 vsm[0]: 0.0\n");
          v_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] = 0.0;
        }
      }

      if (!PADDING) {
        if (c_gl < nc_c - 1) {
          // if (debug) printf("load down vsm[%d]: %f <- %d %d %d\n", c_sm * 2 +
          // 3, *v2( r_gl, c_gl, f_gl), r_gl, c_gl, f_gl);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)] =
              *v2(r_gl, c_gl, f_gl);
        } else {
          // if (debug) printf("load down vsm[%d]: 0.0\n", c_sm * 2 + 3);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)] = 0.0;
        }
      } else {
        if (c_gl < nc_c - 2) {
          // if (debug) printf("load down vsm[%d]: %f <- %d %d %d\n", c_sm * 2 +
          // 3, *v2( r_gl, c_gl, f_gl), r_gl, c_gl, f_gl);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)] =
              *v2(r_gl, c_gl, f_gl);
        } else {
          // if (debug) printf("load down vsm[%d]: 0.0\n", c_sm * 2 + 3);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)] = 0.0;
        }
      }

      if (c_gl >= 1 &&
          (PADDING && c_gl - 1 < nc_c - 2 || !PADDING && c_gl - 1 < nc_c - 1)) {
        if (c_sm == 0) {
          // if (debug) printf("load down-1 vsm[1]: %f <- %d %d %d\n",
          // *v2( r_gl, c_gl-1, f_gl), r_gl, c_gl-1,
          // f_gl);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, 1, f_sm)] =
              *v2(r_gl, c_gl - 1, f_gl);
        }
      } else {
        if (c_sm == 0) {
          // if (debug) printf("load down-1 vsm[1]: 0.0\n");
          v_sm[get_idx(ldsm1, ldsm2, r_sm, 1, f_sm)] = 0.0;
        }
      }
    }

    // load dist/ratio using f_sm for better performance
    // assumption F >= C
    if (r_sm == 0 && c_sm == 0 && f_sm < actual_C) {
      if (blockId * C * 2 + f_sm < nc) {
        dist_c_sm[2 + f_sm] = *dist_c(blockId * C * 2 + f_sm);
        ratio_c_sm[2 + f_sm] = *ratio_c(blockId * C * 2 + f_sm);
      } else {
        dist_c_sm[2 + f_sm] = 0.0;
        ratio_c_sm[2 + f_sm] = 0.0;
      }

      if (blockId * C * 2 + actual_C + f_sm < nc) {
        dist_c_sm[2 + actual_C + f_sm] =
            *dist_c(blockId * C * 2 + actual_C + f_sm);
        ratio_c_sm[2 + actual_C + f_sm] =
            *ratio_c(blockId * C * 2 + actual_C + f_sm);
      } else {
        dist_c_sm[2 + actual_C + f_sm] = 0.0;
        ratio_c_sm[2 + actual_C + f_sm] = 0.0;
      }
    }

    if (blockId > 0) {
      if (f_sm < 2) {
        dist_c_sm[f_sm] = *dist_c(blockId * C * 2 - 2 + f_sm);
        ratio_c_sm[f_sm] = *ratio_c(blockId * C * 2 - 2 + f_sm);
      }
    } else {
      if (f_sm < 2) {
        dist_c_sm[f_sm] = 0.0;
        ratio_c_sm[f_sm] = 0.0;
      }
    }
  }

  MGARDX_EXEC void Operation3() {

    // __syncthreads();

    if (r_gl < nr && c_gl < nc_c && f_gl < nf_c) {
      T h1 = dist_c_sm[c_sm * 2];
      T h2 = dist_c_sm[c_sm * 2 + 1];
      T h3 = dist_c_sm[c_sm * 2 + 2];
      T h4 = dist_c_sm[c_sm * 2 + 3];
      T r1 = ratio_c_sm[c_sm * 2];
      T r2 = ratio_c_sm[c_sm * 2 + 1];
      T r3 = ratio_c_sm[c_sm * 2 + 2];
      T r4 = 1 - r3;
      T a = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2, f_sm)];
      T b = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 1, f_sm)];
      T c = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 2, f_sm)];
      T d = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)];
      T e = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 4, f_sm)];

      // if (debug) {
      //   printf("c_sm(%d) %f %f %f %f %f\n",c_sm, a,b,c,d,e);
      //   printf("c_sm_h(%d) %f %f %f %f\n",c_sm, h1,h2,h3,h4);
      //   printf("c_sm_r(%d) %f %f %f %f\n",c_sm, r1,r2,r3,r4);
      // }

      // T tb = a * h1 + b * 2 * (h1+h2) + c * h2;
      // T tc = b * h2 + c * 2 * (h2+h3) + d * h3;
      // T td = c * h3 + d * 2 * (h3+h4) + e * h4;

      // if (debug) printf("c_sm(%d) tb tc td tc: %f %f %f %f\n", f_sm, tb, tc,
      // td, tc+tb * r1 + td * r4);

      // tc += tb * r1 + td * r4;

      // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0) {
      //   printf("mr2(%d) mm2: %f -> (%d %d %d)\n", c_sm, tc, r_gl, c_gl,
      //   f_gl);
      //   // printf("f_sm(%d) b c d: %f %f %f\n", f_sm, tb, tc, td);
      // }

      // if (debug) {
      //   printf("f_sm(%d) %f %f %f %f %f f_sm_h %f %f %f %f f_sm_r %f %f %f
      //   %f, out: %f\n",f_sm, a,b,c,d,e, h1,h2,h3,h4,r1,r2,r3,r4,
      //   mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));
      // }

      *w(r_gl, c_gl, f_gl) =
          mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4);

      // if (debug) printf("store[%d %d %d] %f \n", r_gl, c_gl, f_gl,
      //           mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));

      // printf("%d %d %d\n", r_gl, c_gl, f_gl);
      // if (blockId * C + C == nc-1) {
      // if (c_gl + 1 == nc_c - 1) {
      //   // T te = h4 * d + 2 * h4 * e;
      //   // te += td * r3;
      //   *w( r_gl, blockId * C + actual_C, f_gl) =
      //     mass_trans(c, d, e, (T)0.0, (T)0.0,
      //       h1, h2, (T)0.0, (T)0.0, r1, r2, (T)0.0, (T)0.0);
      // }
      // }
    }
    v1.reset_offset();
    v2.reset_offset();
    w.reset_offset();
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size = (R * (C * 2 + 3) * F + (C * 2 + 3) * 2) * sizeof(T);
    size += (D * 4) * sizeof(SIZE);
    size += (D * 1) * sizeof(DIM);
    return size;
  }

private:
  // functor parameters
  SubArray<1, SIZE, DeviceType> shape;
  SubArray<1, SIZE, DeviceType> shape_c;
  DIM processed_n;
  SubArray<1, DIM, DeviceType> processed_dims;
  DIM curr_dim_r, curr_dim_c, curr_dim_f;
  SubArray<1, T, DeviceType> dist_c;
  SubArray<1, T, DeviceType> ratio_c;
  SubArray<D, T, DeviceType> v1, v2, w;

  // thread local variables

  LENGTH threadId;

  T *v_sm;
  SIZE ldsm1, ldsm2;
  T *dist_c_sm;
  T *ratio_c_sm;

  SIZE *shape_sm;
  SIZE *shape_c_sm;
  DIM *processed_dims_sm;

  SIZE idx[D];
  SIZE nr, nc, nf, nf_c, nc_c;
  bool zero_other;
  bool PADDING;

  SIZE bidx;
  SIZE firstD;
  SIZE blockId_f;
  SIZE blockId;

  SIZE zero_r;
  SIZE zero_c;
  SIZE zero_f;

  SIZE r_gl;
  SIZE c_gl;
  SIZE f_gl;

  SIZE r_sm;
  SIZE c_sm;
  SIZE f_sm;

  SIZE actual_C;
};

template <DIM D, typename T, typename DeviceType>
class Lpk2Reo : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  Lpk2Reo() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Lpk2ReoFunctor<D, T, R, C, F, DeviceType>>
  GenTask(SubArray<1, SIZE, DeviceType> shape,
          SubArray<1, SIZE, DeviceType> shape_c, DIM processed_n,
          SubArray<1, DIM, DeviceType> processed_dims, DIM curr_dim_r,
          DIM curr_dim_c, DIM curr_dim_f, SubArray<1, T, DeviceType> dist_c,
          SubArray<1, T, DeviceType> ratio_c, SubArray<D, T, DeviceType> v1,
          SubArray<D, T, DeviceType> v2, SubArray<D, T, DeviceType> w,
          int queue_idx) {
    using FunctorType = Lpk2ReoFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(shape, shape_c, processed_n, processed_dims, curr_dim_r,
                        curr_dim_c, curr_dim_f, dist_c, ratio_c, v1, v2, w);

    SIZE nr = shape.dataHost()[curr_dim_r];
    SIZE nc = shape.dataHost()[curr_dim_c];
    SIZE nf = shape.dataHost()[curr_dim_f];
    SIZE nc_c = shape_c.dataHost()[curr_dim_c];
    SIZE nf_c = shape_c.dataHost()[curr_dim_f];

    SIZE total_thread_z = nr;
    SIZE total_thread_y = nc_c;
    SIZE total_thread_x = nf_c;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
        SIZE t = shape.dataHost()[d];
        for (DIM k = 0; k < processed_n; k++) {
          if (d == processed_dims.dataHost()[k]) {
            t = shape_c.dataHost()[d];
          }
        }
        gridx *= t;
      }
    }

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "Lpk2Reo");
  }

  MGARDX_CONT
  void Execute(SubArray<1, SIZE, DeviceType> shape,
               SubArray<1, SIZE, DeviceType> shape_c, DIM processed_n,
               SubArray<1, DIM, DeviceType> processed_dims, DIM curr_dim_r,
               DIM curr_dim_c, DIM curr_dim_f,
               SubArray<1, T, DeviceType> dist_c,
               SubArray<1, T, DeviceType> ratio_c,
               SubArray<D, T, DeviceType> v1, SubArray<D, T, DeviceType> v2,
               SubArray<D, T, DeviceType> w, int queue_idx) {
    int range_l =
        std::min(6, (int)std::log2(shape_c.dataHost()[curr_dim_f]) - 1);
    int arch = DeviceRuntime<DeviceType>::GetArchitectureGeneration();
    int prec = TypeToIdx<T>();
    // int config =
    // AutoTuner<DeviceType>::autoTuningTable.auto_tuning_mr2[arch][prec][range_l];
    int config = AutoTuner<DeviceType>::autoTuningTable.lpk2_nd[prec][range_l];

    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;

#define LPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = LPK_CONFIG[D - 1][CONFIG][0];                                \
    const int C = LPK_CONFIG[D - 1][CONFIG][1];                                \
    const int F = LPK_CONFIG[D - 1][CONFIG][2];                                \
    using FunctorType = Lpk2ReoFunctor<D, T, R, C, F, DeviceType>;             \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<R, C, F>(                                          \
        shape, shape_c, processed_n, processed_dims, curr_dim_r, curr_dim_c,   \
        curr_dim_f, dist_c, ratio_c, v1, v2, w, queue_idx);                    \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ExecutionReturn ret = adapter.Execute(task);                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (min_time > ret.execution_time) {                                     \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    LPK(0)
    LPK(1)
    LPK(2)
    LPK(3)
    LPK(4)
    LPK(5)
    LPK(6)
#undef LPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("lpk2_nd", prec, range_l, min_config);
    }
  }
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class Lpk3ReoFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT Lpk3ReoFunctor() {}
  MGARDX_CONT Lpk3ReoFunctor(
      SubArray<1, SIZE, DeviceType> shape,
      SubArray<1, SIZE, DeviceType> shape_c, DIM processed_n,
      SubArray<1, DIM, DeviceType> processed_dims, DIM curr_dim_r,
      DIM curr_dim_c, DIM curr_dim_f, SubArray<1, T, DeviceType> dist_r,
      SubArray<1, T, DeviceType> ratio_r, SubArray<D, T, DeviceType> v1,
      SubArray<D, T, DeviceType> v2, SubArray<D, T, DeviceType> w)
      : shape(shape), shape_c(shape_c), processed_n(processed_n),
        processed_dims(processed_dims), curr_dim_r(curr_dim_r),
        curr_dim_c(curr_dim_c), curr_dim_f(curr_dim_f), dist_r(dist_r),
        ratio_r(ratio_r), v1(v1), v2(v2), w(w) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                (FunctorBase<DeviceType>::GetBlockDimX() *
                 FunctorBase<DeviceType>::GetBlockDimY())) +
               (FunctorBase<DeviceType>::GetThreadIdY() *
                FunctorBase<DeviceType>::GetBlockDimX()) +
               FunctorBase<DeviceType>::GetThreadIdX();

    T *sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F;
    ldsm2 = C;
    v_sm = sm;
    sm += ldsm1 * ldsm2 * (R * 2 + 3);

    dist_r_sm = sm;
    sm += (R * 2 + 3);
    ratio_r_sm = sm;
    sm += (R * 2 + 3);

    SIZE *sm_size = (SIZE *)sm;
    shape_sm = sm_size;
    sm_size += D;
    shape_c_sm = sm_size;
    sm_size += D;
    sm = (T *)sm_size;

    DIM *sm_dim = (DIM *)sm;
    processed_dims_sm = sm_dim;
    sm_dim += D;
    sm = (T *)sm_dim;

    if (threadId < D) {
      shape_sm[threadId] = *shape(threadId);
      shape_c_sm[threadId] = *shape_c(threadId);
    }
    if (threadId < processed_n) {
      processed_dims_sm[threadId] = *processed_dims(threadId);
    }
  }
  MGARDX_EXEC void Operation2() {
    // __syncthreads();

    for (DIM d = 0; d < D; d++)
      idx[d] = 0;

    nr = shape_sm[curr_dim_r];
    nf_c = shape_c_sm[curr_dim_f];
    nc_c = shape_c_sm[curr_dim_c];
    nr_c = shape_c_sm[curr_dim_r];
    PADDING = (nr % 2 == 0);

    bidx = FunctorBase<DeviceType>::GetBlockIdX();
    firstD = div_roundup(nf_c, FunctorBase<DeviceType>::GetBlockDimX());
    blockId_f = bidx % firstD;
    bidx /= firstD;

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
        SIZE t = shape_sm[d];
        for (DIM k = 0; k < processed_n; k++) {
          if (d == processed_dims_sm[k]) {
            t = shape_c_sm[d];
          }
        }
        idx[d] = bidx % t;
        bidx /= t;
      }
    }

    v1.offset(idx);
    v2.offset(idx);
    w.offset(idx);

    // LENGTH other_offset_v = get_idx<D>(ldvs_sm, idx);
    // LENGTH other_offset_w = get_idx<D>(ldws_sm, idx);

    // dv1 = dv1 + other_offset_v;
    // dv2 = dv2 + other_offset_v;
    // dw = dw + other_offset_w;

    // if (debug2) {
    //   printf("idx: %d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
    //   printf("other_offset_v: %llu\n", other_offset_v);
    //   printf("other_offset_w: %llu\n", other_offset_w);
    // }

    r_gl = FunctorBase<DeviceType>::GetBlockIdZ() *
               FunctorBase<DeviceType>::GetBlockDimZ() +
           FunctorBase<DeviceType>::GetThreadIdZ();
    c_gl = FunctorBase<DeviceType>::GetBlockIdY() *
               FunctorBase<DeviceType>::GetBlockDimY() +
           FunctorBase<DeviceType>::GetThreadIdY();
    f_gl = blockId_f * FunctorBase<DeviceType>::GetBlockDimX() +
           FunctorBase<DeviceType>::GetThreadIdX();

    blockId = FunctorBase<DeviceType>::GetBlockIdZ();

    r_sm = FunctorBase<DeviceType>::GetThreadIdZ();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();

    actual_R = R;
    if (nr_c - FunctorBase<DeviceType>::GetBlockIdZ() *
                   FunctorBase<DeviceType>::GetBlockDimZ() <
        R) {
      actual_R = nr_c - FunctorBase<DeviceType>::GetBlockIdZ() *
                            FunctorBase<DeviceType>::GetBlockDimZ();
    }
    // if (nr_c % 2 == 1){
    //   if(nr_c-1 - FunctorBase<DeviceType>::GetBlockIdZ() *
    //   FunctorBase<DeviceType>::GetBlockDimZ() < R) { actual_R = nr_c - 1 -
    //   FunctorBase<DeviceType>::GetBlockIdZ() *
    //   FunctorBase<DeviceType>::GetBlockDimZ(); }
    // } else {
    //   if(nr_c - FunctorBase<DeviceType>::GetBlockIdZ() *
    //   FunctorBase<DeviceType>::GetBlockDimZ() < R) { actual_R = nr_c -
    //   FunctorBase<DeviceType>::GetBlockIdZ() *
    //   FunctorBase<DeviceType>::GetBlockDimZ(); }
    // }

    // if (debug) printf("actual_R %d\n", actual_R);

    // bool debug = false;
    // if (idx[3] == 0 && idx[2] == 0  && f_gl == 2 && c_gl == 1) debug = false;

    // if (debug) printf("RCF: %d %d %d\n", R, C, F);
    if (r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
      // if (debug) printf("load front vsm[%d]: %f <- %d %d %d\n", r_sm * 2 + 2,
      // *v1( r_gl, c_gl, f_gl), r_gl, c_gl, f_gl);
      v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 2, c_sm, f_sm)] =
          *v1(r_gl, c_gl, f_gl);

      if (r_sm == actual_R - 1) {
        if (r_gl + 1 < nr_c) {
          // if (debug) printf("load front+1 vsm[%d]: %f <- %d %d %d\n",
          // actual_R
          // * 2 + 2, dv1[get_idx(lddv11, lddv12, blockId * R + actual_R, c_gl,
          // f_gl)], blockId * R + actual_R, c_gl, f_gl);
          v_sm[get_idx(ldsm1, ldsm2, actual_R * 2 + 2, c_sm, f_sm)] =
              *v1(r_gl + 1, c_gl, f_gl);
        } else {
          // if (debug) printf("load front+1 vsm[%d]: 0.0\n", actual_R * 2 + 2);
          v_sm[get_idx(ldsm1, ldsm2, actual_R * 2 + 2, c_sm, f_sm)] = 0.0;
        }
      }

      if (r_sm == 0) {
        if (r_gl >= 1) {
          // if (debug) printf("load front-1 vsm[0]: %f <- %d %d %d\n",
          // *v1( r_gl-1, c_gl, f_gl), r_gl-1, c_gl,
          // f_gl);
          v_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
              *v1(r_gl - 1, c_gl, f_gl);
        } else {
          // if (debug) printf("load front-1 vsm[0]: 0.0\n");
          v_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 0.0;
        }
      }

      if (!PADDING) {
        if (r_gl < nr_c - 1) {
          // if (debug) printf("load back vsm[%d]: 0.0\n", r_sm * 2 + 3);
          v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)] =
              *v2(r_gl, c_gl, f_gl);
        } else {
          // if (debug) printf("load back vsm[%d]: %f <- %d %d %d\n", r_sm * 2 +
          // 3, *v2(r_gl, c_gl, f_gl), r_gl, c_gl, f_gl);
          v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)] = 0.0;
        }
      } else {
        if (r_gl < nr_c - 2) {
          // if (debug) printf("load back vsm[%d]: %f <- %d %d %d\n", r_sm * 2 +
          // 3, *v2(r_gl, c_gl, f_gl), r_gl, c_gl, f_gl);
          v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)] =
              *v2(r_gl, c_gl, f_gl);
        } else {
          // if (debug) printf("load back vsm[%d]: 0.0\n", r_sm * 2 + 3);
          v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)] = 0.0;
        }
      }

      if (r_gl >= 1 &&
          (PADDING && r_gl - 1 < nr_c - 2 || !PADDING && r_gl - 1 < nr_c - 1)) {
        // if (blockId > 0) {
        if (r_sm == 0) {
          // if (debug) printf("load back-1 vsm[1]: %f <- %d %d %d\n",
          // *v2(r_gl-1, c_gl, f_gl), r_gl-1, c_gl,
          // f_gl);
          v_sm[get_idx(ldsm1, ldsm2, 1, c_sm, f_sm)] =
              *v2(r_gl - 1, c_gl, f_gl);
        }
      } else {
        if (r_sm == 0) {
          // if (debug) printf("load back-1 vsm[1]: 0.0\n");
          v_sm[get_idx(ldsm1, ldsm2, 1, c_sm, f_sm)] = 0.0;
        }
      }
    }

    // load dist/ratio using f_sm for better performance
    // assumption F >= R
    if (r_sm == 0 && c_sm == 0 && f_sm < actual_R) {
      if (blockId * R * 2 + f_sm < nr) {
        dist_r_sm[2 + f_sm] = *dist_r(blockId * R * 2 + f_sm);
        // if (debug2 ) printf("load dist 1 [%d]: %f [%d]\n", 2 + f_sm,
        // dist_r_sm[2 + f_sm], blockId * R * 2 + f_sm);
        ratio_r_sm[2 + f_sm] = *ratio_r(blockId * R * 2 + f_sm);
        // if (debug2 )printf("load ratio 1 [%d]: %f [%d]\n", 2 + f_sm,
        // ratio_r_sm[2 + f_sm], blockId * R * 2 + f_sm);
      } else {
        dist_r_sm[2 + f_sm] = 0.0;
        ratio_r_sm[2 + f_sm] = 0.0;
      }
      if (blockId * R * 2 + actual_R + f_sm < nr) {
        dist_r_sm[2 + actual_R + f_sm] =
            *dist_r(blockId * R * 2 + actual_R + f_sm);
        // if (debug2 )printf("load dist 2 [%d]: %f [%d]\n", 2 + actual_R +
        // f_sm, dist_r_sm[2 + actual_R + f_sm], blockId * R * 2 + actual_R +
        // f_sm);
        ratio_r_sm[2 + actual_R + f_sm] =
            *ratio_r(blockId * R * 2 + actual_R + f_sm);
        // if (debug2 )printf("load ratio 2 [%d]: %f [%d]\n", 2 + actual_R +
        // f_sm, ratio_r_sm[2 + actual_R + f_sm], blockId * R * 2 + actual_R +
        // f_sm);
      } else {
        dist_r_sm[2 + actual_R + f_sm] = 0.0;
        ratio_r_sm[2 + actual_R + f_sm] = 0.0;
      }
    }

    if (blockId > 0) {
      if (f_sm < 2) {
        dist_r_sm[f_sm] = *dist_r(blockId * R * 2 - 2 + f_sm);
        // if (debug2 )printf("load dist -1 [%d]: %f [%d]\n", f_sm,
        // dist_r_sm[f_sm], blockId * R * 2 - 2 + f_sm);
        ratio_r_sm[f_sm] = *ratio_r(blockId * R * 2 - 2 + f_sm);
        // if (debug2 )printf("load ratio -1 [%d]: %f [%d]\n", f_sm,
        // ratio_r_sm[f_sm], blockId * R * 2 - 2 + f_sm);
      }
    } else {
      if (f_sm < 2) {
        dist_r_sm[f_sm] = 0.0;
        ratio_r_sm[f_sm] = 0.0;
      }
    }
  }

  MGARDX_EXEC void Operation3() {
    // __syncthreads();

    // int adjusted_nr_c = nr_c;
    if (r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
      T h1 = dist_r_sm[r_sm * 2];
      T h2 = dist_r_sm[r_sm * 2 + 1];
      T h3 = dist_r_sm[r_sm * 2 + 2];
      T h4 = dist_r_sm[r_sm * 2 + 3];
      T r1 = ratio_r_sm[r_sm * 2];
      T r2 = ratio_r_sm[r_sm * 2 + 1];
      T r3 = ratio_r_sm[r_sm * 2 + 2];
      T r4 = 1 - r3;
      T a = v_sm[get_idx(ldsm1, ldsm2, r_sm * 2, c_sm, f_sm)];
      T b = v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 1, c_sm, f_sm)];
      T c = v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 2, c_sm, f_sm)];
      T d = v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)];
      T e = v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 4, c_sm, f_sm)];

      // __syncthreads();
      // if (debug) {
      //   printf("r_sm(%d) %f %f %f %f %f\n",r_sm, a,b,c,d,e);
      //   printf("r_sm_h(%d) %f %f %f %f\n",r_sm, h1,h2,h3,h4);
      //   printf("r_sm_r(%d) %f %f %f %f\n",r_sm, r1,r2,r3,r4);
      // }
      // __syncthreads();

      // T tb = a * h1 + b * 2 * (h1+h2) + c * h2;
      // T tc = b * h2 + c * 2 * (h2+h3) + d * h3;
      // T td = c * h3 + d * 2 * (h3+h4) + e * h4;

      // if (debug) printf("f_sm(%d) tb tc td tc: %f %f %f %f\n", f_sm, tb, tc,
      // td, tc+tb * r1 + td * r4);

      // tc += tb * r1 + td * r4;

      // if (debug) {
      //   printf("f_sm(%d) %f %f %f %f %f f_sm_h %f %f %f %f f_sm_r %f %f %f
      //   %f, out: %f\n",f_sm, a,b,c,d,e, h1,h2,h3,h4,r1,r2,r3,r4,
      //   mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));
      // }

      *w(r_gl, c_gl, f_gl) =
          mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4);

      // if (debug) printf("store[%d %d %d] %f (%f)\n", r_gl, c_gl, f_gl,
      // mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4),
      //                 mass_trans(a, b, c, (T)0.0, (T)0.0, h1, (T)0.0, (T)0.0,
      //                 h4, r1, r2, (T)0.0, (T)0.0));
      // // printf("%d %d %d\n", r_gl, c_gl, f_gl);
      // if (blockId * R + R == nr-1) {
      // if (r_gl+1 == nr_c - 1) {
      // if (r_gl+1 == nr_c - 1) {
      //   // T te = h4 * d + 2 * h4 * e;
      //   // te += td * r3;
      //   *w(blockId * R + actual_R, c_gl, f_gl) =
      //     mass_trans(c, d, e, (T)0.0, (T)0.0,
      //       h1, h2, (T)0.0, (T)0.0, r1, r2, (T)0.0, (T)0.0);

      //   if (debug) printf("store-last[%d %d %d] %f\n", blockId * R +
      //   actual_R, c_gl, f_gl,
      //             mass_trans(c, d, e, (T)0.0, (T)0.0,
      //       h1, h2, (T)0.0, (T)0.0, r1, r2, (T)0.0, (T)0.0));
      // }
      //}
    }
    v1.reset_offset();
    v2.reset_offset();
    w.reset_offset();
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size = ((R * 2 + 3) * C * F + (R * 2 + 3) * 2) * sizeof(T);
    size += (D * 4) * sizeof(SIZE);
    size += (D * 1) * sizeof(DIM);
    return size;
  }

private:
  // functor parameters
  SubArray<1, SIZE, DeviceType> shape;
  SubArray<1, SIZE, DeviceType> shape_c;
  DIM processed_n;
  SubArray<1, DIM, DeviceType> processed_dims;
  DIM curr_dim_r, curr_dim_c, curr_dim_f;
  SubArray<1, T, DeviceType> dist_r;
  SubArray<1, T, DeviceType> ratio_r;
  SubArray<D, T, DeviceType> v1, v2, w;

  // thread local variables

  LENGTH threadId;

  T *v_sm;
  SIZE ldsm1, ldsm2;
  T *dist_r_sm;
  T *ratio_r_sm;

  SIZE *shape_sm;
  SIZE *shape_c_sm;
  DIM *processed_dims_sm;

  SIZE idx[D];
  SIZE nr, nc, nf, nf_c, nc_c, nr_c;
  bool zero_other;
  bool PADDING;

  SIZE bidx;
  SIZE firstD;
  SIZE blockId_f;
  SIZE blockId;

  SIZE zero_r;
  SIZE zero_c;
  SIZE zero_f;

  SIZE r_gl;
  SIZE c_gl;
  SIZE f_gl;

  SIZE r_sm;
  SIZE c_sm;
  SIZE f_sm;

  SIZE actual_R;
};

template <DIM D, typename T, typename DeviceType>
class Lpk3Reo : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  Lpk3Reo() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Lpk3ReoFunctor<D, T, R, C, F, DeviceType>>
  GenTask(SubArray<1, SIZE, DeviceType> shape,
          SubArray<1, SIZE, DeviceType> shape_c, DIM processed_n,
          SubArray<1, DIM, DeviceType> processed_dims, DIM curr_dim_r,
          DIM curr_dim_c, DIM curr_dim_f, SubArray<1, T, DeviceType> dist_r,
          SubArray<1, T, DeviceType> ratio_r, SubArray<D, T, DeviceType> v1,
          SubArray<D, T, DeviceType> v2, SubArray<D, T, DeviceType> w,
          int queue_idx) {
    using FunctorType = Lpk3ReoFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(shape, shape_c, processed_n, processed_dims, curr_dim_r,
                        curr_dim_c, curr_dim_f, dist_r, ratio_r, v1, v2, w);

    SIZE nr = shape.dataHost()[curr_dim_r];
    SIZE nc = shape.dataHost()[curr_dim_c];
    SIZE nf = shape.dataHost()[curr_dim_f];
    SIZE nr_c = shape_c.dataHost()[curr_dim_r];
    SIZE nc_c = shape_c.dataHost()[curr_dim_c];
    SIZE nf_c = shape_c.dataHost()[curr_dim_f];

    SIZE total_thread_z = nr_c;
    SIZE total_thread_y = nc_c;
    SIZE total_thread_x = nf_c;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
        SIZE t = shape.dataHost()[d];
        for (DIM k = 0; k < processed_n; k++) {
          if (d == processed_dims.dataHost()[k]) {
            t = shape_c.dataHost()[d];
          }
        }
        gridx *= t;
      }
    }

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "Lpk3Reo");
  }

  MGARDX_CONT
  void Execute(SubArray<1, SIZE, DeviceType> shape,
               SubArray<1, SIZE, DeviceType> shape_c, DIM processed_n,
               SubArray<1, DIM, DeviceType> processed_dims, DIM curr_dim_r,
               DIM curr_dim_c, DIM curr_dim_f,
               SubArray<1, T, DeviceType> dist_r,
               SubArray<1, T, DeviceType> ratio_r,
               SubArray<D, T, DeviceType> v1, SubArray<D, T, DeviceType> v2,
               SubArray<D, T, DeviceType> w, int queue_idx) {
    int range_l =
        std::min(6, (int)std::log2(shape_c.dataHost()[curr_dim_f]) - 1);
    int arch = DeviceRuntime<DeviceType>::GetArchitectureGeneration();
    int prec = TypeToIdx<T>();
    // int config =
    // AutoTuner<DeviceType>::autoTuningTable.auto_tuning_mr3[arch][prec][range_l];
    int config = AutoTuner<DeviceType>::autoTuningTable.lpk3_nd[prec][range_l];

    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;

#define LPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = LPK_CONFIG[D - 1][CONFIG][0];                                \
    const int C = LPK_CONFIG[D - 1][CONFIG][1];                                \
    const int F = LPK_CONFIG[D - 1][CONFIG][2];                                \
    using FunctorType = Lpk3ReoFunctor<D, T, R, C, F, DeviceType>;             \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<R, C, F>(                                          \
        shape, shape_c, processed_n, processed_dims, curr_dim_r, curr_dim_c,   \
        curr_dim_f, dist_r, ratio_r, v1, v2, w, queue_idx);                    \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ExecutionReturn ret = adapter.Execute(task);                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (min_time > ret.execution_time) {                                     \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    LPK(0)
    LPK(1)
    LPK(2)
    LPK(3)
    LPK(4)
    LPK(5)
    LPK(6)
#undef LPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("lpk3_nd", prec, range_l, min_config);
    }
  }
};

// template <DIM D, typename T, SIZE R, SIZE C, SIZE F>
// __global__ void
// _lpk_reo_1(SIZE *shape, SIZE *shape_c, SIZE *ldvs, SIZE *ldws, DIM
// processed_n,
//            DIM *processed_dims, DIM curr_dim_r, DIM curr_dim_c, DIM
//            curr_dim_f, T *ddist_f, T *dratio_f, T *dv1, LENGTH lddv11, LENGTH
//            lddv12, T *dv2, LENGTH lddv21, LENGTH lddv22, T *dw, LENGTH lddw1,
//            LENGTH lddw2) {

//   // bool debug = false;
//   // if (blockIdx.z == 0 && blockIdx.y == 1 && blockIdx.x == 1 &&
//   // threadIdx.z == 0 && threadIdx.y == 0 ) debug = false;

//   // bool debug = false;
//   // if (threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0 ) debug =
//   // true;

//   LENGTH threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
//                     (threadIdx.y * blockDim.x) + threadIdx.x;

//   T *sm = SharedMemory<T>();
//   SIZE ldsm1 = F * 2 + 3;
//   SIZE ldsm2 = C;
//   T *v_sm = sm; sm += ldsm1 * ldsm2 * R;

//   T *dist_f_sm = sm; sm += ldsm1;
//   T *ratio_f_sm = sm; sm += ldsm1;

//   SIZE * sm_size = (SIZE*)sm;
//   SIZE *shape_sm = sm_size; sm_size += D;
//   SIZE *shape_c_sm = sm_size; sm_size += D;
//   SIZE *ldvs_sm = sm_size; sm_size += D;
//   SIZE *ldws_sm = sm_size; sm_size += D;
//   sm = (T*)sm_size;

//   DIM * sm_dim = (DIM*)sm;
//   DIM *processed_dims_sm = sm_dim; sm_dim += D;
//   sm = (T*)sm_dim;

//   SIZE idx[D];
//   if (threadId < D) {
//     shape_sm[threadId] = shape[threadId];
//     shape_c_sm[threadId] = shape_c[threadId];
//     ldvs_sm[threadId] = ldvs[threadId];
//     ldws_sm[threadId] = ldws[threadId];
//   }
//   if (threadId < processed_n) {
//     processed_dims_sm[threadId] = processed_dims[threadId];
//   }
//   __syncthreads();

//   for (DIM d = 0; d < D; d++)
//     idx[d] = 0;

//   SIZE nr = shape_sm[curr_dim_r];
//   SIZE nc = shape_sm[curr_dim_c];
//   SIZE nf = shape_sm[curr_dim_f];
//   SIZE nf_c = shape_c_sm[curr_dim_f];

//   bool zero_other = true;
//   bool PADDING = (nf % 2 == 0);

//   SIZE bidx = blockIdx.x;
//   SIZE firstD;
//   if (nf_c % 2 == 1) {
//     firstD = div_roundup(nf_c, blockDim.x);
//   } else {
//     firstD = div_roundup(nf_c, blockDim.x);
//   }
//   SIZE blockId = bidx % firstD;
//   bidx /= firstD;

//   for (DIM d = 0; d < D; d++) {
//     if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
//       SIZE t = shape_sm[d];
//       for (DIM k = 0; k < processed_n; k++) {
//         if (d == processed_dims[k]) {
//           t = shape_c_sm[d];
//         }
//       }
//       idx[d] = bidx % t;
//       bidx /= t;
//       if (idx[d] >= shape_c_sm[d])
//         zero_other = false;
//     }
//   }

//   SIZE zero_r = shape_c_sm[curr_dim_r];
//   SIZE zero_c = shape_c_sm[curr_dim_c];
//   SIZE zero_f = shape_c_sm[curr_dim_f];

//   if (D < 3) {
//     nr = 1;
//     zero_r = 1;
//   }
//   if (D < 2) {
//     nc = 1;
//     zero_c = 1;
//   }

//   LENGTH other_offset_v = get_idx<D>(ldvs_sm, idx);
//   LENGTH other_offset_w = get_idx<D>(ldws_sm, idx);

//   dv1 = dv1 + other_offset_v;
//   dv2 = dv2 + other_offset_v;
//   dw = dw + other_offset_w;

//   // if (debug2) {
//   //   printf("idx: %d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
//   //   printf("other_offset_v: %llu\n", other_offset_v);
//   //   printf("other_offset_w: %llu\n", other_offset_w);
//   // }
//   SIZE r_gl = blockIdx.z * blockDim.z + threadIdx.z;
//   SIZE c_gl = blockIdx.y * blockDim.y + threadIdx.y;
//   SIZE f_gl = blockId * blockDim.x + threadIdx.x;

//   SIZE r_sm = threadIdx.z;
//   SIZE c_sm = threadIdx.y;
//   SIZE f_sm = threadIdx.x;

//   SIZE actual_F = F;
//   if (nf_c - blockId * blockDim.x < F) {
//     actual_F = nf_c - blockId * blockDim.x;
//   }

//   // if (nf_c % 2 == 1){
//   //   if(nf_c-1 - blockId * blockDim.x < F) { actual_F = nf_c - 1 - blockId
//   *
//   //   blockDim.x; }
//   // } else {
//   //   if(nf_c - blockId * blockDim.x < F) { actual_F = nf_c - blockId *
//   //   blockDim.x; }
//   // }

//   // if (debug) printf("actual_F %d\n", actual_F);

//   if (r_gl < nr && c_gl < nc && f_gl < nf_c) {
//     if (zero_other && r_gl < zero_r && c_gl < zero_c && f_gl < zero_f) {
//       // if (debug) printf("load left vsm[%d]: 0.0\n", f_sm * 2 + 2);
//       v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 2)] = 0.0;
//     } else {
//       // if (debug) printf("load left vsm[%d]<-dv1[%d, %d, %d]: %f\n", f_sm *
//       2
//       // + 2, r_gl, c_gl, f_gl, dv1[get_idx(lddv11, lddv12, r_gl, c_gl,
//       f_gl)]); v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 2)] =
//           dv1[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)];
//     }

//     if (f_sm == actual_F - 1) {
//       if (zero_other && r_gl < zero_r && c_gl < zero_c && f_gl < zero_f) {
//         // if (debug) printf("load left+1 vsm[%d]: 0.0\n", actual_F * 2 + 2);
//         v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 2)] = 0.0;
//       } else {
//         if (f_gl + 1 < nf_c) {
//           // if (debug) printf("load left+1 vsm[%d]: %f\n", actual_F * 2 + 2,
//           // dv1[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl + 1)]);
//           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 2)] =
//               dv1[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl + 1)];
//         } else {
//           // if (debug) printf("load left+1 vsm[%d]: 0.0\n", actual_F * 2 +
//           2); v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 2)] =
//           0.0;
//         }
//       }
//     }

//     if (f_sm == 0) {
//       // left
//       if (zero_other && r_gl < zero_r && c_gl < zero_c && f_gl < zero_f) {
//         // coarse (-1)
//         // if (debug) printf("load left-1 vsm[0]: 0.0\n");
//         v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 0.0;
//       } else {
//         if (f_gl >= 1) {
//           // other (-1)
//           // if (debug) printf("load left-1 vsm[0]: %f\n",
//           dv1[get_idx(lddv11,
//           // lddv12, r_gl, c_gl, f_gl-1)]);
//           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
//               dv1[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl - 1)];
//         } else {
//           // other (-1)
//           // if (debug) printf("load left-1 vsm[0]: 0.0\n");
//           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 0.0;
//         }
//       }
//     }

//     // right
//     if (!PADDING) { //other = nf_c - 1
//       if (nf_c % 2 != 0) {
//         if (f_gl >= 1 && f_gl < nf_c) { //shift for better memory access
//         pattern
//           // if (debug) printf("load right vsm[%d]: %f <- %d %d %d\n", f_sm *
//           2
//           // + 1, dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl - 1)], r_gl,
//           // c_gl, f_gl - 1);
//           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)] =
//               dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl - 1)];
//         } else {
//           // if (debug) printf("load right vsm[%d]: 0\n", f_sm * 2 + 1);
//           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)] = 0.0;
//         }
//       } else { // nf_c % 2 == 0, do not shift
//         if (f_gl < nf_c - 1) {
//           // if (debug) printf("load right vsm[%d]: %f <- %d %d %d\n", f_sm *
//           2
//           // + 3, dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl)], r_gl, c_gl,
//           // f_gl);
//           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 3)] =
//               dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl)];
//         } else {
//           // if (debug) printf("load right vsm[%d]: 0\n", f_sm * 2 + 3);
//           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 3)] = 0.0;
//         }
//       }
//     } else { // PADDING other = nf_c - 2
//       if (nf_c % 2 != 0) {
//         if (f_gl >= 1 && f_gl < nf_c - 1) { //shift for better memory access
//         pattern
//           // if (debug) printf("load right vsm[%d]: %f <- %d %d %d\n", f_sm *
//           2
//           // + 1, dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl - 1)], r_gl,
//           // c_gl, f_gl - 1);
//           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)] =
//               dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl - 1)];
//         } else {
//           // if (debug) printf("load right vsm[%d]: 0\n", f_sm * 2 + 1);
//           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)] = 0.0;
//         }
//       } else { // nf_c % 2 == 0
//         if (f_gl < nf_c - 2) { // do not shift
//           // if (debug) printf("load right vsm[%d]: %f <- %d %d %d\n", f_sm *
//           2
//           // + 3, dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl)], r_gl, c_gl,
//           // f_gl);
//           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 3)] =
//               dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl)];
//         } else {
//           // if (debug) printf("load right vsm[%d]: 0\n", f_sm * 2 + 3);
//           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 3)] = 0.0;
//         }
//       }
//     }

//     if (f_sm == actual_F - 1) {
//       // right (+1)
//       if (!PADDING) {
//         if (nf_c % 2 != 0) {
//           if (f_gl < nf_c - 1) {
//             // if (debug) printf("load right+1 vsm[%d]: %f <- %d %d %d\n",
//             // actual_F * 2 + 1, dv2[get_idx(lddv21, lddv22, r_gl, c_gl,
//             f_gl)],
//             // r_gl, c_gl, f_gl);
//             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 1)] =
//                 dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl)];
//           } else {
//             // if (debug) printf("load right+1 vsm[%d]: 0.0\n", actual_F * 2
//             +
//             // 1);
//             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 1)] = 0.0;
//           }
//         } else { // nf_c % 2 == 0
//           if (f_gl >= actual_F) {
//             // if (debug) printf("load right-1 vsm[1]: %f <- %d %d %d\n",
//             // dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl - actual_F)],
//             r_gl,
//             // c_gl, f_gl - actual_F);
//             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 1)] =
//                 dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl - actual_F)];
//           } else {
//             // if (debug) printf("load right-1 vsm[1]: 0.0\n");
//             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 1)] = 0.0;
//           }
//         }
//       } else {
//         if (nf_c % 2 != 0) {
//           if (f_gl < nf_c - 2) {
//             // if (debug) printf("actual_F(%d), load right+1 vsm[%d]: %f <-
//             %d %d %d\n",
//                               // actual_F, actual_F * 2 + 1,
//                               dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl)],
//                               r_gl, c_gl, f_gl);
//             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 1)] =
//                 dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl)];
//           } else {
//             // if (debug) printf("load right+1 vsm[%d]: 0.0\n", actual_F * 2
//             +
//             // 1);
//             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 1)] = 0.0;
//           }
//         } else { // nf_c % 2 == 0
//           if (f_gl >= actual_F && f_gl - actual_F < nf_c - 2) {
//             // if (debug) printf("load right-1 vsm[1]: %f <- %d %d %d\n",
//             // dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl - actual_F)],
//             r_gl,
//             // c_gl, f_gl - actual_F);
//             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 1)] =
//                 dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl - actual_F)];
//           } else {
//             // if (debug) printf("load right-1 vsm[1]: 0.0\n");
//             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 1)] = 0.0;
//           }
//         }
//       }
//     }
//   }

//   // if (debug)  printf("actual_F: %d\n", actual_F);
//   if (r_sm == 0 && c_sm == 0 && f_sm < actual_F) {
//     // if (debug) printf("blockId * F * 2 + f_sm = %d\n", blockId * F * 2 +
//     f_sm); if (blockId * F * 2 + f_sm < nf) { // padding: num of dist == nf,
//     non-padding: non of dist == nf - 1
//       // if (debug) printf("load dist/ratio1[%d]: %f <- %d\n", 2 + f_sm,
//       ddist_f[blockId * F * 2 + f_sm], blockId * F * 2 + f_sm); dist_f_sm[2 +
//       f_sm] = ddist_f[blockId * F * 2 + f_sm]; ratio_f_sm[2 + f_sm] =
//       dratio_f[blockId * F * 2 + f_sm];
//     } else {
//       // if (debug) printf("load dist/ratio1[%d]: 0.0\n", 2 + f_sm);
//       dist_f_sm[2 + f_sm] = 0.0;
//       ratio_f_sm[2 + f_sm] = 0.0;
//     }

//     if (blockId * F * 2 + actual_F + f_sm < nf) {
//       // if (debug) printf("load dist/ratio2[%d]: %f <- %d\n", 2 + actual_F +
//       f_sm, ddist_f[blockId * F * 2 + actual_F + f_sm], blockId * F * 2 +
//       actual_F + f_sm); dist_f_sm[2 + actual_F + f_sm] =
//           ddist_f[blockId * F * 2 + actual_F + f_sm];
//       ratio_f_sm[2 + actual_F + f_sm] =
//           dratio_f[blockId * F * 2 + actual_F + f_sm];
//     } else {
//       // if (debug) printf("load dist/ratio2[%d]: 0.0\n", 2 + actual_F +
//       f_sm); dist_f_sm[2 + actual_F + f_sm] = 0.0; ratio_f_sm[2 + actual_F +
//       f_sm] = 0.0;
//     }
//   }

//   if (blockId > 0) {
//     if (f_sm < 2) {
//       // dist_f_sm[f_sm] = ddist_f[f_gl - 2];
//       // ratio_f_sm[f_sm] = dratio_f[f_gl - 2];
//       // if (debug) printf("load dist/ratio-1[%d]: %f <- %d\n", f_sm,
//       ddist_f[blockId * F * 2 + f_sm - 2], blockId * F * 2 + f_sm - 2);
//       dist_f_sm[f_sm] = ddist_f[blockId * F * 2 + f_sm - 2];
//       ratio_f_sm[f_sm] = dratio_f[blockId * F * 2 + f_sm - 2];
//     }
//   } else {
//     if (f_sm < 2) {
//       // if (debug) printf("load dist/ratio-1[%d]: 0.0 <- %d\n", f_sm);
//       dist_f_sm[f_sm] = 0.0;
//       ratio_f_sm[f_sm] = 0.0;
//     }
//   }

//   __syncthreads();

//   if (r_gl < nr && c_gl < nc && f_gl < nf_c) {
//     T h1 = dist_f_sm[f_sm * 2];
//     T h2 = dist_f_sm[f_sm * 2 + 1];
//     T h3 = dist_f_sm[f_sm * 2 + 2];
//     T h4 = dist_f_sm[f_sm * 2 + 3];
//     T r1 = ratio_f_sm[f_sm * 2];
//     T r2 = ratio_f_sm[f_sm * 2 + 1];
//     T r3 = ratio_f_sm[f_sm * 2 + 2];
//     T r4 = 1 - r3;
//     T a = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2)];
//     T b = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)];
//     T c = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 2)];
//     T d = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 3)];
//     T e = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 4)];

//     // bool debug = false;
//     // if (idx[3] == 0) debug = false;
//     // if (debug) {
//     //   printf("f_sm(%d) %f %f %f %f %f f_sm_h %f %f %f %f f_sm_r %f %f %f
//     %f, out: %f\n",f_sm, a,b,c,d,e, h1,h2,h3,h4,r1,r2,r3,r4, mass_trans(a, b,
//     c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));
//     // }

//     // T tb = a * h1/6 + b * (h1+h2)/3 + c * h2/6;
//     // T tc = b * h2/6 + c * (h2+h3)/3 + d * h3/6;
//     // T td = c * h3/6 + d * (h3+h4)/3 + e * h4/6;

//     // if (debug) printf("f_sm(%d) tb tc td tc: %f %f %f %f\n", f_sm, tb, tc,
//     // td, tc+tb * r1 + td * r4);

//     // tc += tb * r1 + td * r4;

//     dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] =
//         mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4);

//     // if (debug) printf("store[%d %d %d] %f \n", r_gl, c_gl, f_gl,
//     //           mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));

//     // printf("test block %d F %d nf %d\n", blockId, F, nf);
//     // if (f_gl+1 == nf_c-1) {

//     //     // T te = h4 * d + 2 * h4 * e;
//     //     //printf("f_sm(%d) mm-e: %f\n", f_sm, te);
//     //     // te += td * r3;
//     //     dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl+1)] =
//     //       mass_trans(c, d, e, (T)0.0, (T)0.0, h1, h2, (T)0.0, (T)0.0, r1,
//     r2,
//     //       (T)0.0, (T)0.0);
//     // }
//   }
// }

// template <DIM D, typename T, SIZE R, SIZE C, SIZE F>
// void lpk_reo_1_adaptive_launcher(Handle<D, T> &handle, SIZE *shape_h,
//                                  SIZE *shape_c_h, SIZE *shape_d, SIZE
//                                  *shape_c_d, SIZE *ldvs, SIZE *ldws, DIM
//                                  processed_n, DIM *processed_dims_h, DIM
//                                  *processed_dims_d, DIM curr_dim_r, DIM
//                                  curr_dim_c, DIM curr_dim_f, T *ddist_f, T
//                                  *dratio_f, T *dv1, LENGTH lddv11, LENGTH
//                                  lddv12, T *dv2, LENGTH lddv21, LENGTH
//                                  lddv22, T *dw, LENGTH lddw1, LENGTH lddw2,
//                                  int queue_idx) {
//   SIZE nr = shape_h[curr_dim_r];
//   SIZE nc = shape_h[curr_dim_c];
//   SIZE nf = shape_h[curr_dim_f];
//   SIZE nf_c = shape_c_h[curr_dim_f];

//   SIZE total_thread_z = nr;
//   SIZE total_thread_y = nc;
//   SIZE total_thread_x = nf_c;
//   // if (nf_c % 2 == 1) { total_thread_x = nf_c - 1; }
//   // else { total_thread_x = nf_c; }
//   SIZE tbx, tby, tbz, gridx, gridy, gridz;
//   dim3 threadsPerBlock, blockPerGrid;
//   size_t sm_size;

//   tbz = R;
//   tby = C;
//   tbx = F;
//   sm_size = (R * C * (F * 2 + 3) + (F * 2 + 3) * 2) * sizeof(T);
//   sm_size += (D * 4) * sizeof(SIZE);
//   sm_size += (D * 1) * sizeof(DIM);
//   gridz = ceil((float)total_thread_z / tbz);
//   gridy = ceil((float)total_thread_y / tby);
//   gridx = ceil((float)total_thread_x / tbx);
//   for (DIM d = 0; d < D; d++) {
//     if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
//       SIZE t = shape_h[d];
//       for (DIM k = 0; k < processed_n; k++) {
//         if (d == processed_dims_h[k]) {
//           t = shape_c_h[d];
//         }
//       }
//       gridx *= t;
//     }
//   }
//   threadsPerBlock = dim3(tbx, tby, tbz);
//   blockPerGrid = dim3(gridx, gridy, gridz);

//   // printf("lpk_reo_1 exec config (%d %d %d) (%d %d %d)\n", tbx, tby, tbz,
//   // gridx, gridy, gridz);
//   _lpk_reo_1<D, T, R, C, F><<<blockPerGrid, threadsPerBlock, sm_size,
//                               *(cudaStream_t *)handle.get(queue_idx)>>>(
//       shape_d, shape_c_d, ldvs, ldws, processed_n, processed_dims_d,
//       curr_dim_r, curr_dim_c, curr_dim_f, ddist_f, dratio_f, dv1, lddv11,
//       lddv12, dv2, lddv21, lddv22, dw, lddw1, lddw2);
//   gpuErrchk(cudaGetLastError());
//   if (handle.sync_and_check_all_kernels) {
//     gpuErrchk(cudaDeviceSynchronize());
//   }
// }

// template <DIM D, typename T>
// void lpk_reo_1(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_c_h, SIZE
// *shape_d,
//                SIZE *shape_c_d, SIZE *ldvs, SIZE *ldws, DIM processed_n,
//                DIM *processed_dims_h, DIM *processed_dims_d, DIM curr_dim_r,
//                DIM curr_dim_c, DIM curr_dim_f, T *ddist_f, T *dratio_f, T
//                *dv1, LENGTH lddv11, LENGTH lddv12, T *dv2, LENGTH lddv21,
//                LENGTH lddv22, T *dw, LENGTH lddw1, LENGTH lddw2, int
//                queue_idx, int config) {
//   #define LPK(R, C, F) \
//   { \
//     lpk_reo_1_adaptive_launcher<D, T, R, C, F>( \
//         handle, shape_h, shape_c_h, shape_d, shape_c_d, ldvs, ldws, \
//         processed_n, processed_dims_h, processed_dims_d, curr_dim_r, \
//         curr_dim_c, curr_dim_f, ddist_f, dratio_f, dv1, lddv11, lddv12, dv2,
//         \
//         lddv21, lddv22, dw, lddw1, lddw2, queue_idx); \
//   }

//   bool profile = false;
//   if (handle.profile_kernels) {
//     profile = true;
//   }
//   if (D >= 3) {
//     if (profile || config == 6) {
//       LPK(2, 2, 128)
//     }
//     if (profile || config == 5) {
//       LPK(2, 2, 64)
//     }
//     if (profile || config == 4) {
//       LPK(2, 2, 32)
//     }
//     if (profile || config == 3) {
//       LPK(4, 4, 16)
//     }
//     if (profile || config == 2) {
//       LPK(8, 8, 8)
//     }
//     if (profile || config == 1) {
//       LPK(4, 4, 4)
//     }
//     if (profile || config == 0) {
//       LPK(2, 2, 2)
//     }
//   } else if (D == 2) {
//     if (profile || config == 6) {
//       LPK(1, 2, 128)
//     }
//     if (profile || config == 5) {
//       LPK(1, 2, 64)
//     }
//     if (profile || config == 4) {
//       LPK(1, 2, 32)
//     }
//     if (profile || config == 3) {
//       LPK(1, 4, 16)
//     }
//     if (profile || config == 2) {
//       LPK(1, 8, 8)
//     }
//     if (profile || config == 1) {
//       LPK(1, 4, 4)
//     }
//     if (profile || config == 0) {
//       LPK(1, 2, 4)
//     }
//   } else if (D == 1) {
//     if (profile || config == 6) {
//       LPK(1, 1, 128)
//     }
//     if (profile || config == 5) {
//       LPK(1, 1, 64)
//     }
//     if (profile || config == 4) {
//       LPK(1, 1, 32)
//     }
//     if (profile || config == 3) {
//       LPK(1, 1, 16)
//     }
//     if (profile || config == 2) {
//       LPK(1, 1, 8)
//     }
//     if (profile || config == 1) {
//       LPK(1, 1, 8)
//     }
//     if (profile || config == 0) {
//       LPK(1, 1, 8)
//     }
//   }

//   #undef LPK
// }

// template <DIM D, typename T, SIZE R, SIZE C, SIZE F>
// __global__ void
// _lpk_reo_2(SIZE *shape, SIZE *shape_c, SIZE *ldvs, SIZE *ldws, DIM
// processed_n,
//            DIM *processed_dims, DIM curr_dim_r, DIM curr_dim_c, DIM
//            curr_dim_f, T *ddist_c, T *dratio_c, T *dv1, LENGTH lddv11, LENGTH
//            lddv12, T *dv2, LENGTH lddv21, LENGTH lddv22, T *dw, LENGTH lddw1,
//            LENGTH lddw2) {

//   // bool debug = false;
//   // if (blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 &&
//   // threadIdx.z == 0 && threadIdx.x == 0 ) debug = false;

//   // bool debug2 = false;
//   // if (threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0 ) debug2 =
//   // false;

//   LENGTH threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
//                     (threadIdx.y * blockDim.x) + threadIdx.x;

//   T *sm = SharedMemory<T>();
//   SIZE ldsm1 = F;
//   SIZE ldsm2 = C * 2 + 3;
//   T *v_sm = sm; sm += ldsm1 * ldsm2 * R;

//   T *dist_c_sm = sm; sm += ldsm2;
//   T *ratio_c_sm = sm; sm += ldsm2;

//   SIZE * sm_size = (SIZE*)sm;
//   SIZE *shape_sm = sm_size; sm_size += D;
//   SIZE *shape_c_sm = sm_size; sm_size += D;
//   SIZE *ldvs_sm = sm_size; sm_size += D;
//   SIZE *ldws_sm = sm_size; sm_size += D;
//   sm = (T*)sm_size;

//   DIM * sm_dim = (DIM*)sm;
//   DIM *processed_dims_sm = sm_dim; sm_dim += D;
//   sm = (T*)sm_dim;

//   SIZE idx[D];
//   if (threadId < D) {
//     shape_sm[threadId] = shape[threadId];
//     shape_c_sm[threadId] = shape_c[threadId];
//     ldvs_sm[threadId] = ldvs[threadId];
//     ldws_sm[threadId] = ldws[threadId];
//   }
//   if (threadId < processed_n) {
//     processed_dims_sm[threadId] = processed_dims[threadId];
//   }
//   __syncthreads();

//   for (DIM d = 0; d < D; d++)
//     idx[d] = 0;

//   SIZE nr = shape_sm[curr_dim_r];
//   SIZE nc = shape_sm[curr_dim_c];
//   SIZE nf_c = shape_c_sm[curr_dim_f];
//   SIZE nc_c = shape_c_sm[curr_dim_c];
//   bool PADDING = (nc % 2 == 0);

//   if (D < 3) {
//     nr = 1;
//   }

//   SIZE bidx = blockIdx.x;
//   SIZE firstD = div_roundup(nf_c, blockDim.x);
//   SIZE blockId_f = bidx % firstD;
//   bidx /= firstD;

//   for (DIM d = 0; d < D; d++) {
//     if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
//       SIZE t = shape_sm[d];
//       for (DIM k = 0; k < processed_n; k++) {
//         if (d == processed_dims[k]) {
//           t = shape_c_sm[d];
//         }
//       }
//       idx[d] = bidx % t;
//       bidx /= t;
//     }
//   }

//   LENGTH other_offset_v = get_idx<D>(ldvs_sm, idx);
//   LENGTH other_offset_w = get_idx<D>(ldws_sm, idx);

//   dv1 = dv1 + other_offset_v;
//   dv2 = dv2 + other_offset_v;
//   dw = dw + other_offset_w;

//   // if (debug2) {
//   //   printf("idx: %d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
//   //   printf("other_offset_v: %llu\n", other_offset_v);
//   //   printf("other_offset_w: %llu\n", other_offset_w);
//   // }

//   SIZE r_gl = blockIdx.z * blockDim.z + threadIdx.z;
//   SIZE c_gl = blockIdx.y * blockDim.y + threadIdx.y;
//   SIZE f_gl = blockId_f * blockDim.x + threadIdx.x;

//   SIZE blockId = blockIdx.y;

//   SIZE r_sm = threadIdx.z;
//   SIZE c_sm = threadIdx.y;
//   SIZE f_sm = threadIdx.x;

//   SIZE actual_C = C;
//   if (nc_c - blockIdx.y * blockDim.y < C) {
//     actual_C = nc_c - blockIdx.y * blockDim.y;
//   }

//   // if (nc_c % 2 == 1){
//   //   if(nc_c-1 - blockIdx.y * blockDim.y < C) { actual_C = nc_c - 1 -
//   //   blockIdx.y * blockDim.y; }
//   // } else {
//   //   if(nc_c - blockIdx.y * blockDim.y < C) { actual_C = nc_c - blockIdx.y
//   *
//   //   blockDim.y; }
//   // }

//   // bool debug = false;
//   // if (idx[3] == 0 && r_gl == 0 ) debug = false;

//   // if (debug) printf("actual_C %d\n", actual_C);

//   if (r_gl < nr && c_gl < nc_c && f_gl < nf_c) {
//     // if (debug) printf("load up vsm[%d]: %f <- %d %d %d\n", c_sm * 2 + 2,
//     // dv1[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
//     v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 2, f_sm)] =
//         dv1[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)];

//     if (c_sm == actual_C - 1) {
//       if (c_gl + 1 < nc_c) {
//         // if (debug) printf("load up+1 vsm[%d]: %f <- %d %d %d\n", actual_C
//         * 2
//         // + 2, dv1[get_idx(lddv11, lddv12, r_gl, blockId * C + actual_C,
//         // f_gl)], r_gl, blockId * C + actual_C, f_gl);
//         // c_gl+1 == blockId * C + C
//         v_sm[get_idx(ldsm1, ldsm2, r_sm, actual_C * 2 + 2, f_sm)] =
//             dv1[get_idx(lddv11, lddv12, r_gl, c_gl + 1, f_gl)];
//       } else {
//         // if (debug) printf("load up+1 vsm[%d]: 0.0\n", actual_C * 2 + 2);
//         v_sm[get_idx(ldsm1, ldsm2, r_sm, actual_C * 2 + 2, f_sm)] = 0.0;
//       }
//     }

//     if (c_sm == 0) {
//       if (c_gl >= 1) {
//         // if (debug) printf("load up-1 vsm[0]: %f <- %d %d %d\n",
//         // dv1[get_idx(lddv11, lddv12, r_gl, c_gl-1, f_gl)], r_gl, c_gl-1,
//         // f_gl);
//         v_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
//             dv1[get_idx(lddv11, lddv12, r_gl, c_gl - 1, f_gl)];
//       } else {
//         // if (debug) printf("load up-1 vsm[0]: 0.0\n");
//         v_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] = 0.0;
//       }
//     }

//     if (!PADDING) {
//       if (c_gl < nc_c - 1) {
//         // if (debug) printf("load down vsm[%d]: %f <- %d %d %d\n", c_sm * 2
//         +
//         // 3, dv2[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)], r_gl, c_gl,
//         f_gl); v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)] =
//             dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl)];
//       } else {
//         // if (debug) printf("load down vsm[%d]: 0.0\n", c_sm * 2 + 3);
//         v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)] = 0.0;
//       }
//     } else {
//       if (c_gl < nc_c - 2) {
//         // if (debug) printf("load down vsm[%d]: %f <- %d %d %d\n", c_sm * 2
//         +
//         // 3, dv2[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)], r_gl, c_gl,
//         f_gl); v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)] =
//             dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl)];
//       } else {
//         // if (debug) printf("load down vsm[%d]: 0.0\n", c_sm * 2 + 3);
//         v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)] = 0.0;
//       }
//     }

//     if (c_gl >= 1 &&
//         (PADDING && c_gl - 1 < nc_c - 2 || !PADDING && c_gl - 1 < nc_c - 1))
//         {
//       if (c_sm == 0) {
//         // if (debug) printf("load down-1 vsm[1]: %f <- %d %d %d\n",
//         // dv2[get_idx(lddv11, lddv12, r_gl, c_gl-1, f_gl)], r_gl, c_gl-1,
//         // f_gl);
//         v_sm[get_idx(ldsm1, ldsm2, r_sm, 1, f_sm)] =
//             dv2[get_idx(lddv11, lddv12, r_gl, c_gl - 1, f_gl)];
//       }
//     } else {
//       if (c_sm == 0) {
//         // if (debug) printf("load down-1 vsm[1]: 0.0\n");
//         v_sm[get_idx(ldsm1, ldsm2, r_sm, 1, f_sm)] = 0.0;
//       }
//     }
//   }

//   // load dist/ratio using f_sm for better performance
//   // assumption F >= C
//   if (r_sm == 0 && c_sm == 0 && f_sm < actual_C) {
//     if (blockId * C * 2 + f_sm < nc) {
//       dist_c_sm[2 + f_sm] = ddist_c[blockId * C * 2 + f_sm];
//       ratio_c_sm[2 + f_sm] = dratio_c[blockId * C * 2 + f_sm];
//     } else {
//       dist_c_sm[2 + f_sm] = 0.0;
//       ratio_c_sm[2 + f_sm] = 0.0;
//     }

//     if (blockId * C * 2 + actual_C + f_sm < nc) {
//       dist_c_sm[2 + actual_C + f_sm] =
//           ddist_c[blockId * C * 2 + actual_C + f_sm];
//       ratio_c_sm[2 + actual_C + f_sm] =
//           dratio_c[blockId * C * 2 + actual_C + f_sm];
//     } else {
//       dist_c_sm[2 + actual_C + f_sm] = 0.0;
//       ratio_c_sm[2 + actual_C + f_sm] = 0.0;
//     }
//   }

//   if (blockId > 0) {
//     if (f_sm < 2) {
//       dist_c_sm[f_sm] = ddist_c[blockId * C * 2 - 2 + f_sm];
//       ratio_c_sm[f_sm] = dratio_c[blockId * C * 2 - 2 + f_sm];
//     }
//   } else {
//     if (f_sm < 2) {
//       dist_c_sm[f_sm] = 0.0;
//       ratio_c_sm[f_sm] = 0.0;
//     }
//   }

//   __syncthreads();

//   if (r_gl < nr && c_gl < nc_c && f_gl < nf_c) {
//     T h1 = dist_c_sm[c_sm * 2];
//     T h2 = dist_c_sm[c_sm * 2 + 1];
//     T h3 = dist_c_sm[c_sm * 2 + 2];
//     T h4 = dist_c_sm[c_sm * 2 + 3];
//     T r1 = ratio_c_sm[c_sm * 2];
//     T r2 = ratio_c_sm[c_sm * 2 + 1];
//     T r3 = ratio_c_sm[c_sm * 2 + 2];
//     T r4 = 1 - r3;
//     T a = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2, f_sm)];
//     T b = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 1, f_sm)];
//     T c = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 2, f_sm)];
//     T d = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)];
//     T e = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 4, f_sm)];

//     // if (debug) {
//     //   printf("c_sm(%d) %f %f %f %f %f\n",c_sm, a,b,c,d,e);
//     //   printf("c_sm_h(%d) %f %f %f %f\n",c_sm, h1,h2,h3,h4);
//     //   printf("c_sm_r(%d) %f %f %f %f\n",c_sm, r1,r2,r3,r4);
//     // }

//     // T tb = a * h1 + b * 2 * (h1+h2) + c * h2;
//     // T tc = b * h2 + c * 2 * (h2+h3) + d * h3;
//     // T td = c * h3 + d * 2 * (h3+h4) + e * h4;

//     // if (debug) printf("c_sm(%d) tb tc td tc: %f %f %f %f\n", f_sm, tb, tc,
//     // td, tc+tb * r1 + td * r4);

//     // tc += tb * r1 + td * r4;

//     // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0) {
//     //   printf("mr2(%d) mm2: %f -> (%d %d %d)\n", c_sm, tc, r_gl, c_gl,
//     f_gl);
//     //   // printf("f_sm(%d) b c d: %f %f %f\n", f_sm, tb, tc, td);
//     // }

//     // if (debug) {
//     //   printf("f_sm(%d) %f %f %f %f %f f_sm_h %f %f %f %f f_sm_r %f %f %f
//     %f, out: %f\n",f_sm, a,b,c,d,e, h1,h2,h3,h4,r1,r2,r3,r4, mass_trans(a, b,
//     c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));
//     // }

//     dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] =
//         mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4);

//     // if (debug) printf("store[%d %d %d] %f \n", r_gl, c_gl, f_gl,
//     //           mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));

//     // printf("%d %d %d\n", r_gl, c_gl, f_gl);
//     // if (blockId * C + C == nc-1) {
//     // if (c_gl + 1 == nc_c - 1) {
//     //   // T te = h4 * d + 2 * h4 * e;
//     //   // te += td * r3;
//     //   dw[get_idx(lddw1, lddw2, r_gl, blockId * C + actual_C, f_gl)] =
//     //     mass_trans(c, d, e, (T)0.0, (T)0.0,
//     //       h1, h2, (T)0.0, (T)0.0, r1, r2, (T)0.0, (T)0.0);
//     // }
//     // }
//   }
// }

// template <DIM D, typename T, SIZE R, SIZE C, SIZE F>
// void lpk_reo_2_adaptive_launcher(Handle<D, T> &handle, SIZE *shape_h,
//                                  SIZE *shape_c_h, SIZE *shape_d, SIZE
//                                  *shape_c_d, SIZE *ldvs, SIZE *ldws, DIM
//                                  processed_n, DIM *processed_dims_h, DIM
//                                  *processed_dims_d, DIM curr_dim_r, DIM
//                                  curr_dim_c, DIM curr_dim_f, T *ddist_c, T
//                                  *dratio_c, T *dv1, LENGTH lddv11, LENGTH
//                                  lddv12, T *dv2, LENGTH lddv21, LENGTH
//                                  lddv22, T *dw, LENGTH lddw1, LENGTH lddw2,
//                                  int queue_idx) {

//   SIZE nr = shape_h[curr_dim_r];
//   SIZE nc = shape_h[curr_dim_c];
//   SIZE nf = shape_h[curr_dim_f];
//   SIZE nc_c = shape_c_h[curr_dim_c];
//   SIZE nf_c = shape_c_h[curr_dim_f];

//   SIZE total_thread_z = nr;
//   SIZE total_thread_y = nc_c;
//   // if (nc_c % 2 == 1) { total_thread_y = nc_c - 1; }
//   // else { total_thread_y = nc_c; }
//   SIZE total_thread_x = nf_c;
//   SIZE tbx, tby, tbz, gridx, gridy, gridz;
//   dim3 threadsPerBlock, blockPerGrid;
//   size_t sm_size;

//   tbz = R;
//   tby = C;
//   tbx = F;
//   sm_size = (R * (C * 2 + 3) * F + (C * 2 + 3) * 2) * sizeof(T);
//   sm_size += (D * 4) * sizeof(SIZE);
//   sm_size += (D * 1) * sizeof(DIM);
//   gridz = ceil((float)total_thread_z / tbz);
//   gridy = ceil((float)total_thread_y / tby);
//   gridx = ceil((float)total_thread_x / tbx);
//   for (DIM d = 0; d < D; d++) {
//     if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
//       SIZE t = shape_h[d];
//       for (DIM k = 0; k < processed_n; k++) {
//         if (d == processed_dims_h[k]) {
//           t = shape_c_h[d];
//         }
//       }
//       gridx *= t;
//     }
//   }
//   threadsPerBlock = dim3(tbx, tby, tbz);
//   blockPerGrid = dim3(gridx, gridy, gridz);
//   // printf("lpk_reo_2 exec config (%d %d %d) (%d %d %d)\n", tbx, tby, tbz,
//   // gridx, gridy, gridz);

//   _lpk_reo_2<D, T, R, C, F><<<blockPerGrid, threadsPerBlock, sm_size,
//                               *(cudaStream_t *)handle.get(queue_idx)>>>(
//       shape_d, shape_c_d, ldvs, ldws, processed_n, processed_dims_d,
//       curr_dim_r, curr_dim_c, curr_dim_f, ddist_c, dratio_c, dv1, lddv11,
//       lddv12, dv2, lddv21, lddv22, dw, lddw1, lddw2);
//   gpuErrchk(cudaGetLastError());
//   if (handle.sync_and_check_all_kernels) {
//     gpuErrchk(cudaDeviceSynchronize());
//   }
// }

// template <DIM D, typename T>
// void lpk_reo_2(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_c_h, SIZE
// *shape_d,
//                SIZE *shape_c_d, SIZE *ldvs, SIZE *ldws, DIM processed_n,
//                DIM *processed_dims_h, DIM *processed_dims_d, DIM curr_dim_r,
//                DIM curr_dim_c, DIM curr_dim_f, T *ddist_c, T *dratio_c, T
//                *dv1, LENGTH lddv11, LENGTH lddv12, T *dv2, LENGTH lddv21,
//                LENGTH lddv22, T *dw, LENGTH lddw1, LENGTH lddw2, int
//                queue_idx, int config) {

//   #define LPK(R, C, F)                                                           \
//   {                                                                            \
//     lpk_reo_2_adaptive_launcher<D, T, R, C, F>(                                \
//         handle, shape_h, shape_c_h, shape_d, shape_c_d, ldvs, ldws,            \
//         processed_n, processed_dims_h, processed_dims_d,\ 
//                                curr_dim_r,                                     \
//         curr_dim_c, curr_dim_f, ddist_c, dratio_c, dv1, lddv11, lddv12, dv2,   \
//         lddv21, lddv22, dw, lddw1, lddw2, queue_idx);                          \
//   }

//   bool profile = false;
//   if (handle.profile_kernels) {
//     profile = true;
//   }
//   if (D >= 3) {
//     if (profile || config == 6) {
//       LPK(2, 2, 128)
//     }
//     if (profile || config == 5) {
//       LPK(2, 2, 64)
//     }
//     if (profile || config == 4) {
//       LPK(2, 2, 32)
//     }
//     if (profile || config == 3) {
//       LPK(4, 4, 16)
//     }
//     if (profile || config == 2) {
//       LPK(8, 8, 8)
//     }
//     if (profile || config == 1) {
//       LPK(4, 4, 4)
//     }
//     if (profile || config == 0) {
//       LPK(2, 2, 2)
//     }
//   } else if (D == 2) {
//     if (profile || config == 6) {
//       LPK(1, 2, 128)
//     }
//     if (profile || config == 5) {
//       LPK(1, 2, 64)
//     }
//     if (profile || config == 4) {
//       LPK(1, 2, 32)
//     }
//     if (profile || config == 3) {
//       LPK(1, 4, 16)
//     }
//     if (profile || config == 2) {
//       LPK(1, 8, 8)
//     }
//     if (profile || config == 1) {
//       LPK(1, 4, 4)
//     }
//     if (profile || config == 0) {
//       LPK(1, 2, 4)
//     }
//   } else {
//     printf("Error: mass_trans_multiply_2_cpt is only for 3D and 2D data\n");
//   }
//   #undef LPK
// }

// template <DIM D, typename T, SIZE R, SIZE C, SIZE F>
// __global__ void
// _lpk_reo_3(SIZE *shape, SIZE *shape_c, SIZE *ldvs, SIZE *ldws, DIM
// processed_n,
//            DIM *processed_dims, DIM curr_dim_r, DIM curr_dim_c, DIM
//            curr_dim_f, T *ddist_r, T *dratio_r, T *dv1, LENGTH lddv11, LENGTH
//            lddv12, T *dv2, LENGTH lddv21, LENGTH lddv22, T *dw, LENGTH lddw1,
//            LENGTH lddw2) {

//   // bool debug = false;
//   // if (blockIdx.z == gridDim.z-1 && blockIdx.y == 0 && blockIdx.x == 0 &&
//   // threadIdx.y == 0 && threadIdx.x == 0 ) debug = false;

//   // bool debug2 = false;
//   // if (blockIdx.z == gridDim.z-1 && blockIdx.y == 1 && blockIdx.x == 16)
//   // debug2 = false;

//   LENGTH threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
//                     (threadIdx.y * blockDim.x) + threadIdx.x;

//   T *sm = SharedMemory<T>();
//   SIZE ldsm1 = F;
//   SIZE ldsm2 = C;
//   T *v_sm = sm; sm += ldsm1 * ldsm2 * (R * 2 + 3);

//   T *dist_r_sm = sm; sm += (R * 2 + 3);
//   T *ratio_r_sm = sm; sm += (R * 2 + 3);

//   SIZE * sm_size = (SIZE*)sm;
//   SIZE *shape_sm = sm_size; sm_size += D;
//   SIZE *shape_c_sm = sm_size; sm_size += D;
//   SIZE *ldvs_sm = sm_size; sm_size += D;
//   SIZE *ldws_sm = sm_size; sm_size += D;
//   sm = (T*)sm_size;

//   DIM * sm_dim = (DIM*)sm;
//   DIM *processed_dims_sm = sm_dim; sm_dim += D;
//   sm = (T*)sm_dim;

//   SIZE idx[D];
//   if (threadId < D) {
//     shape_sm[threadId] = shape[threadId];
//     shape_c_sm[threadId] = shape_c[threadId];
//     ldvs_sm[threadId] = ldvs[threadId];
//     ldws_sm[threadId] = ldws[threadId];
//   }
//   if (threadId < processed_n) {
//     processed_dims_sm[threadId] = processed_dims[threadId];
//   }
//   __syncthreads();

//   for (DIM d = 0; d < D; d++)
//     idx[d] = 0;

//   SIZE nr = shape_sm[curr_dim_r];
//   SIZE nf_c = shape_c_sm[curr_dim_f];
//   SIZE nc_c = shape_c_sm[curr_dim_c];
//   SIZE nr_c = shape_c_sm[curr_dim_r];
//   bool PADDING = (nr % 2 == 0);

//   SIZE bidx = blockIdx.x;
//   SIZE firstD = div_roundup(nf_c, blockDim.x);
//   SIZE blockId_f = bidx % firstD;
//   bidx /= firstD;

//   for (DIM d = 0; d < D; d++) {
//     if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
//       SIZE t = shape_sm[d];
//       for (DIM k = 0; k < processed_n; k++) {
//         if (d == processed_dims[k]) {
//           t = shape_c_sm[d];
//         }
//       }
//       idx[d] = bidx % t;
//       bidx /= t;
//     }
//   }

//   LENGTH other_offset_v = get_idx<D>(ldvs_sm, idx);
//   LENGTH other_offset_w = get_idx<D>(ldws_sm, idx);

//   dv1 = dv1 + other_offset_v;
//   dv2 = dv2 + other_offset_v;
//   dw = dw + other_offset_w;

//   // if (debug2) {
//   //   printf("idx: %d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
//   //   printf("other_offset_v: %llu\n", other_offset_v);
//   //   printf("other_offset_w: %llu\n", other_offset_w);
//   // }

//   SIZE r_gl = blockIdx.z * blockDim.z + threadIdx.z;
//   SIZE c_gl = blockIdx.y * blockDim.y + threadIdx.y;
//   SIZE f_gl = blockId_f * blockDim.x + threadIdx.x;

//   SIZE blockId = blockIdx.z;

//   SIZE r_sm = threadIdx.z;
//   SIZE c_sm = threadIdx.y;
//   SIZE f_sm = threadIdx.x;

//   SIZE actual_R = R;
//   if (nr_c - blockIdx.z * blockDim.z < R) {
//     actual_R = nr_c - blockIdx.z * blockDim.z;
//   }
//   // if (nr_c % 2 == 1){
//   //   if(nr_c-1 - blockIdx.z * blockDim.z < R) { actual_R = nr_c - 1 -
//   //   blockIdx.z * blockDim.z; }
//   // } else {
//   //   if(nr_c - blockIdx.z * blockDim.z < R) { actual_R = nr_c - blockIdx.z
//   *
//   //   blockDim.z; }
//   // }

//   // if (debug) printf("actual_R %d\n", actual_R);

//   // bool debug = false;
//   // if (idx[3] == 0 && idx[2] == 0  && f_gl == 2 && c_gl == 1) debug =
//   false;

//   // if (debug) printf("RCF: %d %d %d\n", R, C, F);
//   if (r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
//     // if (debug) printf("load front vsm[%d]: %f <- %d %d %d\n", r_sm * 2 +
//     2,
//     // dv1[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
//     v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 2, c_sm, f_sm)] =
//         dv1[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)];

//     if (r_sm == actual_R - 1) {
//       if (r_gl + 1 < nr_c) {
//         // if (debug) printf("load front+1 vsm[%d]: %f <- %d %d %d\n",
//         actual_R
//         // * 2 + 2, dv1[get_idx(lddv11, lddv12, blockId * R + actual_R, c_gl,
//         // f_gl)], blockId * R + actual_R, c_gl, f_gl);
//         v_sm[get_idx(ldsm1, ldsm2, actual_R * 2 + 2, c_sm, f_sm)] =
//             dv1[get_idx(lddv11, lddv12, r_gl + 1, c_gl, f_gl)];
//       } else {
//         // if (debug) printf("load front+1 vsm[%d]: 0.0\n", actual_R * 2 +
//         2); v_sm[get_idx(ldsm1, ldsm2, actual_R * 2 + 2, c_sm, f_sm)] = 0.0;
//       }
//     }

//     if (r_sm == 0) {
//       if (r_gl >= 1) {
//         // if (debug) printf("load front-1 vsm[0]: %f <- %d %d %d\n",
//         // dv1[get_idx(lddv11, lddv12, r_gl-1, c_gl, f_gl)], r_gl-1, c_gl,
//         // f_gl);
//         v_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
//             dv1[get_idx(lddv11, lddv12, r_gl - 1, c_gl, f_gl)];
//       } else {
//         // if (debug) printf("load front-1 vsm[0]: 0.0\n");
//         v_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 0.0;
//       }
//     }

//     if (!PADDING) {
//       if (r_gl < nr_c - 1) {
//         // if (debug) printf("load back vsm[%d]: 0.0\n", r_sm * 2 + 3);
//         v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)] =
//             dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl)];
//       } else {
//         // if (debug) printf("load back vsm[%d]: %f <- %d %d %d\n", r_sm * 2
//         +
//         // 3, dv2[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)], r_gl, c_gl,
//         f_gl); v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)] = 0.0;
//       }
//     } else {
//       if (r_gl < nr_c - 2) {
//         // if (debug) printf("load back vsm[%d]: %f <- %d %d %d\n", r_sm * 2
//         +
//         // 3, dv2[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)], r_gl, c_gl,
//         f_gl); v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)] =
//             dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl)];
//       } else {
//         // if (debug) printf("load back vsm[%d]: 0.0\n", r_sm * 2 + 3);
//         v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)] = 0.0;
//       }
//     }

//     if (r_gl >= 1 &&
//         (PADDING && r_gl - 1 < nr_c - 2 || !PADDING && r_gl - 1 < nr_c - 1))
//         {
//       // if (blockId > 0) {
//       if (r_sm == 0) {
//         // if (debug) printf("load back-1 vsm[1]: %f <- %d %d %d\n",
//         // dv2[get_idx(lddv11, lddv12, r_gl-1, c_gl, f_gl)], r_gl-1, c_gl,
//         // f_gl);
//         v_sm[get_idx(ldsm1, ldsm2, 1, c_sm, f_sm)] =
//             dv2[get_idx(lddv11, lddv12, r_gl - 1, c_gl, f_gl)];
//       }
//     } else {
//       if (r_sm == 0) {
//         // if (debug) printf("load back-1 vsm[1]: 0.0\n");
//         v_sm[get_idx(ldsm1, ldsm2, 1, c_sm, f_sm)] = 0.0;
//       }
//     }
//   }

//   // load dist/ratio using f_sm for better performance
//   // assumption F >= R
//   if (r_sm == 0 && c_sm == 0 && f_sm < actual_R) {
//     if (blockId * R * 2 + f_sm < nr) {
//       dist_r_sm[2 + f_sm] = ddist_r[blockId * R * 2 + f_sm];
//       // if (debug2 ) printf("load dist 1 [%d]: %f [%d]\n", 2 + f_sm,
//       // dist_r_sm[2 + f_sm], blockId * R * 2 + f_sm);
//       ratio_r_sm[2 + f_sm] = dratio_r[blockId * R * 2 + f_sm];
//       // if (debug2 )printf("load ratio 1 [%d]: %f [%d]\n", 2 + f_sm,
//       // ratio_r_sm[2 + f_sm], blockId * R * 2 + f_sm);
//     } else {
//       dist_r_sm[2 + f_sm] = 0.0;
//       ratio_r_sm[2 + f_sm] = 0.0;
//     }
//     if (blockId * R * 2 + actual_R + f_sm < nr) {
//       dist_r_sm[2 + actual_R + f_sm] =
//           ddist_r[blockId * R * 2 + actual_R + f_sm];
//       // if (debug2 )printf("load dist 2 [%d]: %f [%d]\n", 2 + actual_R +
//       f_sm,
//       // dist_r_sm[2 + actual_R + f_sm], blockId * R * 2 + actual_R + f_sm);
//       ratio_r_sm[2 + actual_R + f_sm] =
//           dratio_r[blockId * R * 2 + actual_R + f_sm];
//       // if (debug2 )printf("load ratio 2 [%d]: %f [%d]\n", 2 + actual_R +
//       f_sm,
//       // ratio_r_sm[2 + actual_R + f_sm], blockId * R * 2 + actual_R + f_sm);
//     } else {
//       dist_r_sm[2 + actual_R + f_sm] = 0.0;
//       ratio_r_sm[2 + actual_R + f_sm] = 0.0;
//     }
//   }

//   if (blockId > 0) {
//     if (f_sm < 2) {
//       dist_r_sm[f_sm] = ddist_r[blockId * R * 2 - 2 + f_sm];
//       // if (debug2 )printf("load dist -1 [%d]: %f [%d]\n", f_sm,
//       // dist_r_sm[f_sm], blockId * R * 2 - 2 + f_sm);
//       ratio_r_sm[f_sm] = dratio_r[blockId * R * 2 - 2 + f_sm];
//       // if (debug2 )printf("load ratio -1 [%d]: %f [%d]\n", f_sm,
//       // ratio_r_sm[f_sm], blockId * R * 2 - 2 + f_sm);
//     }
//   } else {
//     if (f_sm < 2) {
//       dist_r_sm[f_sm] = 0.0;
//       ratio_r_sm[f_sm] = 0.0;
//     }
//   }

//   __syncthreads();

//   int adjusted_nr_c = nr_c;
//   if (r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
//     T h1 = dist_r_sm[r_sm * 2];
//     T h2 = dist_r_sm[r_sm * 2 + 1];
//     T h3 = dist_r_sm[r_sm * 2 + 2];
//     T h4 = dist_r_sm[r_sm * 2 + 3];
//     T r1 = ratio_r_sm[r_sm * 2];
//     T r2 = ratio_r_sm[r_sm * 2 + 1];
//     T r3 = ratio_r_sm[r_sm * 2 + 2];
//     T r4 = 1 - r3;
//     T a = v_sm[get_idx(ldsm1, ldsm2, r_sm * 2, c_sm, f_sm)];
//     T b = v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 1, c_sm, f_sm)];
//     T c = v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 2, c_sm, f_sm)];
//     T d = v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)];
//     T e = v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 4, c_sm, f_sm)];

//     // __syncthreads();
//     // if (debug) {
//     //   printf("r_sm(%d) %f %f %f %f %f\n",r_sm, a,b,c,d,e);
//     //   printf("r_sm_h(%d) %f %f %f %f\n",r_sm, h1,h2,h3,h4);
//     //   printf("r_sm_r(%d) %f %f %f %f\n",r_sm, r1,r2,r3,r4);
//     // }
//     // __syncthreads();

//     // T tb = a * h1 + b * 2 * (h1+h2) + c * h2;
//     // T tc = b * h2 + c * 2 * (h2+h3) + d * h3;
//     // T td = c * h3 + d * 2 * (h3+h4) + e * h4;

//     // if (debug) printf("f_sm(%d) tb tc td tc: %f %f %f %f\n", f_sm, tb, tc,
//     // td, tc+tb * r1 + td * r4);

//     // tc += tb * r1 + td * r4;

//     // if (debug) {
//     //   printf("f_sm(%d) %f %f %f %f %f f_sm_h %f %f %f %f f_sm_r %f %f %f
//     %f, out: %f\n",f_sm, a,b,c,d,e, h1,h2,h3,h4,r1,r2,r3,r4, mass_trans(a, b,
//     c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));
//     // }

//     dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] =
//         mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4);

//     // if (debug) printf("store[%d %d %d] %f (%f)\n", r_gl, c_gl, f_gl,
//     // mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4),
//     //                 mass_trans(a, b, c, (T)0.0, (T)0.0, h1, (T)0.0,
//     (T)0.0,
//     //                 h4, r1, r2, (T)0.0, (T)0.0));
//     // // printf("%d %d %d\n", r_gl, c_gl, f_gl);
//     // if (blockId * R + R == nr-1) {
//     // if (r_gl+1 == nr_c - 1) {
//     // if (r_gl+1 == nr_c - 1) {
//     //   // T te = h4 * d + 2 * h4 * e;
//     //   // te += td * r3;
//     //   dw[get_idx(lddw1, lddw2, blockId * R + actual_R, c_gl, f_gl)] =
//     //     mass_trans(c, d, e, (T)0.0, (T)0.0,
//     //       h1, h2, (T)0.0, (T)0.0, r1, r2, (T)0.0, (T)0.0);

//     //   if (debug) printf("store-last[%d %d %d] %f\n", blockId * R +
//     actual_R,
//     //   c_gl, f_gl,
//     //             mass_trans(c, d, e, (T)0.0, (T)0.0,
//     //       h1, h2, (T)0.0, (T)0.0, r1, r2, (T)0.0, (T)0.0));
//     // }
//     //}
//   }
// }

// template <DIM D, typename T, SIZE R, SIZE C, SIZE F>
// void lpk_reo_3_adaptive_launcher(Handle<D, T> &handle, SIZE *shape_h,
//                                  SIZE *shape_c_h, SIZE *shape_d, SIZE
//                                  *shape_c_d, SIZE *ldvs, SIZE *ldws, DIM
//                                  processed_n, DIM *processed_dims_h, DIM
//                                  *processed_dims_d, DIM curr_dim_r, DIM
//                                  curr_dim_c, DIM curr_dim_f, T *ddist_r, T
//                                  *dratio_r, T *dv1, LENGTH lddv11, LENGTH
//                                  lddv12, T *dv2, LENGTH lddv21, LENGTH
//                                  lddv22, T *dw, LENGTH lddw1, LENGTH lddw2,
//                                  int queue_idx) {

//   SIZE nr = shape_h[curr_dim_r];
//   SIZE nc = shape_h[curr_dim_c];
//   SIZE nf = shape_h[curr_dim_f];
//   SIZE nr_c = shape_c_h[curr_dim_r];
//   SIZE nc_c = shape_c_h[curr_dim_c];
//   SIZE nf_c = shape_c_h[curr_dim_f];

//   SIZE total_thread_z = nr_c;
//   // if (nr_c % 2 == 1){ total_thread_z = nr_c - 1; }
//   // else { total_thread_z = nr_c; }
//   SIZE total_thread_y = nc_c;
//   SIZE total_thread_x = nf_c;

//   SIZE tbx, tby, tbz, gridx, gridy, gridz;
//   dim3 threadsPerBlock, blockPerGrid;
//   size_t sm_size;

//   tbz = R;
//   tby = C;
//   tbx = F;
//   sm_size = ((R * 2 + 3) * C * F + (R * 2 + 3) * 2) * sizeof(T);
//   sm_size += (D * 4) * sizeof(SIZE);
//   sm_size += (D * 1) * sizeof(DIM);
//   gridz = ceil((float)total_thread_z / tbz);
//   gridy = ceil((float)total_thread_y / tby);
//   gridx = ceil((float)total_thread_x / tbx);
//   for (DIM d = 0; d < D; d++) {
//     if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
//       SIZE t = shape_h[d];
//       for (DIM k = 0; k < processed_n; k++) {
//         if (d == processed_dims_h[k]) {
//           t = shape_c_h[d];
//         }
//       }
//       gridx *= t;
//     }
//   }
//   threadsPerBlock = dim3(tbx, tby, tbz);
//   blockPerGrid = dim3(gridx, gridy, gridz);
//   // printf("lpk_reo_3 exec config (%d %d %d) (%d %d %d)\n", tbx, tby, tbz,
//   // gridx, gridy, gridz);

//   _lpk_reo_3<D, T, R, C, F><<<blockPerGrid, threadsPerBlock, sm_size,
//                               *(cudaStream_t *)handle.get(queue_idx)>>>(
//       shape_d, shape_c_d, ldvs, ldws, processed_n, processed_dims_d,
//       curr_dim_r, curr_dim_c, curr_dim_f, ddist_r, dratio_r, dv1, lddv11,
//       lddv12, dv2, lddv21, lddv22, dw, lddw1, lddw2);
//   gpuErrchk(cudaGetLastError());
//   if (handle.sync_and_check_all_kernels) {
//     gpuErrchk(cudaDeviceSynchronize());
//   }
// }

// template <DIM D, typename T>
// void lpk_reo_3(Handle<D, T> &handle, SIZE *shape_h, SIZE *shape_c_h, SIZE
// *shape_d,
//                SIZE *shape_c_d, SIZE *ldvs, SIZE *ldws, DIM processed_n,
//                DIM *processed_dims_h, DIM *processed_dims_d, DIM curr_dim_r,
//                DIM curr_dim_c, DIM curr_dim_f, T *ddist_r, T *dratio_r, T
//                *dv1, LENGTH lddv11, LENGTH lddv12, T *dv2, LENGTH lddv21,
//                LENGTH lddv22, T *dw, LENGTH lddw1, LENGTH lddw2, int
//                queue_idx, int config) {

//   #define LPK(R, C, F)                                                           \
//   {                                                                            \
//     lpk_reo_3_adaptive_launcher<D, T, R, C, F>(                                \
//         handle, shape_h, shape_c_h, shape_d, shape_c_d, ldvs, ldws,            \
//         processed_n, processed_dims_h, processed_dims_d,\ 
//                                curr_dim_r,                                     \
//         curr_dim_c, curr_dim_f, ddist_r, dratio_r, dv1, lddv11, lddv12, dv2,   \
//         lddv21, lddv22, dw, lddw1, lddw2, queue_idx);                          \
//   }
//   bool profile = false;
//   if (handle.profile_kernels) {
//     profile = true;
//   }
//   if (D >= 3) {
//     if (profile || config == 6) {
//       LPK(2, 2, 128)
//     }
//     if (profile || config == 5) {
//       LPK(2, 2, 64)
//     }
//     if (profile || config == 4) {
//       LPK(2, 2, 32)
//     }
//     if (profile || config == 3) {
//       LPK(4, 4, 16)
//     }
//     if (profile || config == 2) {
//       LPK(8, 8, 8)
//     }
//     if (profile || config == 1) {
//       LPK(4, 4, 4)
//     }
//     if (profile || config == 0) {
//       LPK(2, 2, 2)
//     }
//   } else {
//     printf("Error: mass_trans_multiply_3_cpt is only for 3D data\n");
//   }

//   #undef LPK
// }

} // namespace mgard_x

#endif