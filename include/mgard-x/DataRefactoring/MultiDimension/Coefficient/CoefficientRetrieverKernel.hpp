/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_COEFFICIENT_RETRIEVER_TEMPLATE
#define MGARD_X_COEFFICIENT_RETRIEVER_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"
#include "GPKFunctor.h"

#define NO_FEATURE 0
#define HAS_FEATURE 1
#define DISCARD_CHILDREN 0
#define KEEP_CHILDREN 1

namespace mgard_x {


template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class CoefficientRetrieverFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT CoefficientRetrieverFunctor() {}
  MGARDX_CONT CoefficientRetrieverFunctor(
      SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
      SubArray<D, T, DeviceType> w, SubArray<D, T, DeviceType> wf,
      SubArray<D, T, DeviceType> wc, SubArray<D, T, DeviceType> wr,
      SubArray<D, T, DeviceType> wcf, SubArray<D, T, DeviceType> wrf,
      SubArray<D, T, DeviceType> wrc, SubArray<D, T, DeviceType> wrcf,
      SubArray<D, SIZE, DeviceType> refinement_flag,
      SubArray<D, T, DeviceType> w_new, SubArray<D, T, DeviceType> wf_new,
      SubArray<D, T, DeviceType> wc_new, SubArray<D, T, DeviceType> wr_new,
      SubArray<D, T, DeviceType> wcf_new, SubArray<D, T, DeviceType> wrf_new,
      SubArray<D, T, DeviceType> wrc_new, SubArray<D, T, DeviceType> wrcf_new)
      : nr(nr), nc(nc), nf(nf), nr_c(nr_c), nc_c(nc_c), nf_c(nf_c),
        wf(wf), wc(wc), wr(wr), wrcf(wrcf), wcf(wcf), wrf(wrf), wrc(wrc),
        refinement_flag(refinement_flag),
        wf_new(wf_new), wc_new(wc_new), wr_new(wr_new), 
        wrcf_new(wrcf_new), wcf_new(wcf_new), wrf_new(wrf_new), wrc_new(wrc_new) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    r = FunctorBase<DeviceType>::GetBlockIdZ() *
        FunctorBase<DeviceType>::GetBlockDimZ();
    c = FunctorBase<DeviceType>::GetBlockIdY() *
        FunctorBase<DeviceType>::GetBlockDimY();
    f = FunctorBase<DeviceType>::GetBlockIdX() *
        FunctorBase<DeviceType>::GetBlockDimX();

    r_sm = FunctorBase<DeviceType>::GetThreadIdZ();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();

    r_sm_ex = (R / 2) * 2;
    c_sm_ex = (C / 2) * 2;
    f_sm_ex = (F / 2) * 2;

    sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();

    ldsm1 = (F / 2) * 2 + 1;
    ldsm2 = (C / 2) * 2 + 1;
    v_sm = sm;
    ratio_f_sm = sm + ((F / 2) * 2 + 1) * ((C / 2) * 2 + 1) * ((R / 2) * 2 + 1);
    ratio_c_sm = ratio_f_sm + (F / 2) * 2;
    ratio_r_sm = ratio_c_sm + (C / 2) * 2;

    rest_r = nr - r;
    rest_c = nc - c;
    rest_f = nf - f;

    nr_p = nr;
    nc_p = nc;
    nf_p = nf;

    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                (FunctorBase<DeviceType>::GetBlockDimX() *
                 FunctorBase<DeviceType>::GetBlockDimY())) +
               (FunctorBase<DeviceType>::GetThreadIdY() *
                FunctorBase<DeviceType>::GetBlockDimX()) +
               FunctorBase<DeviceType>::GetThreadIdX();

    rest_r_p = rest_r;
    rest_c_p = rest_c;
    rest_f_p = rest_f;

    if (nr % 2 == 0) {
      nr_p = nr + 1;
      rest_r_p = nr_p - r;
    }
    if (nc % 2 == 0) {
      nc_p = nc + 1;
      rest_c_p = nc_p - c;
    }
    if (nf % 2 == 0) {
      nf_p = nf + 1;
      rest_f_p = nf_p - f;
    }
  }

  MGARDX_EXEC void Operation2() {

    if (!w.isNull() && threadId < (R / 2) * (C / 2) * (F / 2)) {
      r_sm = (threadId / ((C / 2) * (F / 2))) * 2;
      c_sm = ((threadId % ((C / 2) * (F / 2))) / (F / 2)) * 2;
      f_sm = ((threadId % ((C / 2) * (F / 2))) % (F / 2)) * 2;
      r_gl = r / 2 + threadId / ((C / 2) * (F / 2));
      c_gl = c / 2 + threadId % ((C / 2) * (F / 2)) / (F / 2);
      f_gl = f / 2 + threadId % ((C / 2) * (F / 2)) % (F / 2);
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w_new(r_gl, c_gl, f_gl) = *w(r_gl, c_gl, f_gl);
        // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
        // printf("block: (%d %d %d) thread: (%d %d %d) load0 (%d %d %d): %f
        // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
        // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
        //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
        //                 r_gl, c_gl, f_gl);
      }
    }

    int base = 0;
    if (!w.isNull() && threadId >= base &&
        threadId < base + (C / 2) * (F / 2)) {
      r_sm = (R / 2) * 2;
      c_sm = ((threadId - base) / (F / 2)) * 2;
      f_sm = ((threadId - base) % (F / 2)) * 2;
      r_gl = r / 2 + (R / 2);
      c_gl = c / 2 + (threadId - base) / (F / 2);
      f_gl = f / 2 + (threadId - base) % (F / 2);
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w_new(r_gl, c_gl, f_gl) = *w(r_gl, c_gl, f_gl);
        // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
        // printf("block: (%d %d %d) thread: (%d %d %d) load1 (%d %d %d): %f
        // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
        // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
        //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
        //                 r_gl, c_gl, f_gl);
      }
    }
    base += (C / 2) * (F / 2); // ROUND_UP_WARP((C/2) * (F/2)) * WARP_SIZE;
    if (!w.isNull() && threadId >= base &&
        threadId < base + (R / 2) * (F / 2)) {
      r_sm = ((threadId - base) / (F / 2)) * 2;
      c_sm = (C / 2) * 2;
      f_sm = ((threadId - base) % (F / 2)) * 2;
      r_gl = r / 2 + (threadId - base) / (F / 2);
      c_gl = c / 2 + (C / 2);
      f_gl = f / 2 + (threadId - base) % (F / 2);
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w_new(r_gl, c_gl, f_gl) = *w(r_gl, c_gl, f_gl);
        // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
        // printf("block: (%d %d %d) thread: (%d %d %d) load2 (%d %d %d): %f
        // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
        // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
        //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
        //                 r_gl, c_gl, f_gl);
      }
    }
    base += (R / 2) * (F / 2); // ROUND_UP_WARP((R/2) * (F/2)) * WARP_SIZE;
    if (!w.isNull() && threadId >= base &&
        threadId < base + (R / 2) * (C / 2)) {
      r_sm = ((threadId - base) / (C / 2)) * 2;
      c_sm = ((threadId - base) % (C / 2)) * 2;
      f_sm = (F / 2) * 2;
      r_gl = r / 2 + (threadId - base) / (C / 2);
      c_gl = c / 2 + (threadId - base) % (C / 2);
      f_gl = f / 2 + (F / 2);
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w_new(r_gl, c_gl, f_gl) = *w(r_gl, c_gl, f_gl);
        // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
        // printf("block: (%d %d %d) thread: (%d %d %d) load3 (%d %d %d): %f
        // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
        // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
        //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
        //                 r_gl, c_gl, f_gl);
      }
    }
    base += (R / 2) * (C / 2); // ROUND_UP_WARP((R/2) * (C/2)) * WARP_SIZE;
    // load extra edges
    if (!w.isNull() && threadId >= base && threadId < base + (R / 2)) {
      r_sm = (threadId - base) * 2;
      c_sm = (C / 2) * 2;
      f_sm = (F / 2) * 2;
      r_gl = r / 2 + threadId - base;
      c_gl = c / 2 + (C / 2);
      f_gl = f / 2 + (F / 2);
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w_new(r_gl, c_gl, f_gl) = *w(r_gl, c_gl, f_gl);
        // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
        // printf("block: (%d %d %d) thread: (%d %d %d) load4 (%d %d %d): %f
        // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
        // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
        //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
        //                 r_gl, c_gl, f_gl);
      }
    }
    base += (R / 2); // ROUND_UP_WARP((R/2)) * WARP_SIZE;
    if (!w.isNull() && threadId >= base && threadId < base + (C / 2)) {
      r_sm = (R / 2) * 2;
      c_sm = (threadId - base) * 2;
      f_sm = (F / 2) * 2;
      r_gl = r / 2 + (R / 2);
      c_gl = c / 2 + threadId - base;
      f_gl = f / 2 + (F / 2);
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w_new(r_gl, c_gl, f_gl) = *w(r_gl, c_gl, f_gl);
        // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
        // printf("block: (%d %d %d) thread: (%d %d %d) load5 (%d %d %d): %f
        // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
        // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
        //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
        //                 r_gl, c_gl, f_gl);
      }
    }
    base += (C / 2); // ROUND_UP_WARP((C/2)) * WARP_SIZE;
    if (!w.isNull() && threadId >= base && threadId < base + (F / 2)) {
      r_sm = (R / 2) * 2;
      c_sm = (C / 2) * 2;
      f_sm = (threadId - base) * 2;
      r_gl = r / 2 + (R / 2);
      c_gl = c / 2 + (C / 2);
      f_gl = f / 2 + threadId - base;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w_new(r_gl, c_gl, f_gl) = *w(r_gl, c_gl, f_gl);
        // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
        // printf("block: (%d %d %d) thread: (%d %d %d) load6 (%d %d %d): %f
        // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
        // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
        //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
        //                 r_gl, c_gl, f_gl);
      }
    }
    base += (F / 2); // ROUND_UP_WARP((F/2)) * WARP_SIZE;
    // // load extra vertex
    if (!w.isNull() && threadId >= base && threadId < base + 1) {
      r_sm = (R / 2) * 2;
      c_sm = (C / 2) * 2;
      f_sm = (F / 2) * 2;
      r_gl = r / 2 + (R / 2);
      c_gl = c / 2 + (C / 2);
      f_gl = f / 2 + (F / 2);
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w_new(r_gl, c_gl, f_gl) = *w(r_gl, c_gl, f_gl);
        // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
        // printf("block: (%d %d %d) thread: (%d %d %d) load7 (%d %d %d): %f
        // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
        // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
        //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
        //                 r_gl, c_gl, f_gl);
      }
    }

    // __syncthreads();
    // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    //   printf("rest_p: %u %u %u RCF %u %u %u\n", rest_r_p, rest_c_p, rest_f_p,
    //   R, C, F);

    //   for (int i = 0; i < min(rest_r_p, (R/2) * 2 + 1); i++) {
    //     for (int j = 0; j < min(rest_c_p, (C/2) * 2 + 1); j++) {
    //       for (int k = 0; k < min(rest_f_p, (F/2) * 2 + 1); k++) {
    //         printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
    //       }
    //       printf("\n");
    //     }
    //     printf("\n");
    //   }
    // }
    // __syncthreads();
  }

  MGARDX_EXEC void Operation3() {
    if (!wf.isNull() && threadId >= (R / 2) * (C / 2) * (F / 2) &&
        threadId < (R / 2) * (C / 2) * (F / 2) * 2) {
      r_sm =
          ((threadId - (R / 2) * (C / 2) * (F / 2)) / ((C / 2) * (F / 2))) * 2;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2)) % ((C / 2) * (F / 2))) /
              (F / 2)) *
             2;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2)) % ((C / 2) * (F / 2))) %
              (F / 2)) *
                 2 +
             1;
      r_gl = r / 2 +
             (threadId - (R / 2) * (C / 2) * (F / 2)) / ((C / 2) * (F / 2));
      c_gl = c / 2 +
             ((threadId - (R / 2) * (C / 2) * (F / 2)) % ((C / 2) * (F / 2))) /
                 (F / 2);
      f_gl = f / 2 +
             ((threadId - (R / 2) * (C / 2) * (F / 2)) % ((C / 2) * (F / 2))) %
                 (F / 2);

      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
        res = *wf(r_gl, c_gl, f_gl);
        if (f_gl < nf_c-1) {
          bool cond1 = r_gl < nr_c-1 && c_gl < nc_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
          bool cond2 = r_gl < nr_c-1 && c_gl >= 1     && *refinement_flag(r_gl, c_gl-1, f_gl) == KEEP_CHILDREN;
          bool cond3 = r_gl >= 1     && c_gl < nc_c-1 && *refinement_flag(r_gl-1, c_gl, f_gl) == KEEP_CHILDREN;
          bool cond4 = r_gl >= 1     && c_gl >= 1     && *refinement_flag(r_gl-1, c_gl-1, f_gl) == KEEP_CHILDREN;
          if (cond1 || cond2 || cond3 || cond4) *wf_new(r_gl, c_gl, f_gl) = res;
        }
      }
    }

    if (!wc.isNull() && threadId >= (R / 2) * (C / 2) * (F / 2) * 2 &&
        threadId < (R / 2) * (C / 2) * (F / 2) * 3) {
      r_sm =
          ((threadId - (R / 2) * (C / 2) * (F / 2) * 2) / ((C / 2) * (F / 2))) *
          2;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 2) %
               ((C / 2) * (F / 2))) /
              (F / 2)) *
                 2 +
             1;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 2) %
               ((C / 2) * (F / 2))) %
              (F / 2)) *
             2;
      r_gl = r / 2 +
             (threadId - (R / 2) * (C / 2) * (F / 2) * 2) / ((C / 2) * (F / 2));
      c_gl = c / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 2) %
                      ((C / 2) * (F / 2))) /
                         (F / 2);
      f_gl = f / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 2) %
                      ((C / 2) * (F / 2))) %
                         (F / 2);
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
        res = *wc(r_gl, c_gl, f_gl);
        if (c_gl < nc_c-1) {
          bool cond1 = r_gl < nr_c-1 && f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
          bool cond2 = r_gl < nr_c-1 && f_gl >= 1     && *refinement_flag(r_gl, c_gl, f_gl-1) == KEEP_CHILDREN;
          bool cond3 = r_gl >= 1     && f_gl < nf_c-1 && *refinement_flag(r_gl-1, c_gl, f_gl) == KEEP_CHILDREN;
          bool cond4 = r_gl >= 1     && f_gl >= 1     && *refinement_flag(r_gl-1, c_gl, f_gl-1) == KEEP_CHILDREN;
          if (cond1 || cond2 || cond3 || cond4) *wc_new(r_gl, c_gl, f_gl) = res;
        }
      }
    }

    if (!wr.isNull() && threadId >= (R / 2) * (C / 2) * (F / 2) * 3 &&
        threadId < (R / 2) * (C / 2) * (F / 2) * 4) {
      r_sm =
          ((threadId - (R / 2) * (C / 2) * (F / 2) * 3) / ((C / 2) * (F / 2))) *
              2 +
          1;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 3) %
               ((C / 2) * (F / 2))) /
              (F / 2)) *
             2;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 3) %
               ((C / 2) * (F / 2))) %
              (F / 2)) *
             2;
      r_gl = r / 2 +
             (threadId - (R / 2) * (C / 2) * (F / 2) * 3) / ((C / 2) * (F / 2));
      c_gl = c / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 3) %
                      ((C / 2) * (F / 2))) /
                         (F / 2);
      f_gl = f / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 3) %
                      ((C / 2) * (F / 2))) %
                         (F / 2);

      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
        res = *wr(r_gl, c_gl, f_gl);
        if (r_gl < nr_c-1) {
          bool cond1 = c_gl < nc_c-1 && f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
          bool cond2 = c_gl < nc_c-1 && f_gl >= 1     && *refinement_flag(r_gl, c_gl, f_gl-1) == KEEP_CHILDREN;
          bool cond3 = c_gl >= 1     && f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl-1, f_gl) == KEEP_CHILDREN;
          bool cond4 = c_gl >= 1     && f_gl >= 1     && *refinement_flag(r_gl, c_gl-1, f_gl-1) == KEEP_CHILDREN;
          if (cond1 || cond2 || cond3 || cond4) *wr_new(r_gl, c_gl, f_gl) = res;
        }
      }
    }

    if (!wcf.isNull() && threadId >= (R / 2) * (C / 2) * (F / 2) * 4 &&
        threadId < (R / 2) * (C / 2) * (F / 2) * 5) {
      r_sm =
          ((threadId - (R / 2) * (C / 2) * (F / 2) * 4) / ((C / 2) * (F / 2))) *
          2;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 4) %
               ((C / 2) * (F / 2))) /
              (F / 2)) *
                 2 +
             1;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 4) %
               ((C / 2) * (F / 2))) %
              (F / 2)) *
                 2 +
             1;
      r_gl = r / 2 +
             (threadId - (R / 2) * (C / 2) * (F / 2) * 4) / ((C / 2) * (F / 2));
      c_gl = c / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 4) %
                      ((C / 2) * (F / 2))) /
                         (F / 2);
      f_gl = f / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 4) %
                      ((C / 2) * (F / 2))) %
                         (F / 2);

      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
        res = *wcf(r_gl, c_gl, f_gl);
        if (c_gl < nc_c-1 && f_gl < nf_c-1) {
          bool cond1 = r_gl < nr_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
          bool cond2 = r_gl >= 1     && *refinement_flag(r_gl-1, c_gl, f_gl) == KEEP_CHILDREN;
          if (cond1 || cond2) *wcf_new(r_gl, c_gl, f_gl) = res;
        }
      }
    }

    if (!wrf.isNull() && threadId >= (R / 2) * (C / 2) * (F / 2) * 5 &&
        threadId < (R / 2) * (C / 2) * (F / 2) * 6) {
      r_sm =
          ((threadId - (R / 2) * (C / 2) * (F / 2) * 5) / ((C / 2) * (F / 2))) *
              2 +
          1;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 5) %
               ((C / 2) * (F / 2))) /
              (F / 2)) *
             2;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 5) %
               ((C / 2) * (F / 2))) %
              (F / 2)) *
                 2 +
             1;
      r_gl = r / 2 +
             (threadId - (R / 2) * (C / 2) * (F / 2) * 5) / ((C / 2) * (F / 2));
      c_gl = c / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 5) %
                      ((C / 2) * (F / 2))) /
                         (F / 2);
      f_gl = f / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 5) %
                      ((C / 2) * (F / 2))) %
                         (F / 2);
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
        res = *wrf(r_gl, c_gl, f_gl);
        if (r_gl < nr_c-1 && f_gl < nf_c-1) {
          bool cond1 = c_gl < nc_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
          bool cond2 = c_gl >= 1     && *refinement_flag(r_gl, c_gl-1, f_gl) == KEEP_CHILDREN;
          if (cond1 || cond2) *wrf_new(r_gl, c_gl, f_gl) = res;
        }
      }
    }

    if (!wrc.isNull() && threadId >= (R / 2) * (C / 2) * (F / 2) * 6 &&
        threadId < (R / 2) * (C / 2) * (F / 2) * 7) {
      r_sm =
          ((threadId - (R / 2) * (C / 2) * (F / 2) * 6) / ((C / 2) * (F / 2))) *
              2 +
          1;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 6) %
               ((C / 2) * (F / 2))) /
              (F / 2)) *
                 2 +
             1;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 6) %
               ((C / 2) * (F / 2))) %
              (F / 2)) *
             2;
      r_gl = r / 2 +
             (threadId - (R / 2) * (C / 2) * (F / 2) * 6) / ((C / 2) * (F / 2));
      c_gl = c / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 6) %
                      ((C / 2) * (F / 2))) /
                         (F / 2);
      f_gl = f / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 6) %
                      ((C / 2) * (F / 2))) %
                         (F / 2);
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
        res = *wrc(r_gl, c_gl, f_gl);
        if (r_gl < nr_c-1 && c_gl < nc_c-1) {
          bool cond1 = f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
          bool cond2 = f_gl >= 1     && *refinement_flag(r_gl, c_gl, f_gl-1) == KEEP_CHILDREN;
          if (cond1 || cond2) *wrc_new(r_gl, c_gl, f_gl) = res;
        }
      }
    }

    if (!wrcf.isNull() && threadId >= (R / 2) * (C / 2) * (F / 2) * 7 &&
        threadId < (R / 2) * (C / 2) * (F / 2) * 8) {
      r_sm =
          ((threadId - (R / 2) * (C / 2) * (F / 2) * 7) / ((C / 2) * (F / 2))) *
              2 +
          1;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 7) %
               ((C / 2) * (F / 2))) /
              (F / 2)) *
                 2 +
             1;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 7) %
               ((C / 2) * (F / 2))) %
              (F / 2)) *
                 2 +
             1;
      r_gl = r / 2 +
             (threadId - (R / 2) * (C / 2) * (F / 2) * 7) / ((C / 2) * (F / 2));
      c_gl = c / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 7) %
                      ((C / 2) * (F / 2))) /
                         (F / 2);
      f_gl = f / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 7) %
                      ((C / 2) * (F / 2))) %
                         (F / 2);
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
        res = *wrcf(r_gl, c_gl, f_gl);
        if (r_gl < nr_c-1 && c_gl < nc_c-1 && f_gl < nf_c-1) {
          bool cond1 = *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
          if (cond1) *wrcf_new(r_gl, c_gl, f_gl) = res;
        }
      }
    }

    if (r + (R / 2) * 2 == nr_p - 1) {
      if (threadId < (C / 2) * (F / 2)) {
        if (!wf.isNull()) {
          r_sm = (R / 2) * 2;
          c_sm = (threadId / (F / 2)) * 2;
          f_sm = (threadId % (F / 2)) * 2 + 1;
          r_gl = r / 2 + (R / 2);
          c_gl = c / 2 + threadId / (F / 2);
          f_gl = f / 2 + threadId % (F / 2);
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
            res = *wf(r_gl, c_gl, f_gl);
            if (f_gl < nf_c-1) {
              bool cond1 = r_gl < nr_c-1 && c_gl < nc_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond2 = r_gl < nr_c-1 && c_gl >= 1     && *refinement_flag(r_gl, c_gl-1, f_gl) == KEEP_CHILDREN;
              bool cond3 = r_gl >= 1     && c_gl < nc_c-1 && *refinement_flag(r_gl-1, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond4 = r_gl >= 1     && c_gl >= 1     && *refinement_flag(r_gl-1, c_gl-1, f_gl) == KEEP_CHILDREN;
              if (cond1 || cond2 || cond3 || cond4) *wf_new(r_gl, c_gl, f_gl) = res;
            }
          }
        }

        if (!wc.isNull()) {
          r_sm = (R / 2) * 2;
          c_sm = (threadId / (F / 2)) * 2 + 1;
          f_sm = (threadId % (F / 2)) * 2;
          r_gl = r / 2 + (R / 2);
          c_gl = c / 2 + threadId / (F / 2);
          f_gl = f / 2 + threadId % (F / 2);
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
            res = *wc(r_gl, c_gl, f_gl);
            if (c_gl < nc_c-1) {
              bool cond1 = r_gl < nr_c-1 && f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond2 = r_gl < nr_c-1 && f_gl >= 1     && *refinement_flag(r_gl, c_gl, f_gl-1) == KEEP_CHILDREN;
              bool cond3 = r_gl >= 1     && f_gl < nf_c-1 && *refinement_flag(r_gl-1, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond4 = r_gl >= 1     && f_gl >= 1     && *refinement_flag(r_gl-1, c_gl, f_gl-1) == KEEP_CHILDREN;
              if (cond1 || cond2 || cond3 || cond4) *wc_new(r_gl, c_gl, f_gl) = res;
            }
          }
        }
        if (!wcf.isNull()) {
          r_sm = (R / 2) * 2;
          c_sm = (threadId / (F / 2)) * 2 + 1;
          f_sm = (threadId % (F / 2)) * 2 + 1;
          r_gl = r / 2 + (R / 2);
          c_gl = c / 2 + threadId / (F / 2);
          f_gl = f / 2 + threadId % (F / 2);
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
            res = *wcf(r_gl, c_gl, f_gl);
            if (c_gl < nc_c-1 && f_gl < nf_c-1) {
              bool cond1 = r_gl < nr_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond2 = r_gl >= 1     && *refinement_flag(r_gl-1, c_gl, f_gl) == KEEP_CHILDREN;
              if (cond1 || cond2) *wcf_new(r_gl, c_gl, f_gl) = res;
            }
          }
        }
      }
    }

    if (c + (C / 2) * 2 == nc_p - 1) {
      if (threadId >= (R / 2) * (C / 2) * (F / 2) &&
          threadId < (R / 2) * (C / 2) * (F / 2) + (R / 2) * (F / 2)) {
        if (!wf.isNull()) {
          r_sm = ((threadId - (R / 2) * (C / 2) * (F / 2)) / (F / 2)) * 2;
          c_sm = (C / 2) * 2;
          f_sm = ((threadId - (R / 2) * (C / 2) * (F / 2)) % (F / 2)) * 2 + 1;
          r_gl = r / 2 + (threadId - (R / 2) * (C / 2) * (F / 2)) / (F / 2);
          c_gl = c / 2 + (C / 2);
          f_gl = f / 2 + (threadId - (R / 2) * (C / 2) * (F / 2)) % (F / 2);
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
            res = *wf(r_gl, c_gl, f_gl);
            if (f_gl < nf_c-1) {
              bool cond1 = r_gl < nr_c-1 && c_gl < nc_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond2 = r_gl < nr_c-1 && c_gl >= 1     && *refinement_flag(r_gl, c_gl-1, f_gl) == KEEP_CHILDREN;
              bool cond3 = r_gl >= 1     && c_gl < nc_c-1 && *refinement_flag(r_gl-1, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond4 = r_gl >= 1     && c_gl >= 1     && *refinement_flag(r_gl-1, c_gl-1, f_gl) == KEEP_CHILDREN;
              if (cond1 || cond2 || cond3 || cond4) *wf_new(r_gl, c_gl, f_gl) = res;
            }
          }
        }
        if (!wr.isNull()) {
          r_sm = ((threadId - (R / 2) * (C / 2) * (F / 2)) / (F / 2)) * 2 + 1;
          c_sm = (C / 2) * 2;
          f_sm = ((threadId - (R / 2) * (C / 2) * (F / 2)) % (F / 2)) * 2;
          r_gl = r / 2 + (threadId - (R / 2) * (C / 2) * (F / 2)) / (F / 2);
          c_gl = c / 2 + (C / 2);
          f_gl = f / 2 + (threadId - (R / 2) * (C / 2) * (F / 2)) % (F / 2);
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
            res = *wr(r_gl, c_gl, f_gl);
            if (r_gl < nr_c-1) {
              bool cond1 = c_gl < nc_c-1 && f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond2 = c_gl < nc_c-1 && f_gl >= 1     && *refinement_flag(r_gl, c_gl, f_gl-1) == KEEP_CHILDREN;
              bool cond3 = c_gl >= 1     && f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl-1, f_gl) == KEEP_CHILDREN;
              bool cond4 = c_gl >= 1     && f_gl >= 1     && *refinement_flag(r_gl, c_gl-1, f_gl-1) == KEEP_CHILDREN;
              if (cond1 || cond2 || cond3 || cond4) *wr_new(r_gl, c_gl, f_gl) = res;
            }
          }
        }
        if (!wrf.isNull()) {
          r_sm = ((threadId - (R / 2) * (C / 2) * (F / 2)) / (F / 2)) * 2 + 1;
          c_sm = (C / 2) * 2;
          f_sm = ((threadId - (R / 2) * (C / 2) * (F / 2)) % (F / 2)) * 2 + 1;
          r_gl = r / 2 + (threadId - (R / 2) * (C / 2) * (F / 2)) / (F / 2);
          c_gl = c / 2 + (C / 2);
          f_gl = f / 2 + (threadId - (R / 2) * (C / 2) * (F / 2)) % (F / 2);
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
            res = *wrf(r_gl, c_gl, f_gl);
            if (r_gl < nr_c-1 && f_gl < nf_c-1) {
              bool cond1 = c_gl < nc_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond2 = c_gl >= 1     && *refinement_flag(r_gl, c_gl-1, f_gl) == KEEP_CHILDREN;
              if (cond1 || cond2) *wrf_new(r_gl, c_gl, f_gl) = res;
            }
          }
        }
      }
    }

    if (f + (F / 2) * 2 == nf_p - 1) {
      if (threadId >= (R / 2) * (C / 2) * (F / 2) * 2 &&
          threadId < (R / 2) * (C / 2) * (F / 2) * 2 + (R / 2) * (C / 2)) {
        if (!wc.isNull()) {
          r_sm = ((threadId - (R / 2) * (C / 2) * (F / 2) * 2) / (C / 2)) * 2;
          c_sm =
              ((threadId - (R / 2) * (C / 2) * (F / 2) * 2) % (C / 2)) * 2 + 1;
          f_sm = (F / 2) * 2;
          r_gl = r / 2 + (threadId - (R / 2) * (C / 2) * (F / 2) * 2) / (C / 2);
          c_gl = c / 2 + (threadId - (R / 2) * (C / 2) * (F / 2) * 2) % (C / 2);
          f_gl = f / 2 + (F / 2);
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
            res = *wc(r_gl, c_gl, f_gl);
            if (c_gl < nc_c-1) {
              bool cond1 = r_gl < nr_c-1 && f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond2 = r_gl < nr_c-1 && f_gl >= 1     && *refinement_flag(r_gl, c_gl, f_gl-1) == KEEP_CHILDREN;
              bool cond3 = r_gl >= 1     && f_gl < nf_c-1 && *refinement_flag(r_gl-1, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond4 = r_gl >= 1     && f_gl >= 1     && *refinement_flag(r_gl-1, c_gl, f_gl-1) == KEEP_CHILDREN;
              if (cond1 || cond2 || cond3 || cond4) *wc_new(r_gl, c_gl, f_gl) = res;
            }
          }
        }

        if (!wr.isNull()) {
          r_sm =
              ((threadId - (R / 2) * (C / 2) * (F / 2) * 2) / (C / 2)) * 2 + 1;
          c_sm = ((threadId - (R / 2) * (C / 2) * (F / 2) * 2) % (C / 2)) * 2;
          f_sm = (F / 2) * 2;
          r_gl = r / 2 + (threadId - (R / 2) * (C / 2) * (F / 2) * 2) / (C / 2);
          c_gl = c / 2 + (threadId - (R / 2) * (C / 2) * (F / 2) * 2) % (C / 2);
          f_gl = f / 2 + (F / 2);
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
            res = *wr(r_gl, c_gl, f_gl);
            if (r_gl < nr_c-1) {
              bool cond1 = c_gl < nc_c-1 && f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond2 = c_gl < nc_c-1 && f_gl >= 1     && *refinement_flag(r_gl, c_gl, f_gl-1) == KEEP_CHILDREN;
              bool cond3 = c_gl >= 1     && f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl-1, f_gl) == KEEP_CHILDREN;
              bool cond4 = c_gl >= 1     && f_gl >= 1     && *refinement_flag(r_gl, c_gl-1, f_gl-1) == KEEP_CHILDREN;
              if (cond1 || cond2 || cond3 || cond4) *wr_new(r_gl, c_gl, f_gl) = res;
            }
          }
        }

        if (!wrc.isNull()) {
          r_sm =
              ((threadId - (R / 2) * (C / 2) * (F / 2) * 2) / (C / 2)) * 2 + 1;
          c_sm =
              ((threadId - (R / 2) * (C / 2) * (F / 2) * 2) % (C / 2)) * 2 + 1;
          f_sm = (F / 2) * 2;
          r_gl = r / 2 + (threadId - (R / 2) * (C / 2) * (F / 2) * 2) / (C / 2);
          c_gl = c / 2 + (threadId - (R / 2) * (C / 2) * (F / 2) * 2) % (C / 2);
          f_gl = f / 2 + (F / 2);
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
            res = *wrc(r_gl, c_gl, f_gl);
            if (r_gl < nr_c-1 && c_gl < nc_c-1) {
              bool cond1 = f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond2 = f_gl >= 1     && *refinement_flag(r_gl, c_gl, f_gl-1) == KEEP_CHILDREN;
              if (cond1 || cond2) *wrc_new(r_gl, c_gl, f_gl) = res;
            }
          }
        }
      }
    }

    if (c + (C / 2) * 2 == nc_p - 1 && f + (F / 2) * 2 == nf_p - 1) {
      if (threadId >= (R / 2) * (C / 2) * (F / 2) * 3 &&
          threadId < (R / 2) * (C / 2) * (F / 2) * 3 + (R / 2)) {
        if (!wr.isNull()) {
          r_sm = (threadId - (R / 2) * (C / 2) * (F / 2) * 3) * 2 + 1;
          c_sm = (C / 2) * 2;
          f_sm = (F / 2) * 2;
          r_gl = r / 2 + threadId - (R / 2) * (C / 2) * (F / 2) * 3;
          c_gl = c / 2 + (C / 2);
          f_gl = f / 2 + (F / 2);
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
            res = *wr(r_gl, c_gl, f_gl);
            if (r_gl < nr_c-1) {
              bool cond1 = c_gl < nc_c-1 && f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond2 = c_gl < nc_c-1 && f_gl >= 1     && *refinement_flag(r_gl, c_gl, f_gl-1) == KEEP_CHILDREN;
              bool cond3 = c_gl >= 1     && f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl-1, f_gl) == KEEP_CHILDREN;
              bool cond4 = c_gl >= 1     && f_gl >= 1     && *refinement_flag(r_gl, c_gl-1, f_gl-1) == KEEP_CHILDREN;
              if (cond1 || cond2 || cond3 || cond4) *wr_new(r_gl, c_gl, f_gl) = res;
            }
          }
        }
      }
    }

    if (r + (R / 2) * 2 == nr_p - 1 && f + (F / 2) * 2 == nf_p - 1) {
      if (threadId >= (R / 2) * (C / 2) * (F / 2) * 4 &&
          threadId < (R / 2) * (C / 2) * (F / 2) * 4 + (C / 2)) {
        if (!wc.isNull()) {
          r_sm = (R / 2) * 2;
          c_sm = (threadId - (R / 2) * (C / 2) * (F / 2) * 4) * 2 + 1;
          f_sm = (F / 2) * 2;
          r_gl = r / 2 + (R / 2);
          c_gl = c / 2 + threadId - (R / 2) * (C / 2) * (F / 2) * 4;
          f_gl = f / 2 + (F / 2);
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
            res = *wc(r_gl, c_gl, f_gl);
            if (c_gl < nc_c-1) {
              bool cond1 = r_gl < nr_c-1 && f_gl < nf_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond2 = r_gl < nr_c-1 && f_gl >= 1     && *refinement_flag(r_gl, c_gl, f_gl-1) == KEEP_CHILDREN;
              bool cond3 = r_gl >= 1     && f_gl < nf_c-1 && *refinement_flag(r_gl-1, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond4 = r_gl >= 1     && f_gl >= 1     && *refinement_flag(r_gl-1, c_gl, f_gl-1) == KEEP_CHILDREN;
              if (cond1 || cond2 || cond3 || cond4) *wc_new(r_gl, c_gl, f_gl) = res;
            }
          }
        }
      }
    }

    if (r + (R / 2) * 2 == nr_p - 1 && c + (C / 2) * 2 == nc_p - 1) {
      if (threadId >= (R / 2) * (C / 2) * (F / 2) * 5 &&
          threadId < (R / 2) * (C / 2) * (F / 2) * 5 + (F / 2)) {
        if (!wf.isNull()) {
          r_sm = (R / 2) * 2;
          c_sm = (C / 2) * 2;
          f_sm = (threadId - (R / 2) * (C / 2) * (F / 2) * 5) * 2 + 1;
          r_gl = r / 2 + (R / 2);
          c_gl = c / 2 + (C / 2);
          f_gl = f / 2 + threadId - (R / 2) * (C / 2) * (F / 2) * 5;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
            res = *wf(r_gl, c_gl, f_gl);
            if (f_gl < nf_c-1) {
              bool cond1 = r_gl < nr_c-1 && c_gl < nc_c-1 && *refinement_flag(r_gl, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond2 = r_gl < nr_c-1 && c_gl >= 1     && *refinement_flag(r_gl, c_gl-1, f_gl) == KEEP_CHILDREN;
              bool cond3 = r_gl >= 1     && c_gl < nc_c-1 && *refinement_flag(r_gl-1, c_gl, f_gl) == KEEP_CHILDREN;
              bool cond4 = r_gl >= 1     && c_gl >= 1     && *refinement_flag(r_gl-1, c_gl-1, f_gl) == KEEP_CHILDREN;
              if (cond1 || cond2 || cond3 || cond4) *wf_new(r_gl, c_gl, f_gl) = res;
            }
          }
        }
      }
    }

    // __syncthreads();
    // if (debug) {
    //   printf("TYPE: %d %d %d %d\n", TYPE,
    //           min(rest_r_p, (R/2) * 2 + 1),
    //           min(rest_c_p, (C/2) * 2 + 1),
    //           min(rest_f_p, (F/2) * 2 + 1));
    //   for (int i = 0; i < min(rest_r_p, (R/2) * 2 + 1); i++) {
    //     for (int j = 0; j < min(rest_c_p, (C/2) * 2 + 1); j++) {
    //       for (int k = 0; k < min(rest_f_p, (F/2) * 2 + 1); k++) {
    //         printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
    //       }
    //       printf("\n");
    //     }
    //     printf("\n");
    //   }
    // }
    // __syncthreads();
  }

  MGARDX_EXEC void Operation4() {

  }

  MGARDX_EXEC void Operation5() {
  }

private:
  // functor parameters
  SIZE nr, nc, nf, nr_c, nc_c, nf_c;
  SubArray<D, T, DeviceType> w, wf, wc, wr, wcf, wrf, wrc, wrcf;
  SubArray<D, SIZE, DeviceType> refinement_flag;
  SubArray<D, T, DeviceType> w_new, wf_new, wc_new, wr_new, wcf_new, wrf_new, wrc_new, wrcf_new;

  // thread local variables
  SIZE r, c, f;
  SIZE rest_r, rest_c, rest_f;
  SIZE nr_p, nc_p, nf_p;
  SIZE rest_r_p, rest_c_p, rest_f_p;
  SIZE r_sm, c_sm, f_sm;
  SIZE r_sm_ex, c_sm_ex, f_sm_ex;
  SIZE r_gl, c_gl, f_gl;
  SIZE r_gl_ex, c_gl_ex, f_gl_ex;
  LENGTH threadId;
  T res;
  T *sm;
  SIZE ldsm1;
  SIZE ldsm2;
  T *v_sm;
  T *ratio_f_sm;
  T *ratio_c_sm;
  T *ratio_r_sm;
};

template <DIM D, typename T, typename DeviceType>
class CoefficientRetrieverKernel : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  CoefficientRetrieverKernel() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<CoefficientRetrieverFunctor<D, T, R, C, F, DeviceType>> GenTask(
      SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
      SubArray<D, T, DeviceType> w, SubArray<D, T, DeviceType> wf,
      SubArray<D, T, DeviceType> wc, SubArray<D, T, DeviceType> wr,
      SubArray<D, T, DeviceType> wcf, SubArray<D, T, DeviceType> wrf,
      SubArray<D, T, DeviceType> wrc, SubArray<D, T, DeviceType> wrcf,
      SubArray<D, SIZE, DeviceType> refinement_flag,
      SubArray<D, T, DeviceType> w_new, SubArray<D, T, DeviceType> wf_new,
      SubArray<D, T, DeviceType> wc_new, SubArray<D, T, DeviceType> wr_new,
      SubArray<D, T, DeviceType> wcf_new, SubArray<D, T, DeviceType> wrf_new,
      SubArray<D, T, DeviceType> wrc_new, SubArray<D, T, DeviceType> wrcf_new,
      int queue_idx) {
    using FunctorType = CoefficientRetrieverFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(nr, nc, nf, nr_c, nc_c, nf_c,
                        w, wf, wc, wr, wcf, wrf, wrc, wrcf,
                        refinement_flag,
                        w_new, wf_new, wc_new, wr_new, wcf_new, 
                        wrf_new, wrc_new, wrcf_new);

    SIZE total_thread_z = std::max(nr - 1, (SIZE)1);
    SIZE total_thread_y = std::max(nc - 1, (SIZE)1);
    SIZE total_thread_x = std::max(nf - 1, (SIZE)1);
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size;
    tbz = R;
    tby = C;
    tbx = F;
    sm_size = ((R + 1) * (C + 1) * (F + 1) + R + C + F) * sizeof(T);
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "CoefficientRetriever");
  }

  MGARDX_CONT
  void Execute(SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
              SubArray<D, T, DeviceType> w, SubArray<D, T, DeviceType> wf,
              SubArray<D, T, DeviceType> wc, SubArray<D, T, DeviceType> wr,
              SubArray<D, T, DeviceType> wcf, SubArray<D, T, DeviceType> wrf,
              SubArray<D, T, DeviceType> wrc, SubArray<D, T, DeviceType> wrcf,
              SubArray<D, SIZE, DeviceType> refinement_flag,
              SubArray<D, T, DeviceType> w_new, SubArray<D, T, DeviceType> wf_new,
              SubArray<D, T, DeviceType> wc_new, SubArray<D, T, DeviceType> wr_new,
              SubArray<D, T, DeviceType> wcf_new, SubArray<D, T, DeviceType> wrf_new,
              SubArray<D, T, DeviceType> wrc_new, SubArray<D, T, DeviceType> wrcf_new,
               int queue_idx) {
    int range_l = std::min(6, (int)std::log2(nf) - 1);
    int prec = TypeToIdx<T>();
    int config =
        AutoTuner<DeviceType>::autoTuningTable.gpk_rev_3d[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define GPK(CONFIG)                                                            \
    if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
      const int R = GPK_CONFIG[D - 1][CONFIG][0];                                \
      const int C = GPK_CONFIG[D - 1][CONFIG][1];                                \
      const int F = GPK_CONFIG[D - 1][CONFIG][2];                                \
      using FunctorType = CoefficientRetrieverFunctor<D, T, R, C, F, DeviceType>;\
      using TaskType = Task<FunctorType>;                                        \
      TaskType task = GenTask<R, C, F>(                                          \
          nr, nc, nf, nr_c, nc_c, nf_c, wf, wc,                                  \
          w, wr, wcf, wrf, wrc, wrcf, refinement_flag, w_new, wf_new, wc_new, wr_new, wcf_new, \
                        wrf_new, wrc_new, wrcf_new, queue_idx);                  \
      DeviceAdapter<TaskType, DeviceType> adapter;                               \
      ret = adapter.Execute(task);                                               \
      if (AutoTuner<DeviceType>::ProfileKernels) {                               \
        if (ret.success && min_time > ret.execution_time) {                      \
          min_time = ret.execution_time;                                         \
          min_config = CONFIG;                                                   \
        }                                                                        \
      }                                                                          \
    }

    GPK(6) if (!ret.success) config--;
    GPK(5) if (!ret.success) config--;
    GPK(4) if (!ret.success) config--;
    GPK(3) if (!ret.success) config--;
    GPK(2) if (!ret.success) config--;
    GPK(1) if (!ret.success) config--;
    GPK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err << "no suitable config for CoefficientRetriever.\n";
      exit(-1);
    }
#undef GPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("gpk_rev_3d", prec, range_l, min_config);
    }
  }
};

}

#endif