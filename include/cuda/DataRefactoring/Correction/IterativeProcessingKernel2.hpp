/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#ifndef MGARD_X_ITERATIVE_PROCESSING_KERNEL2_TEMPLATE
#define MGARD_X_ITERATIVE_PROCESSING_KERNEL2_TEMPLATE

#include "../../CommonInternal.h"
#include "IPKFunctor.h"
#include "IterativeProcessingKernel3D.h"

#include "../../Functor.h"
#include "../../AutoTuners/AutoTuner.h"
#include "../../Task.h"
#include "../../DeviceAdapters/DeviceAdapter.h"

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, SIZE G, typename DeviceType>
class Ipk1ReoFunctor: public IterFunctor<DeviceType> {
  public:
  MGARDm_CONT Ipk1ReoFunctor(SubArray<1, SIZE, DeviceType> shape, 
                             SubArray<1, SIZE, DeviceType> shape_c,
                             DIM processed_n, 
                             SubArray<1, DIM, DeviceType> processed_dims, 
                             DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
                             SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm, 
                             SubArray<D, T, DeviceType> v):
                             shape(shape), shape_c(shape_c), 
                             processed_n(processed_n), processed_dims(processed_dims),
                             curr_dim_r(curr_dim_r), curr_dim_c(curr_dim_c), curr_dim_f(curr_dim_f),
                             am(am), bm(bm), v(v) {
    Functor<DeviceType>();                            
  }

  MGARDm_EXEC void
  Operation1() {
    // bool debug = false;
    // if (blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 &&
    //     FunctorBase<DeviceType>::GetThreadIdZ() == 0 && FunctorBase<DeviceType>::GetThreadIdY() == 0)
    //   debug = false;

    // bool debug2 = false;
    // if (FunctorBase<DeviceType>::GetThreadIdZ() == 0 && FunctorBase<DeviceType>::GetThreadIdY() == 0 && FunctorBase<DeviceType>::GetThreadIdX() == 0)
    //   debug2 = false;

    LENGTH threadId = (FunctorBase<DeviceType>::GetThreadIdZ() * (FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetBlockDimY())) +
                      (FunctorBase<DeviceType>::GetThreadIdY() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();

    T * sm = (T*)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F + G;
    ldsm2 = C;
    vec_sm = sm; sm += R * ldsm1 * ldsm2;
    am_sm = sm; sm += ldsm1;
    bm_sm = sm; sm += ldsm1;

    SIZE * sm_size = (SIZE*)sm;
    shape_sm = sm_size; sm_size += D;
    shape_c_sm = sm_size; sm_size += D;
    // SIZE *ldvs_sm = sm_size; sm_size += D;
    // SIZE *ldws_sm = sm_size; sm_size += D;
    sm = (T*)sm_size;

    DIM * sm_dim = (DIM*)sm;
    processed_dims_sm = sm_dim; sm_dim += D;
    sm = (T*)sm_dim;

    // SIZE idx[D];

    for (LENGTH i = threadId; i < D; i += FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetBlockDimY() * blockDim.z) {
      shape_sm[i] = *shape(i);
      shape_c_sm[i] = *shape_c(i);
      // ldvs_sm[i] = ldvs[i];
      // ldws_sm[i] = ldws[i];
    }
    for (LENGTH i = threadId; i < processed_n; i += FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetBlockDimY() * blockDim.z) {
      processed_dims_sm[i] = *processed_dims(i);
    }

  }

  MGARDm_EXEC void
  Operation2() {
    // __syncthreads();

    for (DIM d = 0; d < D; d++)
      idx[d] = 0;

    nr = shape_c_sm[curr_dim_r];
    nc = shape_c_sm[curr_dim_c];
    nf = shape_c_sm[curr_dim_f];

    if (D < 3)
      nr = 1;
    if (D < 2)
      nc = 1;

    SIZE bidx = blockIdx.x;
    SIZE firstD = div_roundup(nc, C);
    SIZE blockId = bidx % firstD;
    // if (debug2) {
    //   printf("blockIdx.x %u nc %u FunctorBase<DeviceType>::GetBlockDimX() %u firstD: %u blockId %u\n", blockIdx.x, nc, FunctorBase<DeviceType>::GetBlockDimX(), firstD, blockId);
    // }
    bidx /= firstD;

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
        SIZE t = shape_sm[d];
        //for (DIM k = 0; k < processed_n; k++) {
        //  if (d == processed_dims[k]) {
            t = shape_c_sm[d];
        //  }
        //}
        // if (debug2) {
        //   printf("%u mod %u = %u, %u / %u = %u (shape_c: %u %u %u %u %u)\n", bidx, t, bidx % t, bidx, t, bidx/t, shape_c_sm[4], shape_c_sm[3],shape_c_sm[2],shape_c_sm[1],shape_c_sm[0]);
        // }
        idx[d] = bidx % t;
        bidx /= t;
      }
    }

    v.offset(idx);
    // size_t other_offset_v = get_idx<D>(ldvs_sm, idx);
    // v = v + other_offset_v;


    // if (debug2) {
    //   printf("ipk1 idx: %u %u %u %u %u ld: %u %u %u %u %u\n", idx[4], idx[3], idx[2], idx[1], idx[0], ldvs_sm[4], ldvs_sm[3], ldvs_sm[2], ldvs_sm[1], ldvs_sm[0]);
    //   printf("ipk1 other_offset_v: %llu\n", other_offset_v);

    //   LENGTH curr_stride = 1;
    //   LENGTH ret_idx = 0;
    //   for (DIM i = 0; i < D; i++) {
    //     ret_idx += idx[i] * curr_stride;
    //     printf("%llu * %llu = %llu\n", curr_stride, ldvs_sm[i], curr_stride*ldvs_sm[i]);
    //     curr_stride *= ldvs_sm[i];

    //   }
    // }

    c_gl = blockId * C;
    r_gl = blockIdx.y * R;
    f_gl = FunctorBase<DeviceType>::GetThreadIdX();

    c_sm = FunctorBase<DeviceType>::GetThreadIdX();
    r_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();

    // if (idx[3] == 0 && idx[4] == 1 && r_gl+r_sm == 0 && c_gl+c_sm == 0) {
    //   printf("offset = %llu\n", other_offset_v);
    //   debug2 = false;
    // }

    // if (debug2) {
    //   printf("ld: (%d %d %d %d %d) (shape_c: %u %u %u %u %u)\n", 
    //     ldvs_sm[4], ldvs_sm[3],ldvs_sm[2],ldvs_sm[1],ldvs_sm[0],
    //     shape_c_sm[4], shape_c_sm[3],shape_c_sm[2],shape_c_sm[1],shape_c_sm[0]);
    // }

    // T *vec = v + get_idx(ldv1, ldv2, r_gl, c_gl, 0);

    v.offset(r_gl, c_gl, 0);

    prev_vec_sm = 0.0;

    c_rest = min(C, nc - blockId * C);
    r_rest = min(R, nr - blockIdx.y * R);

    f_rest = nf;
    f_ghost = min(nf, G);
    f_main = F;

    // printf("r_sm: %d, r_rest: %d, c_sm: %d, c_rest: %d f_sm: %d, f_rest %d ,
    // nf: %d\n", r_sm, r_rest, c_sm, c_rest, f_sm, f_rest, nf);

    // printf("test %f", vec_sm[get_idx(ldsm1, ldsm2, 0, 1, 0)]);
    /* Load first ghost */
    if (r_sm < r_rest && f_sm < f_ghost) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
            *v(r_sm, i, f_gl);
        // if (r_sm == 0) printf("r0_stride = %d, vec_sm[%d] = %f\n", r0_stride,
        // i, vec_sm[i * ldsm + c_sm]);
      } 
    }

    if (r_sm == 0 && f_sm < f_ghost) {
      am_sm[f_sm] = *am(f_gl);
      bm_sm[f_sm] = *bm(f_gl);
    }

    f_rest -= f_ghost;

  }
    // __syncthreads();

  MGARDm_EXEC bool
  LoopCondition1() {
    return f_rest > F - f_ghost;
  }

  MGARDm_EXEC void
  Operation3() {
    //while (f_rest > F - f_ghost) {
    // if (c_gl == 0 && c_sm == 0 && r_gl == 0 && r_sm == 0) printf("%d %d\n",
    // f_rest, F - f_ghost);
    f_main = min(F, f_rest);
    if (r_sm < r_rest && f_sm < f_main) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + f_ghost)] =
            *v(r_sm, i, f_gl + f_ghost);
      }
    }
    if (r_sm == 0 && f_sm < f_main) {
      am_sm[f_sm + f_ghost] = *am(f_gl + f_ghost);
      bm_sm[f_sm + f_ghost] = *bm(f_gl + f_ghost);
    }
  }

      // __syncthreads();
  MGARDm_EXEC void
  Operation4() {

    /* Computation of v in parallel*/
    if (r_sm < r_rest && c_sm < c_rest) {
      // if (debug) printf("forward %f <- %f %f %f %f\n",
      //             tridiag_forward2(
      //     prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]),
      //             prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);

      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = tridiag_forward2(
          prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);

      //#pragma unroll 32
      for (SIZE i = 1; i < F; i++) {
        // if (debug) printf("forward %f <- %f %f %f %f\n",
        //           tridiag_forward2(
        //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i], bm_sm[i],
        //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]),
        //           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i], bm_sm[i],
        //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);

        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = tridiag_forward2(
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i], bm_sm[i],
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);

      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, F - 1)];
    }
  }
      // __syncthreads();

  MGARDm_EXEC void
  Operation5() {

    /* flush results to v */
    if (r_sm < r_rest && f_sm < F) {
      for (SIZE i = 0; i < c_rest; i++) {
        // if (blockIdx.x == 0 && blockIdx.y == 0 && r_sm == 0 && i == 1) {
        //   printf("store [%d %d %d] %f<-%f [%d %d %d]\n",
        //     r_sm, i, f_gl, *v(r_sm, i, f_gl),
        //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)], r_sm, i, f_sm);
        // }
        *v(r_sm, i, f_gl) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
        // if (blockIdx.x == 0 && blockIdx.y == 0 && r_sm == 0 && i == 1) {
        //   printf("store [%d %d %d] %f<-%f [%d %d %d]\n",
        //     r_sm, i, f_gl, *v(r_sm, i, f_gl),
        //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)], r_sm, i, f_sm);
        // }
      }
    }

  }


     // __syncthreads();

  MGARDm_EXEC void
  Operation6() {

    /* Update unloaded col */
    f_rest -= f_main;

    /* Advance c */
    f_gl += F;

    /* Copy next ghost to main */
    f_ghost = min(G, f_main - (F - G));
    if (r_sm < r_rest && f_sm < f_ghost) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + F)];
      }
    }
    if (r_sm == 0 && f_sm < f_ghost) {
      am_sm[f_sm] = am_sm[f_sm + FunctorBase<DeviceType>::GetBlockDimX()];
      bm_sm[f_sm] = bm_sm[f_sm + FunctorBase<DeviceType>::GetBlockDimX()];
    }
  }
      // __syncthreads();
   // } // end of while
  MGARDm_EXEC void
  Operation7() {
    /* Load all rest col */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + f_ghost)] =
            *v(r_sm, i, f_gl + f_ghost);
      }
      // if (r_sm == 0) {
      //   bm_sm[f_sm + f_ghost] = *bm(f_gl + f_ghost);
      // }
    }
    if (r_sm == 0 && f_sm < f_rest) {
      am_sm[f_sm + f_ghost] = *am(f_gl + f_ghost);
      bm_sm[f_sm + f_ghost] = *bm(f_gl + f_ghost);
    }
  }

    // __syncthreads();

  MGARDm_EXEC void
  Operation8() {

    /* Only 1 col remain */
    if (f_ghost + f_rest == 1) {
      if (r_sm < r_rest && c_sm < c_rest) {
        // if (debug) printf("forward %f <- %f %f %f %f\n",
        //             tridiag_forward2(
        //     prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]),
        //             prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = tridiag_forward2(
            prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
        // printf ("prev_vec_sm = %f\n", prev_vec_sm );
        // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
      }
      //__syncthreads();

    } else {
      if (r_sm < r_rest && c_sm < c_rest) {
        // if (debug) printf("forward %f <- %f %f %f %f\n",
        //             tridiag_forward2(
        //     prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]),
        //             prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = tridiag_forward2(
            prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
        for (SIZE i = 1; i < f_ghost + f_rest; i++) {
          // if (debug) printf("forward %f <- %f %f %f %f\n",
          //           tridiag_forward2(
          //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i], bm_sm[i],
          //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]),
          //           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i], bm_sm[i],
          //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = tridiag_forward2(
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i], bm_sm[i],
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
        }
      }
    }

  }

  MGARDm_EXEC void
  Operation9() {
    // __syncthreads();
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_ghost + f_rest) {
      for (SIZE i = 0; i < c_rest; i++) {
        *v(r_sm, i, f_gl) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
        // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
        // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
        // c_stride, vec[i * row_stride * lddv + c_stride]);
      }
    }

  }
    // __syncthreads();

  MGARDm_EXEC void
  Operation10() {
    /* backward */
    f_rest = nf;
    f_ghost = min(nf, G);
    f_main = F;
    f_gl = FunctorBase<DeviceType>::GetThreadIdX();
    prev_vec_sm = 0.0;

    /* Load first ghost */
    if (r_sm < r_rest && f_sm < f_ghost) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
            *v(r_sm, i, (nf - 1) - f_gl);
        // if (r_sm == 0) printf("r0_stride = %d, vec_sm[%d] = %f\n", r0_stride,
        // i, vec_sm[i * ldsm + c_sm]);
      }
    }
    if (r_sm == 0 && f_sm < f_ghost) {
      am_sm[f_sm] = *am(nf - f_gl);
      bm_sm[f_sm] = *bm(nf - f_gl);
    }
    f_rest -= f_ghost;

  }
    // __syncthreads();
  MGARDm_EXEC bool
  LoopCondition2() {
    return f_rest > F - f_ghost;
  }

  MGARDm_EXEC void
  Operation11() {
  //while (f_rest > F - f_ghost) {
    f_main = min(F, f_rest);
    if (r_sm < r_rest && f_sm < f_main) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + f_ghost)] =
            *v(r_sm, i, (nf - 1) - f_gl - f_ghost);
      }
    }
    if (r_sm == 0 && f_sm < f_main) {
      am_sm[f_sm + f_ghost] = *am(nf - f_gl - f_ghost);
      bm_sm[f_sm + f_ghost] = *bm(nf - f_gl - f_ghost);
    }
  }
      // __syncthreads();

  MGARDm_EXEC void
  Operation12() {
    /* Computation of v in parallel*/
    if (r_sm < r_rest && c_sm < c_rest) {
      // if (debug) printf("backward %f <- %f %f %f %f\n",
      //             tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
      //                      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]),
      //             prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);

      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
          tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
      //#pragma unroll 32
      for (SIZE i = 1; i < F; i++) {

        // if (debug) printf("backward %f <- %f %f %f %f\n",
        //           tridiag_backward2(
        //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)],
        //     am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]),
        //           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)],
        //     am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);

        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = tridiag_backward2(
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)],
            am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
      }
      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, FunctorBase<DeviceType>::GetBlockDimX() - 1)];
    }
  }
      // __syncthreads();
  MGARDm_EXEC void
  Operation13() {

    /* flush results to v */
    if (r_sm < r_rest && f_sm < F) {
      for (SIZE i = 0; i < c_rest; i++) {
        *v(r_sm, i, (nf - 1) - f_gl) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
  }
      // __syncthreads();

  MGARDm_EXEC void
  Operation14() {
    /* Update unloaded col */
    f_rest -= f_main;

    /* Advance c */
    f_gl += F;

    /* Copy next ghost to main */
    f_ghost = min(G, f_main - (F - G));
    if (r_sm < r_rest && f_sm < f_ghost) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + F)];
      }
      if (r_sm == 0) {
        am_sm[f_sm] = am_sm[f_sm + F];
        bm_sm[f_sm] = bm_sm[f_sm + F];
      }
    }
  }
      // __syncthreads();
    // } // end of while

  MGARDm_EXEC void
  Operation15() {
    /* Load all rest col */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + f_ghost)] =
            *v(r_sm, i, (nf - 1) - f_gl - f_ghost);
      }
    }
    if (r_sm == 0 && f_gl + f_ghost <= nf) {
      am_sm[f_sm + f_ghost] = *am(nf - f_gl - f_ghost);
      bm_sm[f_sm + f_ghost] = *bm(nf - f_gl - f_ghost);
    }
  }
    // __syncthreads();

  MGARDm_EXEC void
  Operation16() {
    /* Only 1 col remain */
    if (f_ghost + f_rest == 1) {
      if (r_sm < r_rest && c_sm < c_rest) {
        // if (debug) printf("backward %f <- %f %f %f %f\n",
        //             tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
        //                      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]),
        //             prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);

        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
            tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
        // printf ("prev_vec_sm = %f\n", prev_vec_sm );
        // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
      }
      //__syncthreads();

    } else {
      if (r_sm < r_rest && c_sm < c_rest) {
        // if (debug) printf("backward %f <- %f %f %f %f\n",
        //             tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
        //                      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]),
        //             prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);

        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
            tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
        for (SIZE i = 1; i < f_ghost + f_rest; i++) {

          // if (debug) printf("backward %f <- %f %f %f %f\n",
          //           tridiag_backward2(
          //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)],
          //     am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]),
          //           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)],
          //     am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);


          vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = tridiag_backward2(
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], 
              am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
        }
      }
    }
  }

  MGARDm_EXEC void
  Operation17() {
    // __syncthreads();
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_ghost + f_rest) {
      for (SIZE i = 0; i < c_rest; i++) {
        *v(r_sm, i, (nf - 1) - f_gl) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
        // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
        // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
        // c_stride, vec[i * row_stride * lddv + c_stride]);
      }
    }
    // __syncthreads();
  }

  MGARDm_CONT size_t
  shared_memory_size() {
    size_t size = 0;
    size = (R * C + 2) * (F + G) * sizeof(T);
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
  SubArray<1, T, DeviceType> am, bm;
  SubArray<D, T, DeviceType> v;

  // thread local variables
  LENGTH threadId;

  T *vec_sm;
  SIZE ldsm1, ldsm2;
  T *am_sm;
  T *bm_sm;

  SIZE *shape_sm;
  SIZE *shape_c_sm;
  DIM *processed_dims_sm;

  SIZE idx[D];

  SIZE nr;
  SIZE nc;
  SIZE nf;

  SIZE c_gl, r_gl, f_gl;
  SIZE c_sm, r_sm, f_sm;
  
  T prev_vec_sm;
  SIZE c_rest, r_rest;
  SIZE f_rest, f_ghost, f_main;
};

template <DIM D, typename T, typename DeviceType>
class Ipk1Reo: public AutoTuner<DeviceType> {
  public:
  MGARDm_CONT
  Ipk1Reo():AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F, SIZE G>
  MGARDm_CONT
  Task<Ipk1ReoFunctor<D, T, R, C, F, G, DeviceType> > 
  GenTask(SubArray<1, SIZE, DeviceType> shape, 
          SubArray<1, SIZE, DeviceType> shape_c,
          DIM processed_n, 
          SubArray<1, DIM, DeviceType> processed_dims, 
          DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
          SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm, 
          SubArray<D, T, DeviceType> v,
          int queue_idx) {
    using FunctorType = Ipk1ReoFunctor<D, T, R, C, F, G, DeviceType>;
    FunctorType functor(shape, shape_c, processed_n, processed_dims,
                        curr_dim_r, curr_dim_c, curr_dim_f,
                        am, bm, v);

    SIZE nr = shape_c.dataHost()[curr_dim_r];
    SIZE nc = shape_c.dataHost()[curr_dim_c];
    SIZE nf = shape_c.dataHost()[curr_dim_f];

    SIZE total_thread_x = nc;
    SIZE total_thread_y = nr;
    SIZE total_thread_z = 1;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbx = C;
    tby = R;
    tbz = 1;
    gridx = ceil((float)total_thread_x / tbx);
    gridy = ceil((float)total_thread_y / tby);
    gridz = 1;
    tbx = F; // necessary to ensure width is enough

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
        SIZE t = shape_c.dataHost()[d];
        gridx *= t;
      }
    }

    return Task(functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx, "Ipk1Reo"); 
  }

  MGARDm_CONT
  void Execute(SubArray<1, SIZE, DeviceType> shape, 
               SubArray<1, SIZE, DeviceType> shape_c,
               DIM processed_n, 
               SubArray<1, DIM, DeviceType> processed_dims, 
               DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
               SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm, 
               SubArray<D, T, DeviceType> v,
               int queue_idx) {
    int range_l = std::min(6, (int)std::log2(shape_c.dataHost()[curr_dim_f]) - 1);
    int arch = DeviceRuntime<DeviceType>::GetArchitectureGeneration();
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.auto_tuning_ts1[arch][prec][range_l];
    #define IPK(CONFIG)\
    if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) { \
      const int R=IPK_CONFIG[D-1][CONFIG][0];\
      const int C=IPK_CONFIG[D-1][CONFIG][1];\
      const int F=IPK_CONFIG[D-1][CONFIG][2];\
      const int G=IPK_CONFIG[D-1][CONFIG][3];\
      using FunctorType = Ipk1ReoFunctor<D, T, R, C, F, G, DeviceType>;\
      using TaskType = Task<FunctorType>;\
      TaskType task = GenTask<R, C, F, G>(\
                              shape, shape_c, processed_n, processed_dims,\
                              curr_dim_r, curr_dim_c, curr_dim_f,\
                              am, bm,\
                              v,\
                              queue_idx); \
      DeviceAdapter<TaskType, DeviceType> adapter; \
      adapter.Execute(task);\
    }

    IPK(0)
    IPK(1)
    IPK(2)
    IPK(3)
    IPK(4)  
    IPK(5)
    IPK(6)
    #undef IPK
  }
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, SIZE G, typename DeviceType>
class Ipk2ReoFunctor: public IterFunctor<DeviceType> {
  public:
  MGARDm_CONT Ipk2ReoFunctor(SubArray<1, SIZE, DeviceType> shape, 
                             SubArray<1, SIZE, DeviceType> shape_c,
                             DIM processed_n, 
                             SubArray<1, DIM, DeviceType> processed_dims, 
                             DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
                             SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm, 
                             SubArray<D, T, DeviceType> v):
                             shape(shape), shape_c(shape_c), 
                             processed_n(processed_n), processed_dims(processed_dims),
                             curr_dim_r(curr_dim_r), curr_dim_c(curr_dim_c), curr_dim_f(curr_dim_f),
                             am(am), bm(bm), v(v) {
    Functor<DeviceType>();                            
  }

  MGARDm_EXEC void
  Operation1() {
    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() * (FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetBlockDimY())) +
                      (FunctorBase<DeviceType>::GetThreadIdY() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();


    T * sm = (T*)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F;
    ldsm2 = C + G;
    vec_sm = sm; sm += R * ldsm1 * ldsm2;
    am_sm = sm; sm += ldsm2;
    bm_sm = sm; sm += ldsm2;

    SIZE * sm_size = (SIZE*)sm;
    shape_sm = sm_size; sm_size += D;
    shape_c_sm = sm_size; sm_size += D;
    sm = (T*)sm_size;

    DIM * sm_dim = (DIM*)sm;
    processed_dims_sm = sm_dim; sm_dim += D;
    sm = (T*)sm_dim;

    for (LENGTH i = threadId; i < D; i += FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetBlockDimY() * blockDim.z) {
      shape_sm[i] = *shape(i);
      shape_c_sm[i] = *shape_c(i);
    }
     for (LENGTH i = threadId; i < processed_n; i += FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetBlockDimY() * blockDim.z) {
      processed_dims_sm[i] = *processed_dims(i);
    }

  }

  MGARDm_EXEC void
  Operation2() {
    // __syncthreads();

    for (DIM d = 0; d < D; d++)
      idx[d] = 0;

    nr = shape_c_sm[curr_dim_r];
    nc = shape_c_sm[curr_dim_c];
    nf = shape_c_sm[curr_dim_f];

    if (D < 3)
      nr = 1;

    SIZE bidx = blockIdx.x;
    SIZE firstD = div_roundup(nf, FunctorBase<DeviceType>::GetBlockDimX());
    SIZE blockId = bidx % firstD;

    bidx /= firstD;

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
        SIZE t = shape_sm[d];
        //for (DIM k = 0; k < processed_n; k++) {
        //  if (d == processed_dims[k]) {
            t = shape_c_sm[d];
        //  }
        //}
        idx[d] = bidx % t;
        bidx /= t;
      }
    }

    v.offset(idx);
    // size_t other_offset_v = get_idx<D>(ldvs_sm, idx);
    // v = v + other_offset_v;

    f_gl = blockId * F;
    r_gl = blockIdx.y * R;
    c_gl = 0;

    f_sm = FunctorBase<DeviceType>::GetThreadIdX();
    r_sm = FunctorBase<DeviceType>::GetThreadIdY();
    c_sm = FunctorBase<DeviceType>::GetThreadIdX();

    v.offset(r_gl, 0, f_gl);
    // T *vec = v + get_idx(ldv1, ldv2, r_gl, 0, f_gl);

    prev_vec_sm = 0.0;

    f_rest = min(F, nf - blockId * F);
    r_rest = min(R, nr - blockIdx.y * R);

    // if (blockIdx.x == 1 && blockIdx.y == 0 && f_sm == 0 && r_sm == 0) {
    //   prSIZEf("f_rest: %d r_rest: %d\n", f_rest, r_rest);
    // }

    c_rest = nc;
    c_ghost = min(nc, G);
    c_main = C;

    /* Load first ghost */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
            *v(r_sm, c_gl + i, f_sm);
        // if (r_sm == 0) printf("r0_stride = %d, vec_sm[%d] = %f\n", r0_stride,
        // i, vec_sm[i * ldsm + c_sm]);
      }
    }
    if (r_sm == 0 && c_sm < c_ghost) {
      am_sm[c_sm] = *am(c_gl + c_sm);
      bm_sm[c_sm] = *bm(c_gl + c_sm);
    }
    c_rest -= c_ghost;

  }
    // __syncthreads();

  MGARDm_EXEC bool
  LoopCondition1() {
    return c_rest > C - c_ghost;
  }
    // while (c_rest > C - c_ghost) {
      // printf("%d %d %d\n", c_rest, C, c_ghost);
  MGARDm_EXEC void
  Operation3() {
    c_main = min(C, c_rest);
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_main; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)] =
            *v(r_sm, c_gl + i + c_ghost, f_sm);
      }
    }
    if (r_sm == 0 && c_sm < c_main) {
      am_sm[c_sm + c_ghost] = *am(c_gl + c_sm + c_ghost);
      bm_sm[c_sm + c_ghost] = *bm(c_gl + c_sm + c_ghost);
    }
  }
      // __syncthreads();

  MGARDm_EXEC void
  Operation4() {

    /* Computation of v in parallel*/
    if (r_sm < r_rest && f_sm < f_rest) {

      // #ifdef MGARD_X_FMA
      //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
      //       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2,
      //       r_sm, 0, f_sm)]);
      // #else
      //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] -= prev_vec_sm *
      //       bm_sm[0];
      // #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] = tridiag_forward2(
          prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);

      for (SIZE i = 1; i < C; i++) {
        // #ifdef MGARD_X_FMA
        //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
        //       __fma_rn(vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)],
        //       bm_sm[i],
        //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
        // #else
        //       // if (blockIdx.x == 1 && blockIdx.y == 0 && f_sm == 0 && r_sm
        //       == 0) {
        //       //   printf("calc: %f %f %f -> %f \n", vec_sm[get_idx(ldsm1,
        //       ldsm2, r_sm, i, f_sm)],
        //       //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)],
        //       bm_sm[i],  vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -
        //       //    vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)] *
        //       bm_sm[i]);
        //       // }

        //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -=
        //          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)] * bm_sm[i];
        // #endif
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = tridiag_forward2(
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], am_sm[i], bm_sm[i],
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
      }
      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, C - 1, f_sm)];
    }
  }
      // __syncthreads();

  MGARDm_EXEC void
  Operation5() {

    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < C; i++) {
        // if (blockIdx.x == 1 && blockIdx.y == 0 && f_sm == 0 && r_sm == 0) {
        //   printf("store: %f\n", vec_sm[get_idx(ldsm1, ldsm2, r_sm, i,
        //   f_sm)]);
        // }
        *v(r_sm, c_gl + i, f_sm) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
  }
      // __syncthreads();
  MGARDm_EXEC void
  Operation6() {
    /* Update unloaded col */
    c_rest -= c_main;

    /* Advance c */
    c_gl += C;

    /* Copy next ghost to main */
    c_ghost = min(G, c_main - (C - G));
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + C, f_sm)];
      }
    }
    if (r_sm == 0 && c_sm < c_ghost) {
      am_sm[c_sm] = am_sm[c_sm + C];
      bm_sm[c_sm] = bm_sm[c_sm + C];
    }
  }
    // __syncthreads();

    // } // end of while

  MGARDm_EXEC void
  Operation7() {
    /* Load all rest col */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)] =
            *v(r_sm, c_gl + i + c_ghost, f_sm);
      }
    }
    if (r_sm == 0 && c_sm < c_rest) {
      am_sm[c_sm + c_ghost] = *am(c_gl + c_sm + c_ghost);
      bm_sm[c_sm + c_ghost] = *bm(c_gl + c_sm + c_ghost);
    }
  }
    // __syncthreads();

  MGARDm_EXEC void
  Operation8() {
    /* Only 1 col remain */
    if (c_ghost + c_rest == 1) {
      if (r_sm < r_rest && f_sm < f_rest) {
        // vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] -= prev_vec_sm * bm_sm[0];
        // #ifdef MGARD_X_FMA
        //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
        //       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2,
        //       r_sm, 0, f_sm)]);
        // #else
        //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] -= prev_vec_sm *
        //       bm_sm[0];
        // #endif
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] = tridiag_forward2(
            prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
        // printf ("prev_vec_sm = %f\n", prev_vec_sm );
        // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
      }
      //__syncthreads();

    } else {
      if (r_sm < r_rest && f_sm < f_rest) {
        // #ifdef MGARD_X_FMA
        //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
        //       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2,
        //       r_sm, 0, f_sm)]);
        // #else
        //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] -= prev_vec_sm *
        //       bm_sm[0];
        // #endif
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] = tridiag_forward2(
            prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
        for (SIZE i = 1; i < c_ghost + c_rest; i++) {
          // #ifdef MGARD_X_FMA
          //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
          //       __fma_rn(vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)],
          //       bm_sm[i],
          //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
          // #else
          //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -=
          //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)] * bm_sm[i];
          // #endif
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = tridiag_forward2(
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], am_sm[i], bm_sm[i],
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
        }
      }
    }
  }
    // __syncthreads();

  MGARDm_EXEC void
  Operation9() {
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_ghost + c_rest; i++) {
        *v(r_sm, c_gl + i, f_sm) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
        // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
        // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
        // c_stride, vec[i * row_stride * lddv + c_stride]);
      }
    }
  }
    // __syncthreads();
  MGARDm_EXEC void
  Operation10() {
    /* backward */
    c_rest = nc;
    c_ghost = min(nc, G);
    c_main = C;
    c_gl = 0;
    prev_vec_sm = 0.0;

    // if (f_gl + f_sm == 0 && r_gl + r_sm == 0 && idx[3] == 0)
    //   debug = false;
    // if (debug)
    //   printf("block id: (%d %d %d) thread id: (%d %d %d)\n", blockIdx.x,
    //          blockIdx.y, blockIdx.z, FunctorBase<DeviceType>::GetThreadIdX(), FunctorBase<DeviceType>::GetThreadIdY(), FunctorBase<DeviceType>::GetThreadIdZ());

    /* Load first ghost */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
            *v(r_sm, (nc - 1) - (c_gl + i), f_sm);
        // if (debug)
        //   printf("load vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, i, f_sm),
        //          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
      }
    }
    if (r_sm == 0 && c_sm < c_ghost) {
      am_sm[c_sm] = *am(nc - (c_gl + c_sm));
      bm_sm[c_sm] = *bm(nc - (c_gl + c_sm));
    }
    c_rest -= c_ghost;
  }
    // __syncthreads();

  MGARDm_EXEC bool
  LoopCondition2() {
    return c_rest > C - c_ghost;
  }
    // while (c_rest > C - c_ghost) {
      // printf("%d %d %d\n", c_rest, C, c_ghost);
  MGARDm_EXEC void
  Operation11() {
    c_main = min(C, c_rest);
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_main; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)] = 
          *v(r_sm, (nc - 1) - (c_gl + i + c_ghost), f_sm);
        // if (debug)
        //   printf("load vec_sm[%d] = %f\n",
        //          get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm),
        //          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)]);
      }
    }
    if (r_sm == 0 && c_sm < c_main) {
      am_sm[c_sm + c_ghost] = *am(nc - (c_gl + c_sm + c_ghost));
      bm_sm[c_sm + c_ghost] = *bm(nc - (c_gl + c_sm + c_ghost));
    }
  }
      // __syncthreads();
  MGARDm_EXEC void
  Operation12() {
    // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0)
    // printf("*****test\n");
    /* Computation of v in parallel*/
    if (r_sm < r_rest && f_sm < f_rest) {
      // #ifdef MGARD_X_FMA
      //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
      //       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2,
      //       r_sm, 0, f_sm)]) * am_sm[0];
      // #else
      //       // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0)
      //       //     printf("(%f + %f * %f) * %f -> %f\n",
      //       //             vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)],
      //       //             dist_sm[0], prev_vec_sm, am_sm[0],
      //       //             (vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] -
      //       dist_sm[0] * prev_vec_sm) / am_sm[0]); vec_sm[get_idx(ldsm1,
      //       ldsm2, r_sm, 0, f_sm)] = (vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0,
      //       f_sm)] - dist_sm[0] * prev_vec_sm) / am_sm[0];
      // #endif
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)] =
          tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                           vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)]);
      // if (debug)
      //   printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, 0, f_sm),
      //          vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);

      for (SIZE i = 1; i < C; i++) {
        // #ifdef MGARD_X_FMA
        //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
        //       __fma_rn(dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i,
        //       f_sm)],
        //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)]) * am_sm[i];
        // #else
        //       // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0)
        //       //   printf("(%f + %f * %f) * %f -> %f\n",
        //       //             vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)],
        //       //             dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm,
        //       i-1, f_sm)], am_sm[i],
        //       //             (vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -
        //       //   dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1,
        //       f_sm)]) / am_sm[i]);

        //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
        //          (vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -
        //          dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1,
        //          f_sm)]) / am_sm[i];

        // #endif
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = tridiag_backward2(
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], 
            am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);

        // if (debug)
        //   printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, i, f_sm),
        //          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, C - 1, f_sm)];
    }
  }
      // __syncthreads();
  MGARDm_EXEC void
  Operation13() {
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < C; i++) {
        *v(r_sm, (nc - 1) - (c_gl + i), f_sm) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
  }
      // __syncthreads();

  MGARDm_EXEC void
  Operation14() {
    /* Update unloaded col */
    c_rest -= c_main;

    /* Advance c */
    c_gl += C;

    /* Copy next ghost to main */
    c_ghost = min(G, c_main - (C - G));
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + C, f_sm)];
      }
    }
    if (r_sm == 0 && c_sm < c_ghost) {
      am_sm[c_sm] = am_sm[c_sm + C];
      bm_sm[c_sm] = bm_sm[c_sm + C];
    }
  }
      // __syncthreads();

    // } // end of while

  MGARDm_EXEC void
  Operation15() {
    // Load all rest col
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)] = 
          *v(r_sm, (nc - 1) - (c_gl + i + c_ghost), f_sm);

        // if (debug)
        //   printf("load ec_sm[%d] = %f\n",
        //          get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm),
        //          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)]);
      }
    }
    if (r_sm == 0 && c_sm < c_rest) {
      am_sm[c_sm + c_ghost] = *am(nc - (c_gl + c_sm + c_ghost));
      bm_sm[c_sm + c_ghost] = *bm(nc - (c_gl + c_sm + c_ghost));
    }
  }
    // __syncthreads();

  MGARDm_EXEC void
  Operation16() {
    /* Only 1 col remain */
    if (c_ghost + c_rest == 1) {
      if (r_sm < r_rest && f_sm < f_rest) {
        // #ifdef MGARD_X_FMA
        //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
        //       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2,
        //       r_sm, 0, f_sm)]) * am_sm[0];
        // #else
        //       // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0)
        //       //     printf("(%f + %f * %f) * %f -> %f\n",
        //       //             vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)],
        //       //             dist_sm[0], prev_vec_sm, am_sm[0],
        //       //             (vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] -
        //       dist_sm[0] * prev_vec_sm) / am_sm[0]); vec_sm[get_idx(ldsm1,
        //       ldsm2, r_sm, 0, f_sm)] = (vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0,
        //       f_sm)] - dist_sm[0] * prev_vec_sm) / am_sm[0];
        // #endif
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)] =
            tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)]);
        // if (debug)
        //   printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, 0, f_sm),
        //          vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
        // printf ("prev_vec_sm = %f\n", prev_vec_sm );
        // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
      }
      //__syncthreads();

    } else {
      if (r_sm < r_rest && f_sm < f_rest) {
        // #ifdef MGARD_X_FMA
        //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
        //       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2,
        //       r_sm, 0, f_sm)]) * am_sm[0];
        // #else
        //       // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0)
        //       //     printf("(%f + %f * %f) * %f -> %f\n",
        //       //             vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)],
        //       //             dist_sm[0], prev_vec_sm, am_sm[0],
        //       //             (vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] -
        //       dist_sm[0] * prev_vec_sm) / am_sm[0]); vec_sm[get_idx(ldsm1,
        //       ldsm2, r_sm, 0, f_sm)] = (vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0,
        //       f_sm)] - dist_sm[0] * prev_vec_sm) / am_sm[0];
        // #endif
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)] =
            tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)]);
        // if (debug)
        //   printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, 0, f_sm),
        //          vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
        for (SIZE i = 1; i < c_ghost + c_rest; i++) {

          // #ifdef MGARD_X_FMA
          //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
          //       __fma_rn(dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i,
          //       f_sm)],
          //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)]) * am_sm[i];
          // #else
          //       // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0)
          //       //   printf("(%f + %f * %f) * %f -> %f\n",
          //       //             vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)],
          //       //             dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm,
          //       i-1, f_sm)], am_sm[i],
          //       //             (vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -
          //       //    dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1,
          //       f_sm)]) / am_sm[i]);
          //        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
          //          (vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -
          //          dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1,
          //          f_sm)]) / am_sm[i];
          // #endif
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = tridiag_backward2(
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)],
              am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
          // if (debug)
          //   printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, i, f_sm),
          //          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
        }
      }
    }
  }
    // __syncthreads();
  
  MGARDm_EXEC void
  Operation17() {
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_ghost + c_rest; i++) {
        *v(r_sm, (nc - 1) - (c_gl + i), f_sm) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
        // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
        // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
        // c_stride, vec[i * row_stride * lddv + c_stride]);
      }
    }
    __syncthreads();
  }

  MGARDm_CONT size_t
  shared_memory_size() {
    size_t size = 0;
    size = (R * F + 2) * (C + G) * sizeof(T);
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
  SubArray<1, T, DeviceType> am, bm;
  SubArray<D, T, DeviceType> v;

  // thread local variables
  LENGTH threadId;

  T *vec_sm;
  SIZE ldsm1, ldsm2;
  T *am_sm;
  T *bm_sm;

  SIZE *shape_sm;
  SIZE *shape_c_sm;
  DIM *processed_dims_sm;

  SIZE idx[D];

  SIZE nr;
  SIZE nc;
  SIZE nf;

  SIZE c_gl, r_gl, f_gl;
  SIZE c_sm, r_sm, f_sm;
  
  T prev_vec_sm;
  SIZE f_rest, r_rest;
  SIZE c_rest, c_ghost, c_main;
};

template <DIM D, typename T, typename DeviceType>
class Ipk2Reo: public AutoTuner<DeviceType> {
  public:
  MGARDm_CONT
  Ipk2Reo():AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F, SIZE G>
  MGARDm_CONT
  Task<Ipk2ReoFunctor<D, T, R, C, F, G, DeviceType> > 
  GenTask(SubArray<1, SIZE, DeviceType> shape, 
          SubArray<1, SIZE, DeviceType> shape_c,
          DIM processed_n, 
          SubArray<1, DIM, DeviceType> processed_dims, 
          DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
          SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm, 
          SubArray<D, T, DeviceType> v,
          int queue_idx) {
    using FunctorType = Ipk2ReoFunctor<D, T, R, C, F, G, DeviceType>;
    FunctorType functor(shape, shape_c, processed_n, processed_dims,
                        curr_dim_r, curr_dim_c, curr_dim_f,
                        am, bm, v);

    SIZE nr = shape_c.dataHost()[curr_dim_r];
    SIZE nc = shape_c.dataHost()[curr_dim_c];
    SIZE nf = shape_c.dataHost()[curr_dim_f];

    SIZE total_thread_x = nf;
    SIZE total_thread_y = nr;
    SIZE total_thread_z = 1;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbx = F;
    tby = R;
    tbz = 1;
    gridx = ceil((float)total_thread_x / tbx);
    gridy = ceil((float)total_thread_y / tby);
    gridz = 1;
    // tbx = F; // necessary to ensure width is enough

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
        SIZE t = shape_c.dataHost()[d];
        gridx *= t;
      }
    }

    return Task(functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx, "Ipk2Reo"); 
  }

  MGARDm_CONT
  void Execute(SubArray<1, SIZE, DeviceType> shape, 
               SubArray<1, SIZE, DeviceType> shape_c,
               DIM processed_n, 
               SubArray<1, DIM, DeviceType> processed_dims, 
               DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
               SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm, 
               SubArray<D, T, DeviceType> v,
               int queue_idx) {
    int range_l = std::min(6, (int)std::log2(shape_c.dataHost()[curr_dim_f]) - 1);
    int arch = DeviceRuntime<DeviceType>::GetArchitectureGeneration();
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.auto_tuning_ts2[arch][prec][range_l];
    #define IPK(CONFIG)\
    if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) { \
      const int R=IPK_CONFIG[D-1][CONFIG][0];\
      const int C=IPK_CONFIG[D-1][CONFIG][1];\
      const int F=IPK_CONFIG[D-1][CONFIG][2];\
      const int G=IPK_CONFIG[D-1][CONFIG][3];\
      using FunctorType = Ipk2ReoFunctor<D, T, R, C, F, G, DeviceType>;\
      using TaskType = Task<FunctorType>;\
      TaskType task = GenTask<R, C, F, G>(\
                              shape, shape_c, processed_n, processed_dims,\
                              curr_dim_r, curr_dim_c, curr_dim_f,\
                              am, bm,\
                              v,\
                              queue_idx); \
      DeviceAdapter<TaskType, DeviceType> adapter; \
      adapter.Execute(task);\
    }

    IPK(0)
    IPK(1)
    IPK(2)
    IPK(3)
    IPK(4)  
    IPK(5)
    IPK(6)
    #undef IPK
  }
};


template <DIM D, typename T, SIZE R, SIZE C, SIZE F, SIZE G, typename DeviceType>
class Ipk3ReoFunctor: public IterFunctor<DeviceType> {
  public:
  MGARDm_CONT Ipk3ReoFunctor(SubArray<1, SIZE, DeviceType> shape, 
                             SubArray<1, SIZE, DeviceType> shape_c,
                             DIM processed_n, 
                             SubArray<1, DIM, DeviceType> processed_dims, 
                             DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
                             SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm, 
                             SubArray<D, T, DeviceType> v):
                             shape(shape), shape_c(shape_c), 
                             processed_n(processed_n), processed_dims(processed_dims),
                             curr_dim_r(curr_dim_r), curr_dim_c(curr_dim_c), curr_dim_f(curr_dim_f),
                             am(am), bm(bm), v(v) {
    IterFunctor<DeviceType>();                            
  }

  MGARDm_EXEC void
  Operation1() {

    // printf("%llu %llu %llu, %llu %llu %llu\n",
    //         FunctorBase<DeviceType>::GetBlockIdX(),
    //         FunctorBase<DeviceType>::GetBlockIdY(),
    //         FunctorBase<DeviceType>::GetBlockIdZ(),
    //         FunctorBase<DeviceType>::GetThreadIdX(),
    //         FunctorBase<DeviceType>::GetThreadIdY(),
    //         FunctorBase<DeviceType>::GetThreadIdZ());

    LENGTH threadId = (FunctorBase<DeviceType>::GetThreadIdZ() * (FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetBlockDimY())) +
                      (FunctorBase<DeviceType>::GetThreadIdY() * FunctorBase<DeviceType>::GetBlockDimX()) + FunctorBase<DeviceType>::GetThreadIdX();

    T * sm = (T*)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F;
    ldsm2 = C;

    vec_sm = sm; sm += (R + G) * ldsm1 * ldsm2;
    am_sm = sm; sm += (R + G);
    bm_sm = sm; sm += (R + G);

    SIZE * sm_size = (SIZE*)sm;
    shape_sm = sm_size; sm_size += D;
    shape_c_sm = sm_size; sm_size += D;
    sm = (T*)sm_size;

    DIM * sm_dim = (DIM*)sm;
    processed_dims_sm = sm_dim; sm_dim += D;
    sm = (T*)sm_dim;

    for (LENGTH i = threadId; i < D; i += FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetBlockDimY() * blockDim.z) {
      shape_sm[i] = *shape(i);
      shape_c_sm[i] = *shape_c(i);
    }
    for (LENGTH i = threadId; i < processed_n; i += FunctorBase<DeviceType>::GetBlockDimX() * FunctorBase<DeviceType>::GetBlockDimY() * blockDim.z) {
      processed_dims_sm[i] = *processed_dims(i);
    }

  }
   
  // __syncthreads();

  MGARDm_EXEC void
  Operation2() {
    for (DIM d = 0; d < D; d++)
      idx[d] = 0;

    SIZE nr = shape_c_sm[curr_dim_r];
    SIZE nc = shape_c_sm[curr_dim_c];
    SIZE nf = shape_c_sm[curr_dim_f];

    SIZE bidx = blockIdx.x;
    SIZE firstD = div_roundup(nf, FunctorBase<DeviceType>::GetBlockDimX());
    SIZE blockId = bidx % firstD;

    bidx /= firstD;

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
        SIZE t = shape_sm[d];
        // for (DIM k = 0; k < processed_n; k++) {
          // if (d == processed_dims[k]) {
            t = shape_c_sm[d];
          // }
        // }
        idx[d] = bidx % t;
        bidx /= t;
      }
    }

    v.offset(idx);
    // size_t other_offset_v = get_idx<D>(ldvs_sm, idx);
    // v = v + other_offset_v;

    f_gl = blockId * F;
    c_gl = blockIdx.y * C;
    r_gl = 0;

    f_sm = FunctorBase<DeviceType>::GetThreadIdX();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    r_sm = FunctorBase<DeviceType>::GetThreadIdX();

    v.offset(0, c_gl, f_gl);
    // T *vec = v + get_idx(ldv1, ldv2, 0, c_gl, f_gl);

    prev_vec_sm = 0.0;

    f_rest = min(F, nf - blockId * F);
    c_rest = min(C, nc - blockIdx.y * C);

    r_rest = nr;
    r_ghost = min(nr, G);
    r_main = R;

    // if (f_gl + f_sm == 32 && c_gl + c_sm == 1 ) debug = false;

    /* Load first ghost */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
            *v(r_gl + i, c_sm, f_sm);
        // if (debug) printf("load first sm[%d] %f [%d]\n", i,
        //             vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)], r_gl + i);
      }
    }

    if (c_sm == 0 && r_sm < r_ghost) {
      am_sm[r_sm] = *am(r_gl + r_sm);
      bm_sm[r_sm] = *bm(r_gl + r_sm);
    }
    r_rest -= r_ghost;
  }

  MGARDm_EXEC bool
  LoopCondition1() {
    return r_rest > R - r_ghost;
  }
    // __syncthreads();

  MGARDm_EXEC void
  Operation3() {
  // while (r_rest > R - r_ghost) {
    r_main = min(R, r_rest);
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_main; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)] =
            *v(r_gl + i + r_ghost, c_sm, f_sm);
        // if (debug) printf("load ghost sm[%d] %f [%d]\n", i + r_ghost,
        //              vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)],
        //              r_gl + i + r_ghost);
      }
    }
    if (c_sm == 0 && r_sm < r_main) {
      am_sm[r_sm + r_ghost] = *am(r_gl + r_sm + r_ghost);
      bm_sm[r_sm + r_ghost] = *bm(r_gl + r_sm + r_ghost);
    }
  }
      // __syncthreads();

  MGARDm_EXEC void
  Operation4() {
    /* Computation of v in parallel*/
    if (c_sm < c_rest && f_sm < f_rest) {

      // #ifdef MGARD_X_FMA
      //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
      //       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0,
      //       c_sm, f_sm)]);
      // #else
      //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] -= prev_vec_sm *
      //       bm_sm[0];
      // #endif
      // if (debug) printf("compute sm[%d] %f <- %f %f %f\n", 0,
      //               tridiag_forward(prev_vec_sm, bm_sm[0],
      //               vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]),
      //               prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0,
      //               c_sm, f_sm)]);

      vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = tridiag_forward2(
          prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);

      for (SIZE i = 1; i < R; i++) {
        // #ifdef MGARD_X_FMA
        //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
        //       __fma_rn(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
        //       bm_sm[i],
        //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
        // #else
        //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] -=
        //          vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)] * bm_sm[i];
        // #endif

        // if (debug) printf("compute sm[%d] %f <- %f %f %f\n", i,
        //             tridiag_forward(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm,
        //             f_sm)],
        //              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]),
        //             vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
        //              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);

        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = tridiag_forward2(
            vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], am_sm[i], bm_sm[i],
            vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, R - 1, c_sm, f_sm)];
    }
  }
      // __syncthreads();
  MGARDm_EXEC void
  Operation5() {
    /* flush results to v */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < R; i++) {
        *v(r_gl + i, c_sm, f_sm) =
            vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];
        // if (debug) printf("store[%d] %f [%d]\n", r_gl + i,
        //   *v(r_gl + i, c_sm, f_sm), i);
      }
    }
  }
      // __syncthreads();

  MGARDm_EXEC void
  Operation6() {

    // /* Update unloaded col */
    r_rest -= r_main;

    /* Advance c */
    r_gl += R;

    /* Copy next ghost to main */
    r_ghost = min(G, r_main - (R - G));
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
            vec_sm[get_idx(ldsm1, ldsm2, i + R, c_sm, f_sm)];

        // if (debug) printf("copy next ghost[%d] %f [%d]\n", i,
        //   vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)], i+R);
      }
    }
    if (c_sm == 0 && r_sm < r_ghost) {
      am_sm[r_sm] = am_sm[r_sm + R];
      bm_sm[r_sm] = bm_sm[r_sm + R];
    }
  }


      // __syncthreads();

    // } // end of while

  MGARDm_EXEC void
  Operation7() {
    /* Load all rest col */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)] =
            *v(r_gl + i + r_ghost, c_sm, f_sm);

        // if (debug) printf("load ghost-rest sm[%d] %f [%d]\n", i + r_ghost,
        //               vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)],
        //               r_gl + i + r_ghost);
      }
    }

    if (c_sm == 0 && r_sm < r_rest) {
      am_sm[r_sm + r_ghost] = *am(r_gl + r_sm + r_ghost);
      bm_sm[r_sm + r_ghost] = *bm(r_gl + r_sm + r_ghost);
    }
  }

    // __syncthreads();

  MGARDm_EXEC void
  Operation8() {

    /* Only 1 col remain */
    if (r_ghost + r_rest == 1) {
      if (c_sm < c_rest && f_sm < f_rest) {

        // #ifdef MGARD_X_FMA
        //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
        //       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0,
        //       c_sm, f_sm)]);
        // #else
        //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] -= prev_vec_sm *
        //       bm_sm[0];
        // #endif
        // if (debug) printf("compute-rest sm[%d] %f <- %f %f %f\n", 0,
        //               tridiag_forward(prev_vec_sm, bm_sm[0],
        //               vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]),
        //               prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0,
        //               c_sm, f_sm)]);

        vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = tridiag_forward2(
            prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
        // printf ("prev_vec_sm = %f\n", prev_vec_sm );
        // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
      }
      //__syncthreads();

    } else {
      if (c_sm < c_rest && f_sm < f_rest) {
        // #ifdef MGARD_X_FMA
        //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
        //       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0,
        //       c_sm, f_sm)]);
        // #else
        //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] -= prev_vec_sm *
        //       bm_sm[0];
        // #endif

        // if (debug) printf("compute-rest sm[%d] %f <- %f %f %f\n", 0,
        //               tridiag_forward(prev_vec_sm, bm_sm[0],
        //               vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]),
        //               prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0,
        //               c_sm, f_sm)]);

        vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = tridiag_forward2(
            prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
        for (SIZE i = 1; i < r_ghost + r_rest; i++) {
          // #ifdef MGARD_X_FMA
          //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
          //       __fma_rn(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
          //       bm_sm[i],
          //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
          // #else
          //       vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] -=
          //         vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)] * bm_sm[i];
          // #endif
          // if (debug) printf("compute-rest sm[%d] %f <- %f %f %f\n", i,
          //             tridiag_forward(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm,
          //             f_sm)],
          //              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]),
          //             vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
          //              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);

          vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = tridiag_forward2(
              vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], am_sm[i], bm_sm[i],
              vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
        }
      }
    }
  }

  MGARDm_EXEC void
  Operation9() {
    // __syncthreads();
    /* flush results to v */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_ghost + r_rest; i++) {

        *v(r_gl + i, c_sm, f_sm) =
            vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];

        // if (debug) printf("store-rest[%d] %f [%d]\n", r_gl + i,
        //     *v(r_gl + i, c_sm, f_sm), i);
        // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
        // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
        // c_stride, vec[i * row_stride * lddv + c_stride]);
      }
    }
  }
    // __syncthreads();

  MGARDm_EXEC void
  Operation10() {

    /* backward */
    r_rest = nr;
    r_ghost = min(nr, G);
    r_main = R;
    r_gl = 0;
    prev_vec_sm = 0.0;

    /* Load first ghost */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
            *v((nr - 1) - (r_gl + i), c_sm, f_sm);

        // if (debug) printf("load first sm[%d] %f [%d]\n", i,
        //             vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)], (nr - 1) -
        //             (r_gl + i));

        // if (r_sm == 0) printf("r0_stride = %d, vec_sm[%d] = %f\n", r0_stride,
        // i, vec_sm[i * ldsm + c_sm]);
      }
    }

    if (c_sm == 0 && r_sm < r_ghost) {
      am_sm[r_sm] = *am(nr - (r_gl + r_sm));
      bm_sm[r_sm] = *bm(nr - (r_gl + r_sm));
    }
    r_rest -= r_ghost;
  }
    // __syncthreads();

  MGARDm_EXEC bool
  LoopCondition2() {
    return r_rest > R - r_ghost;
  }

  MGARDm_EXEC void
  Operation11() {
    // while (r_rest > R - r_ghost) {
    r_main = min(R, r_rest);
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_main; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)] = 
        *v((nr - 1) - (r_gl + i + r_ghost), c_sm, f_sm);
        // if (debug) printf("load ghost sm[%d] %f [%d]\n", i + r_ghost,
        //             vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)],
        //             (nr - 1) - (r_gl + i + r_ghost));
      }
    }
    if (c_sm == 0 && r_sm < r_main) {
      am_sm[r_sm + r_ghost] = *am(nr - (r_gl + r_sm + r_ghost));
      bm_sm[r_sm + r_ghost] = *bm(nr - (r_gl + r_sm + r_ghost));
    }
  }
      // __syncthreads();

  MGARDm_EXEC void
  Operation12() {
    /* Computation of v in parallel*/
    if (c_sm < c_rest && f_sm < f_rest) {
      // #ifdef MGARD_X_FMA
      //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
      //       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2, 0,
      //       c_sm, f_sm)]) * am_sm[0];
      // #else
      //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
      //       (vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] - dist_sm[0] *
      //       prev_vec_sm) / am_sm[0];
      // #endif

      // if (debug) printf("compute sm[%d] %f <- %f %f %f %f\n", 0,
      //              tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0],
      //    vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]),
      //              prev_vec_sm, dist_sm[0], am_sm[0],
      //    vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);

      vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
          tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                           vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
      for (SIZE i = 1; i < R; i++) {

        // #ifdef MGARD_X_FMA
        //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
        //       __fma_rn(dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm,
        //       f_sm)],
        //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]) * am_sm[i];
        // #else
        //       vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
        //         (vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] -
        //          dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm,
        //          f_sm)]) / am_sm[i];
        // #endif

        // if (debug) printf("compute sm[%d] %f <- %f %f %f %f\n", i,
        //             tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, i - 1,
        //             c_sm, f_sm)],
        //  dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]),
        //             vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
        //  dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);

        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = tridiag_backward2(
            vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], 
            am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, R - 1, c_sm, f_sm)];
    }
  }
      // __syncthreads();

  MGARDm_EXEC void
  Operation13() {
    /* flush results to v */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < R; i++) {
        // if (blockIdx.x == 0 && blockIdx.y == 0 && FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
        // FunctorBase<DeviceType>::GetThreadIdY() == 0) {
        //   printf("%d %d %d (%f) <- %d %d %d\n", (nr - 1) - (r_gl + i), c_sm,
        //   f_sm,
        //           vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)], i, c_sm,
        //           f_sm);
        // }
        *v((nr - 1) - (r_gl + i), c_sm, f_sm) =
            vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];

        // if (debug) printf("store[%d] %f [%d]\n", (nr - 1) - (r_gl + i),
        //   *v((nr - 1) - (r_gl + i), c_sm, f_sm), i);
      }
    }
  }
      // __syncthreads();

  MGARDm_EXEC void
  Operation14() {
    // /* Update unloaded col */
    r_rest -= r_main;

    /* Advance c */
    r_gl += R;

    /* Copy next ghost to main */
    r_ghost = min(G, r_main - (R - G));
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
            vec_sm[get_idx(ldsm1, ldsm2, i + R, c_sm, f_sm)];

        // if (debug) printf("copy next ghost[%d] %f [%d]\n", i,
        //  vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)], i+R);
      }
    }
    if (c_sm == 0 && r_sm < r_ghost) {
      am_sm[r_sm] = am_sm[r_sm + R];
      bm_sm[r_sm] = bm_sm[r_sm + R];
    }
  }
      // __syncthreads();

    // } // end of while

  MGARDm_EXEC void
  Operation15() {
    /* Load all rest col */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)] = 
        *v((nr - 1) - (r_gl + i + r_ghost), c_sm, f_sm);

        // if (debug) printf("load ghost-rest sm[%d] %f [%d]\n", i + r_ghost,
        //               vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)],
        //               (nr - 1) - (r_gl + i + r_ghost));
      }
    }
    if (c_sm == 0 && r_sm < r_rest) {
      am_sm[r_sm + r_ghost] = *am(nr - (r_gl + r_sm + r_ghost));
      bm_sm[r_sm + r_ghost] = *bm(nr - (r_gl + r_sm + r_ghost));
    }
    // __syncthreads();
  }

  MGARDm_EXEC void
  Operation16() {
    /* Only 1 col remain */
    if (r_ghost + r_rest == 1) {
      if (c_sm < c_rest && f_sm < f_rest) {
        // #ifdef MGARD_X_FMA
        //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
        //       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2, 0,
        //       c_sm, f_sm)]) * am_sm[0];
        // #else
        //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
        //       (vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] - dist_sm[0] *
        //       prev_vec_sm) / am_sm[0];
        // #endif
        // if (blockIdx.x == 0 && blockIdx.y == 0 && FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
        // FunctorBase<DeviceType>::GetThreadIdY() == 0) {
        //   printf("backward 1 (%f) %f %f %f %f\n", tridiag_backward(prev_vec_sm,
        //   dist_sm[0], am_sm[0],
        //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]), prev_vec_sm,
        //     dist_sm[0], am_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm,
        //     f_sm)]);

        // }
        // if (debug) printf("compute sm[%d] %f <- %f %f %f %f\n", 0,
        //               tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0],
        //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]),
        //               prev_vec_sm, dist_sm[0], am_sm[0],
        //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);

        vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
            tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
        // printf ("prev_vec_sm = %f\n", prev_vec_sm );
        // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
      }
      //__syncthreads();

    } else {
      if (c_sm < c_rest && f_sm < f_rest) {
        // #ifdef MGARD_X_FMA
        //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
        //       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2, 0,
        //       c_sm, f_sm)]) * am_sm[0];
        // #else
        //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
        //       (vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] - dist_sm[0] *
        //       prev_vec_sm) / am_sm[0];
        // #endif
        // if (blockIdx.x == 0 && blockIdx.y == 0 && FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
        // FunctorBase<DeviceType>::GetThreadIdY() == 0) {
        //   printf("backward 1 (%f) %f %f %f %f\n", tridiag_backward(prev_vec_sm,
        //   dist_sm[0], am_sm[0],
        //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]), prev_vec_sm,
        //     dist_sm[0], am_sm[0], vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm,
        //     f_sm)]);

        // }

        // if (debug) printf("compute sm[%d] %f <- %f %f %f %f\n", 0,
        //               tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0],
        //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]),
        //               prev_vec_sm, dist_sm[0], am_sm[0],
        //     vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);

        vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
            tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
        for (SIZE i = 1; i < r_ghost + r_rest; i++) {

          // #ifdef MGARD_X_FMA
          //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
          //       __fma_rn(dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm,
          //       f_sm)],
          //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]) * am_sm[i];
          // #else
          //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
          //         (vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] -
          //          dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm,
          //          f_sm)]) / am_sm[i];
          // #endif
          //   if (blockIdx.x == 0 && blockIdx.y == 0 && FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
          //   FunctorBase<DeviceType>::GetThreadIdY() == 0) { printf("backward R=%d (%f) %f %f %f %f\n", i,
          //   tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
          //    dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm,
          //    f_sm)]), vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
          //    dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm,
          //    f_sm)]);

          // }

          // if (debug) printf("compute sm[%d] %f <- %f %f %f %f\n", i,
          //             tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, i - 1,
          //             c_sm, f_sm)],
          //  dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]),
          //             vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
          //  dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);

          vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = tridiag_backward2(
              vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], 
              am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
        }
      }
    }
  }
   
  MGARDm_EXEC void
  Operation17() {
    // __syncthreads();
    /* flush results to v */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_ghost + r_rest; i++) {
        *v((nr - 1) - (r_gl + i), c_sm, f_sm) =
            vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];

        // if (debug) printf("store-rest[%d] %f [%d]\n", (nr - 1) - (r_gl + i),
        //     *v((nr - 1) - (r_gl + i), c_sm, f_sm), i);

        // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
        // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
        // c_stride, vec[i * row_stride * lddv + c_stride]);
      }
    }
    // __syncthreads();
  }

  MGARDm_CONT size_t
  shared_memory_size() {
    size_t size = 0;
    size = (C * F + 2) * (R + G) * sizeof(T);
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
  SubArray<1, T, DeviceType> am, bm;
  SubArray<D, T, DeviceType> v;

  // thread local variables
  LENGTH threadId;

  T *vec_sm;
  SIZE ldsm1, ldsm2;
  T *am_sm;
  T *bm_sm;

  SIZE *shape_sm;
  SIZE *shape_c_sm;
  DIM *processed_dims_sm;

  SIZE idx[D];

  SIZE nr;
  SIZE nc;
  SIZE nf;

  SIZE c_gl, r_gl, f_gl;
  SIZE c_sm, r_sm, f_sm;
  
  T prev_vec_sm;
  SIZE c_rest, f_rest;
  SIZE r_rest, r_ghost, r_main;
};

template <DIM D, typename T, typename DeviceType>
class Ipk3Reo: public AutoTuner<DeviceType> {
  public:
  MGARDm_CONT
  Ipk3Reo():AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F, SIZE G>
  MGARDm_CONT
  Task<Ipk3ReoFunctor<D, T, R, C, F, G, DeviceType> > 
  GenTask(SubArray<1, SIZE, DeviceType> shape, 
          SubArray<1, SIZE, DeviceType> shape_c,
          DIM processed_n, 
          SubArray<1, DIM, DeviceType> processed_dims, 
          DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
          SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm, 
          SubArray<D, T, DeviceType> v,
          int queue_idx) {
    using FunctorType = Ipk3ReoFunctor<D, T, R, C, F, G, DeviceType>;
    FunctorType functor(shape, shape_c, processed_n, processed_dims,
                        curr_dim_r, curr_dim_c, curr_dim_f,
                        am, bm, v);

    SIZE nr = shape_c.dataHost()[curr_dim_r];
    SIZE nc = shape_c.dataHost()[curr_dim_c];
    SIZE nf = shape_c.dataHost()[curr_dim_f];

    SIZE total_thread_x = nf;
    SIZE total_thread_y = nc;
    SIZE total_thread_z = 1;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbx = F;
    tby = C;
    tbz = 1;
    gridx = ceil((float)total_thread_x / tbx);
    gridy = ceil((float)total_thread_y / tby);
    gridz = 1;
    // tbx = F; // necessary to ensure width is enough

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
        SIZE t = shape_c.dataHost()[d];
        gridx *= t;
      }
    }

    return Task(functor, gridz, gridy, gridx, 
                tbz, tby, tbx, sm_size, queue_idx, "Ipk3Reo"); 
  }

  MGARDm_CONT
  void Execute(SubArray<1, SIZE, DeviceType> shape, 
               SubArray<1, SIZE, DeviceType> shape_c,
               DIM processed_n, 
               SubArray<1, DIM, DeviceType> processed_dims, 
               DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
               SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm, 
               SubArray<D, T, DeviceType> v,
               int queue_idx) {
    int range_l = std::min(6, (int)std::log2(shape_c.dataHost()[curr_dim_f]) - 1);
    int arch = DeviceRuntime<DeviceType>::GetArchitectureGeneration();
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.auto_tuning_ts2[arch][prec][range_l];
    #define IPK(CONFIG)\
    if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) { \
      const int R=IPK_CONFIG[D-1][CONFIG][0];\
      const int C=IPK_CONFIG[D-1][CONFIG][1];\
      const int F=IPK_CONFIG[D-1][CONFIG][2];\
      const int G=IPK_CONFIG[D-1][CONFIG][3];\
      using FunctorType = Ipk3ReoFunctor<D, T, R, C, F, G, DeviceType>;\
      using TaskType = Task<FunctorType>;\
      TaskType task = GenTask<R, C, F, G>(\
                              shape, shape_c, processed_n, processed_dims,\
                              curr_dim_r, curr_dim_c, curr_dim_f,\
                              am, bm,\
                              v,\
                              queue_idx); \
      DeviceAdapter<TaskType, DeviceType> adapter; \
      adapter.Execute(task);\
    }

    IPK(0)
    IPK(1)
    IPK(2)
    IPK(3)
    IPK(4)  
    IPK(5)
    IPK(6)
    #undef IPK
  }
};

}

#endif