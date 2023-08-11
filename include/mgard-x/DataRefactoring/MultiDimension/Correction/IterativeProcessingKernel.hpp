/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_ITERATIVE_PROCESSING_KERNEL_TEMPLATE
#define MGARD_X_ITERATIVE_PROCESSING_KERNEL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"
#include "IPKFunctor.h"

namespace mgard_x {

namespace data_refactoring {

namespace multi_dimension {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, SIZE G,
          typename DeviceType>
class Ipk1ReoFunctor : public IterFunctor<DeviceType> {
public:
  MGARDX_CONT Ipk1ReoFunctor() {}
  MGARDX_CONT
  Ipk1ReoFunctor(DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
                 SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm,
                 SubArray<D, T, DeviceType> v)
      : curr_dim_r(curr_dim_r), curr_dim_c(curr_dim_c), curr_dim_f(curr_dim_f),
        am(am), bm(bm), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    // bool debug = false;
    // if (blockIdx.z == 0 && FunctorBase<DeviceType>::GetBlockIdY() == 0 &&
    // FunctorBase<DeviceType>::GetBlockIdX() == 0 &&
    //     FunctorBase<DeviceType>::GetThreadIdZ() == 0 &&
    //     FunctorBase<DeviceType>::GetThreadIdY() == 0)
    //   debug = false;

    // bool debug2 = false;
    // if (FunctorBase<DeviceType>::GetThreadIdZ() == 0 &&
    // FunctorBase<DeviceType>::GetThreadIdY() == 0 &&
    // FunctorBase<DeviceType>::GetThreadIdX() == 0)
    //   debug2 = false;

    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                (FunctorBase<DeviceType>::GetBlockDimX() *
                 FunctorBase<DeviceType>::GetBlockDimY())) +
               (FunctorBase<DeviceType>::GetThreadIdY() *
                FunctorBase<DeviceType>::GetBlockDimX()) +
               FunctorBase<DeviceType>::GetThreadIdX();

    T *sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F + G;
    ldsm2 = C;
    vec_sm = sm;
    sm += R * ldsm1 * ldsm2;
    am_sm = sm;
    sm += ldsm1;
    bm_sm = sm;
    sm += ldsm1;
  }

  MGARDX_EXEC void Operation2() {
    // __syncthreads();

    for (DIM d = 0; d < D; d++)
      idx[d] = 0;

    nr = v.shape(curr_dim_r);
    nc = v.shape(curr_dim_c);
    nf = v.shape(curr_dim_f);

    if (D < 3)
      nr = 1;
    if (D < 2)
      nc = 1;

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    SIZE firstD = div_roundup(nc, C);
    SIZE blockId = bidx % firstD;
    // if (debug2) {
    //   printf("FunctorBase<DeviceType>::GetBlockIdX() %u nc %u
    //   FunctorBase<DeviceType>::GetBlockDimX() %u firstD: %u blockId %u\n",
    //   FunctorBase<DeviceType>::GetBlockIdX(), nc,
    //   FunctorBase<DeviceType>::GetBlockDimX(), firstD, blockId);
    // }
    bidx /= firstD;

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
        SIZE t = v.shape(d);
        idx[d] = bidx % t;
        bidx /= t;
      }
    }

    v.offset(idx);
    // size_t other_offset_v = get_idx<D>(ldvs_sm, idx);
    // v = v + other_offset_v;

    // if (debug2) {
    //   printf("ipk1 idx: %u %u %u %u %u ld: %u %u %u %u %u\n", idx[4], idx[3],
    //   idx[2], idx[1], idx[0], ldvs_sm[4], ldvs_sm[3], ldvs_sm[2], ldvs_sm[1],
    //   ldvs_sm[0]); printf("ipk1 other_offset_v: %llu\n", other_offset_v);

    //   SIZE curr_stride = 1;
    //   SIZE ret_idx = 0;
    //   for (DIM i = 0; i < D; i++) {
    //     ret_idx += idx[i] * curr_stride;
    //     printf("%llu * %llu = %llu\n", curr_stride, ldvs_sm[i],
    //     curr_stride*ldvs_sm[i]); curr_stride *= ldvs_sm[i];

    //   }
    // }

    c_gl = blockId * C;
    r_gl = FunctorBase<DeviceType>::GetBlockIdY() * R;
    f_gl = FunctorBase<DeviceType>::GetThreadIdX();

    c_sm = FunctorBase<DeviceType>::GetThreadIdX();
    r_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();

    // if (idx[3] == 0 && idx[4] == 1 && r_gl+r_sm == 0 && c_gl+c_sm == 0) {
    //   printf("offset = %llu\n", other_offset_v);
    //   debug2 = false;
    // }

    // T *vec = v + get_idx(ldv1, ldv2, r_gl, c_gl, 0);

    v.offset_3d(r_gl, c_gl, 0);

    prev_vec_sm = 0.0;

    c_rest = Math<DeviceType>::Min(C, nc - blockId * C);
    r_rest = Math<DeviceType>::Min(
        R, nr - FunctorBase<DeviceType>::GetBlockIdY() * R);

    f_rest = nf;
    f_ghost = Math<DeviceType>::Min(nf, G);
    f_main = F;

    // printf("r_sm: %d, r_rest: %d, c_sm: %d, c_rest: %d f_sm: %d, f_rest %d ,
    // nf: %d\n", r_sm, r_rest, c_sm, c_rest, f_sm, f_rest, nf);

    // printf("test %f", vec_sm[get_idx(ldsm1, ldsm2, 0, 1, 0)]);
    /* Load first ghost */
    if (r_sm < r_rest && f_sm < f_ghost) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = *v(r_sm, i, f_gl);
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

  MGARDX_EXEC bool LoopCondition1() { return f_rest > F - f_ghost; }

  MGARDX_EXEC void Operation3() {
    // while (f_rest > F - f_ghost) {
    // if (c_gl == 0 && c_sm == 0 && r_gl == 0 && r_sm == 0) printf("%d %d\n",
    // f_rest, F - f_ghost);
    f_main = Math<DeviceType>::Min(F, f_rest);
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
  MGARDX_EXEC void Operation4() {

    /* Computation of v in parallel*/
    if (r_sm < r_rest && c_sm < c_rest) {
      // if (debug) printf("forward %f <- %f %f %f %f\n",
      //             tridiag_forward2(
      //     prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2, r_sm,
      //     c_sm, 0)]),
      //             prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1,
      //             ldsm2, r_sm, c_sm, 0)]);

      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
          tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);

      //#pragma unroll 32
      for (SIZE i = 1; i < F; i++) {
        // if (debug) printf("forward %f <- %f %f %f %f\n",
        //           tridiag_forward2(
        //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i],
        //     bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]),
        //           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i],
        //           bm_sm[i],
        //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);

        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = tridiag_forward2(
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i],
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, F - 1)];
    }
  }
  // __syncthreads();

  MGARDX_EXEC void Operation5() {

    /* flush results to v */
    if (r_sm < r_rest && f_sm < F) {
      for (SIZE i = 0; i < c_rest; i++) {
        // if (FunctorBase<DeviceType>::GetBlockIdX() == 0 &&
        // FunctorBase<DeviceType>::GetBlockIdY() == 0 && r_sm == 0 && i == 1) {
        //   printf("store [%d %d %d] %f<-%f [%d %d %d]\n",
        //     r_sm, i, f_gl, *v(r_sm, i, f_gl),
        //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)], r_sm, i, f_sm);
        // }
        *v(r_sm, i, f_gl) = vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
        // if (FunctorBase<DeviceType>::GetBlockIdX() == 0 &&
        // FunctorBase<DeviceType>::GetBlockIdY() == 0 && r_sm == 0 && i == 1) {
        //   printf("store [%d %d %d] %f<-%f [%d %d %d]\n",
        //     r_sm, i, f_gl, *v(r_sm, i, f_gl),
        //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)], r_sm, i, f_sm);
        // }
      }
    }
  }

  // __syncthreads();

  MGARDX_EXEC void Operation6() {

    /* Update unloaded col */
    f_rest -= f_main;

    /* Advance c */
    f_gl += F;

    /* Copy next ghost to main */
    f_ghost = Math<DeviceType>::Min(G, f_main - (F - G));
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
  MGARDX_EXEC void Operation7() {
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

  MGARDX_EXEC void Operation8() {

    /* Only 1 col remain */
    if (f_ghost + f_rest == 1) {
      if (r_sm < r_rest && c_sm < c_rest) {
        // if (debug) printf("forward %f <- %f %f %f %f\n",
        //             tridiag_forward2(
        //     prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2,
        //     r_sm, c_sm, 0)]),
        //             prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1,
        //             ldsm2, r_sm, c_sm, 0)]);
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
            tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
        // printf ("prev_vec_sm = %f\n", prev_vec_sm );
        // printf ("vec_sm[r_sm * ldsm + 0] = %f\n", vec_sm[r_sm * ldsm + 0] );
      }
      //__syncthreads();

    } else {
      if (r_sm < r_rest && c_sm < c_rest) {
        // if (debug) printf("forward %f <- %f %f %f %f\n",
        //             tridiag_forward2(
        //     prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2,
        //     r_sm, c_sm, 0)]),
        //             prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1,
        //             ldsm2, r_sm, c_sm, 0)]);
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
            tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
        for (SIZE i = 1; i < f_ghost + f_rest; i++) {
          // if (debug) printf("forward %f <- %f %f %f %f\n",
          //           tridiag_forward2(
          //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i],
          //     bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]),
          //           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)],
          //           am_sm[i], bm_sm[i],
          //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = tridiag_forward2(
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i],
              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
        }
      }
    }
  }

  MGARDX_EXEC void Operation9() {
    // __syncthreads();
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_ghost + f_rest) {
      for (SIZE i = 0; i < c_rest; i++) {
        *v(r_sm, i, f_gl) = vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
        // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
        // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
        // c_stride, vec[i * row_stride * lddv + c_stride]);
      }
    }
  }
  // __syncthreads();

  MGARDX_EXEC void Operation10() {
    /* backward */
    f_rest = nf;
    f_ghost = Math<DeviceType>::Min(nf, G);
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
  MGARDX_EXEC bool LoopCondition2() { return f_rest > F - f_ghost; }

  MGARDX_EXEC void Operation11() {
    // while (f_rest > F - f_ghost) {
    f_main = Math<DeviceType>::Min(F, f_rest);
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

  MGARDX_EXEC void Operation12() {
    /* Computation of v in parallel*/
    if (r_sm < r_rest && c_sm < c_rest) {
      // if (debug) printf("backward %f <- %f %f %f %f\n",
      //             tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
      //                      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]),
      //             prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1,
      //             ldsm2, r_sm, c_sm, 0)]);

      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
          tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                            vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
      //#pragma unroll 32
      for (SIZE i = 1; i < F; i++) {

        // if (debug) printf("backward %f <- %f %f %f %f\n",
        //           tridiag_backward2(
        //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)],
        //     am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm,
        //     i)]),
        //           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)],
        //     am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm,
        //     i)]);

        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = tridiag_backward2(
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i],
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
      }
      /* Store last v */
      prev_vec_sm =
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm,
                         FunctorBase<DeviceType>::GetBlockDimX() - 1)];
    }
  }
  // __syncthreads();
  MGARDX_EXEC void Operation13() {

    /* flush results to v */
    if (r_sm < r_rest && f_sm < F) {
      for (SIZE i = 0; i < c_rest; i++) {
        *v(r_sm, i, (nf - 1) - f_gl) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
  }
  // __syncthreads();

  MGARDX_EXEC void Operation14() {
    /* Update unloaded col */
    f_rest -= f_main;

    /* Advance c */
    f_gl += F;

    /* Copy next ghost to main */
    f_ghost = Math<DeviceType>::Min(G, f_main - (F - G));
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

  MGARDX_EXEC void Operation15() {
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

  MGARDX_EXEC void Operation16() {
    /* Only 1 col remain */
    if (f_ghost + f_rest == 1) {
      if (r_sm < r_rest && c_sm < c_rest) {
        // if (debug) printf("backward %f <- %f %f %f %f\n",
        //             tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
        //                      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]),
        //             prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1,
        //             ldsm2, r_sm, c_sm, 0)]);

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
        //             prev_vec_sm, am_sm[0], bm_sm[0], vec_sm[get_idx(ldsm1,
        //             ldsm2, r_sm, c_sm, 0)]);

        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
            tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                              vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
        for (SIZE i = 1; i < f_ghost + f_rest; i++) {

          // if (debug) printf("backward %f <- %f %f %f %f\n",
          //           tridiag_backward2(
          //     vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)],
          //     am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm,
          //     i)]),
          //           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)],
          //     am_sm[i], bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm,
          //     i)]);

          vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = tridiag_backward2(
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i],
              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
        }
      }
    }
  }

  MGARDX_EXEC void Operation17() {
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
    v.reset_offset();
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size = (R * C + 2) * (F + G) * sizeof(T);
    return size;
  }

private:
  // functor parameters
  DIM curr_dim_r, curr_dim_c, curr_dim_f;
  SubArray<1, T, DeviceType> am, bm;
  SubArray<D, T, DeviceType> v;

  // thread local variables
  SIZE threadId;

  T *vec_sm;
  SIZE ldsm1, ldsm2;
  T *am_sm;
  T *bm_sm;

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
class Ipk1ReoKernel : public Kernel {
public:
  constexpr static DIM NumDim = D;
  using DataType = T;
  constexpr static std::string_view Name = "ipk1_nd";
  constexpr static SIZE G = 2;
  MGARDX_CONT
  Ipk1ReoKernel(DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
                SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm,
                SubArray<D, T, DeviceType> v)
      : curr_dim_r(curr_dim_r), curr_dim_c(curr_dim_c), curr_dim_f(curr_dim_f),
        am(am), bm(bm), v(v) {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Ipk1ReoFunctor<D, T, R, C, F, G, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = Ipk1ReoFunctor<D, T, R, C, F, G, DeviceType>;
    FunctorType functor(curr_dim_r, curr_dim_c, curr_dim_f, am, bm, v);

    SIZE nr = 1, nc = 1;
    if (D >= 3)
      nr = v.shape(curr_dim_r);
    if (D >= 2)
      nc = v.shape(curr_dim_c);

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
        SIZE t = v.shape(d);
        gridx *= t;
      }
    }

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  DIM curr_dim_r, curr_dim_c, curr_dim_f;
  SubArray<1, T, DeviceType> am, bm;
  SubArray<D, T, DeviceType> v;
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, SIZE G,
          typename DeviceType>
class Ipk2ReoFunctor : public IterFunctor<DeviceType> {
public:
  MGARDX_CONT Ipk2ReoFunctor() {}
  MGARDX_CONT
  Ipk2ReoFunctor(DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
                 SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm,
                 SubArray<D, T, DeviceType> v)
      : curr_dim_r(curr_dim_r), curr_dim_c(curr_dim_c), curr_dim_f(curr_dim_f),
        am(am), bm(bm), v(v) {
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
    ldsm2 = C + G;
    vec_sm = sm;
    sm += R * ldsm1 * ldsm2;
    am_sm = sm;
    sm += ldsm2;
    bm_sm = sm;
    sm += ldsm2;
  }

  MGARDX_EXEC void Operation2() {
    // __syncthreads();

    for (DIM d = 0; d < D; d++)
      idx[d] = 0;

    nr = v.shape(curr_dim_r);
    nc = v.shape(curr_dim_c);
    nf = v.shape(curr_dim_f);

    if (D < 3)
      nr = 1;

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    SIZE firstD = div_roundup(nf, FunctorBase<DeviceType>::GetBlockDimX());
    SIZE blockId = bidx % firstD;

    bidx /= firstD;

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
        SIZE t = v.shape(d);
        idx[d] = bidx % t;
        bidx /= t;
      }
    }

    v.offset(idx);
    // size_t other_offset_v = get_idx<D>(ldvs_sm, idx);
    // v = v + other_offset_v;

    f_gl = blockId * F;
    r_gl = FunctorBase<DeviceType>::GetBlockIdY() * R;
    c_gl = 0;

    f_sm = FunctorBase<DeviceType>::GetThreadIdX();
    r_sm = FunctorBase<DeviceType>::GetThreadIdY();
    c_sm = FunctorBase<DeviceType>::GetThreadIdX();

    v.offset_3d(r_gl, 0, f_gl);
    // T *vec = v + get_idx(ldv1, ldv2, r_gl, 0, f_gl);

    prev_vec_sm = 0.0;

    f_rest = Math<DeviceType>::Min(F, nf - blockId * F);
    r_rest = Math<DeviceType>::Min(
        R, nr - FunctorBase<DeviceType>::GetBlockIdY() * R);

    // if (FunctorBase<DeviceType>::GetBlockIdX() == 1 &&
    // FunctorBase<DeviceType>::GetBlockIdY() == 0 && f_sm == 0 && r_sm == 0) {
    //   prSIZEf("f_rest: %d r_rest: %d\n", f_rest, r_rest);
    // }

    c_rest = nc;
    c_ghost = Math<DeviceType>::Min(nc, G);
    c_main = C;

    /* Load first ghost */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = *v(r_sm, c_gl + i, f_sm);
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

  MGARDX_EXEC bool LoopCondition1() { return c_rest > C - c_ghost; }
  // while (c_rest > C - c_ghost) {
  // printf("%d %d %d\n", c_rest, C, c_ghost);
  MGARDX_EXEC void Operation3() {
    c_main = Math<DeviceType>::Min(C, c_rest);
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

  MGARDX_EXEC void Operation4() {

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
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
          tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                           vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);

      for (SIZE i = 1; i < C; i++) {
        // #ifdef MGARD_X_FMA
        //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
        //       __fma_rn(vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)],
        //       bm_sm[i],
        //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
        // #else
        //       // if (FunctorBase<DeviceType>::GetBlockIdX() == 1 &&
        //       FunctorBase<DeviceType>::GetBlockIdY() == 0 && f_sm == 0 &&
        //       r_sm
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
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], am_sm[i],
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
      }
      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, C - 1, f_sm)];
    }
  }
  // __syncthreads();

  MGARDX_EXEC void Operation5() {

    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < C; i++) {
        // if (FunctorBase<DeviceType>::GetBlockIdX() == 1 &&
        // FunctorBase<DeviceType>::GetBlockIdY() == 0 && f_sm == 0 && r_sm ==
        // 0) {
        //   printf("store: %f\n", vec_sm[get_idx(ldsm1, ldsm2, r_sm, i,
        //   f_sm)]);
        // }
        *v(r_sm, c_gl + i, f_sm) = vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
  }
  // __syncthreads();
  MGARDX_EXEC void Operation6() {
    /* Update unloaded col */
    c_rest -= c_main;

    /* Advance c */
    c_gl += C;

    /* Copy next ghost to main */
    c_ghost = Math<DeviceType>::Min(G, c_main - (C - G));
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

  MGARDX_EXEC void Operation7() {
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

  MGARDX_EXEC void Operation8() {
    /* Only 1 col remain */
    if (c_ghost + c_rest == 1) {
      if (r_sm < r_rest && f_sm < f_rest) {
        // vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] -= prev_vec_sm *
        // bm_sm[0]; #ifdef MGARD_X_FMA
        //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
        //       __fma_rn(prev_vec_sm, bm_sm[0], vec_sm[get_idx(ldsm1, ldsm2,
        //       r_sm, 0, f_sm)]);
        // #else
        //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] -= prev_vec_sm *
        //       bm_sm[0];
        // #endif
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
            tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
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
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
            tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
        for (SIZE i = 1; i < c_ghost + c_rest; i++) {
          // #ifdef MGARD_X_FMA
          //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
          //       __fma_rn(vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)],
          //       bm_sm[i],
          //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
          // #else
          //       vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -=
          //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)] *
          //         bm_sm[i];
          // #endif
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = tridiag_forward2(
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], am_sm[i],
              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
        }
      }
    }
  }
  // __syncthreads();

  MGARDX_EXEC void Operation9() {
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_ghost + c_rest; i++) {
        *v(r_sm, c_gl + i, f_sm) = vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
        // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
        // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
        // c_stride, vec[i * row_stride * lddv + c_stride]);
      }
    }
  }
  // __syncthreads();
  MGARDX_EXEC void Operation10() {
    /* backward */
    c_rest = nc;
    c_ghost = Math<DeviceType>::Min(nc, G);
    c_main = C;
    c_gl = 0;
    prev_vec_sm = 0.0;

    // if (f_gl + f_sm == 0 && r_gl + r_sm == 0 && idx[3] == 0)
    //   debug = false;
    // if (debug)
    //   printf("block id: (%d %d %d) thread id: (%d %d %d)\n",
    //   FunctorBase<DeviceType>::GetBlockIdX(),
    //          FunctorBase<DeviceType>::GetBlockIdY(), blockIdx.z,
    //          FunctorBase<DeviceType>::GetThreadIdX(),
    //          FunctorBase<DeviceType>::GetThreadIdY(),
    //          FunctorBase<DeviceType>::GetThreadIdZ());

    /* Load first ghost */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
            *v(r_sm, (nc - 1) - (c_gl + i), f_sm);
        // if (debug)
        //   printf("load vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, i,
        //   f_sm),
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

  MGARDX_EXEC bool LoopCondition2() { return c_rest > C - c_ghost; }
  // while (c_rest > C - c_ghost) {
  // printf("%d %d %d\n", c_rest, C, c_ghost);
  MGARDX_EXEC void Operation11() {
    c_main = Math<DeviceType>::Min(C, c_rest);
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
  MGARDX_EXEC void Operation12() {
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
      //   printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, 0,
      //   f_sm),
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
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], am_sm[i],
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);

        // if (debug)
        //   printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, i,
        //   f_sm),
        //          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, C - 1, f_sm)];
    }
  }
  // __syncthreads();
  MGARDX_EXEC void Operation13() {
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < C; i++) {
        *v(r_sm, (nc - 1) - (c_gl + i), f_sm) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
  }
  // __syncthreads();

  MGARDX_EXEC void Operation14() {
    /* Update unloaded col */
    c_rest -= c_main;

    /* Advance c */
    c_gl += C;

    /* Copy next ghost to main */
    c_ghost = Math<DeviceType>::Min(G, c_main - (C - G));
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

  MGARDX_EXEC void Operation15() {
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

  MGARDX_EXEC void Operation16() {
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
        //   printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, 0,
        //   f_sm),
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
        //   printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, 0,
        //   f_sm),
        //          vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
        for (SIZE i = 1; i < c_ghost + c_rest; i++) {

          // #ifdef MGARD_X_FMA
          //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
          //       __fma_rn(dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i,
          //       f_sm)],
          //         vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)]) *
          //         am_sm[i];
          // #else
          //       // if (r_gl == 0 && f_gl == 0 && r_sm == 0 && f_sm == 0)
          //       //   printf("(%f + %f * %f) * %f -> %f\n",
          //       //             vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)],
          //       //             dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm,
          //       i-1, f_sm)], am_sm[i],
          //       //             (vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]
          //       -
          //       //    dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1,
          //       f_sm)]) / am_sm[i]);
          //        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
          //          (vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] -
          //          dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1,
          //          f_sm)]) / am_sm[i];
          // #endif
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = tridiag_backward2(
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], am_sm[i],
              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
          // if (debug)
          //   printf("calc vec_sm[%d] = %f\n", get_idx(ldsm1, ldsm2, r_sm, i,
          //   f_sm),
          //          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
        }
      }
    }
  }
  // __syncthreads();

  MGARDX_EXEC void Operation17() {
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
    // __syncthreads();
    v.reset_offset();
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size = (R * F + 2) * (C + G) * sizeof(T);
    return size;
  }

private:
  // functor parameters
  DIM curr_dim_r, curr_dim_c, curr_dim_f;
  SubArray<1, T, DeviceType> am, bm;
  SubArray<D, T, DeviceType> v;

  // thread local variables
  SIZE threadId;

  T *vec_sm;
  SIZE ldsm1, ldsm2;
  T *am_sm;
  T *bm_sm;

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
class Ipk2ReoKernel : public Kernel {
public:
  constexpr static DIM NumDim = D;
  using DataType = T;
  constexpr static std::string_view Name = "ipk2_nd";
  constexpr static SIZE G = 2;
  MGARDX_CONT
  Ipk2ReoKernel(DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
                SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm,
                SubArray<D, T, DeviceType> v)
      : curr_dim_r(curr_dim_r), curr_dim_c(curr_dim_c), curr_dim_f(curr_dim_f),
        am(am), bm(bm), v(v) {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Ipk2ReoFunctor<D, T, R, C, F, G, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = Ipk2ReoFunctor<D, T, R, C, F, G, DeviceType>;
    FunctorType functor(curr_dim_r, curr_dim_c, curr_dim_f, am, bm, v);

    SIZE nr = 1, nf = 1;
    if (D >= 3)
      nr = v.shape(curr_dim_r);
    nf = v.shape(curr_dim_f);

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
        SIZE t = v.shape(d);
        gridx *= t;
      }
    }

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  DIM curr_dim_r, curr_dim_c, curr_dim_f;
  SubArray<1, T, DeviceType> am, bm;
  SubArray<D, T, DeviceType> v;
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, SIZE G,
          typename DeviceType>
class Ipk3ReoFunctor : public IterFunctor<DeviceType> {
public:
  MGARDX_CONT Ipk3ReoFunctor() {}
  MGARDX_CONT
  Ipk3ReoFunctor(DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
                 SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm,
                 SubArray<D, T, DeviceType> v)
      : curr_dim_r(curr_dim_r), curr_dim_c(curr_dim_c), curr_dim_f(curr_dim_f),
        am(am), bm(bm), v(v) {
    IterFunctor<DeviceType>();
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

    vec_sm = sm;
    sm += (R + G) * ldsm1 * ldsm2;
    am_sm = sm;
    sm += (R + G);
    bm_sm = sm;
    sm += (R + G);
  }

  // __syncthreads();

  MGARDX_EXEC void Operation2() {
    for (DIM d = 0; d < D; d++)
      idx[d] = 0;

    nr = v.shape(curr_dim_r);
    nc = v.shape(curr_dim_c);
    nf = v.shape(curr_dim_f);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    SIZE firstD = div_roundup(nf, FunctorBase<DeviceType>::GetBlockDimX());
    SIZE blockId = bidx % firstD;

    bidx /= firstD;

    for (DIM d = 0; d < D; d++) {
      if (d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
        SIZE t = v.shape(d);
        idx[d] = bidx % t;
        bidx /= t;
      }
    }

    v.offset(idx);
    // size_t other_offset_v = get_idx<D>(ldvs_sm, idx);
    // v = v + other_offset_v;

    f_gl = blockId * F;
    c_gl = FunctorBase<DeviceType>::GetBlockIdY() * C;
    r_gl = 0;

    f_sm = FunctorBase<DeviceType>::GetThreadIdX();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    r_sm = FunctorBase<DeviceType>::GetThreadIdX();

    v.offset_3d(0, c_gl, f_gl);
    // T *vec = v + get_idx(ldv1, ldv2, 0, c_gl, f_gl);

    prev_vec_sm = 0.0;

    f_rest = Math<DeviceType>::Min(F, (SIZE)(nf - blockId * F));
    c_rest = Math<DeviceType>::Min(
        C, (SIZE)(nc - FunctorBase<DeviceType>::GetBlockIdY() * C));

    r_rest = nr;
    r_ghost = Math<DeviceType>::Min(nr, G);
    r_main = R;

    // if (f_gl + f_sm == 32 && c_gl + c_sm == 1 ) debug = false;

    /* Load first ghost */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = *v(r_gl + i, c_sm, f_sm);
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

  MGARDX_EXEC bool LoopCondition1() { return r_rest > R - r_ghost; }
  // __syncthreads();

  MGARDX_EXEC void Operation3() {
    // while (r_rest > R - r_ghost) {
    r_main = Math<DeviceType>::Min(R, r_rest);
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

  MGARDX_EXEC void Operation4() {
    /* Computation of v in parallel*/
    if (c_sm < c_rest && f_sm < f_rest) {
      vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
          tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                           vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);

      for (SIZE i = 1; i < R; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = tridiag_forward2(
            vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], am_sm[i],
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, R - 1, c_sm, f_sm)];
    }
  }
  // __syncthreads();
  MGARDX_EXEC void Operation5() {
    /* flush results to v */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < R; i++) {
        *v(r_gl + i, c_sm, f_sm) = vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];
        // if (debug) printf("store[%d] %f [%d]\n", r_gl + i,
        //   *v(r_gl + i, c_sm, f_sm), i);
      }
    }
  }
  // __syncthreads();

  MGARDX_EXEC void Operation6() {

    // /* Update unloaded col */
    r_rest -= r_main;

    /* Advance c */
    r_gl += R;

    /* Copy next ghost to main */
    r_ghost = Math<DeviceType>::Min(G, r_main - (R - G));
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

  //     // __syncthreads();

  //   // } // end of while

  MGARDX_EXEC void Operation7() {
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

  MGARDX_EXEC void Operation8() {

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

        vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
            tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
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

        vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
            tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
        for (SIZE i = 1; i < r_ghost + r_rest; i++) {
          // #ifdef MGARD_X_FMA
          //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
          //       __fma_rn(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
          //       bm_sm[i],
          //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
          // #else
          //       vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] -=
          //         vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)] *
          //         bm_sm[i];
          // #endif
          // if (debug) printf("compute-rest sm[%d] %f <- %f %f %f\n", i,
          //             tridiag_forward(vec_sm[get_idx(ldsm1, ldsm2, i - 1,
          //             c_sm, f_sm)],
          //              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm,
          //              f_sm)]),
          //             vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
          //              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm,
          //              f_sm)]);

          vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = tridiag_forward2(
              vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], am_sm[i],
              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
        }
      }
    }
  }

  MGARDX_EXEC void Operation9() {
    // __syncthreads();
    /* flush results to v */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_ghost + r_rest; i++) {

        *v(r_gl + i, c_sm, f_sm) = vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];

        // if (debug) printf("store-rest[%d] %f [%d]\n", r_gl + i,
        //     *v(r_gl + i, c_sm, f_sm), i);
        // printf("c_stride = %d, c_sm = %d, vec_sm = %f, vec[%d] =
        // %f\n",c_stride, c_sm, vec_sm[r_sm * ldsm + 0],i * row_stride * lddv +
        // c_stride, vec[i * row_stride * lddv + c_stride]);
      }
    }
  }
  // __syncthreads();

  MGARDX_EXEC void Operation10() {

    /* backward */
    r_rest = nr;
    r_ghost = Math<DeviceType>::Min(nr, G);
    r_main = R;
    r_gl = 0;
    prev_vec_sm = 0.0;

    /* Load first ghost */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
            *v((nr - 1) - (r_gl + i), c_sm, f_sm);
      }
    }

    if (c_sm == 0 && r_sm < r_ghost) {
      am_sm[r_sm] = *am(nr - (r_gl + r_sm));
      bm_sm[r_sm] = *bm(nr - (r_gl + r_sm));
    }
    r_rest -= r_ghost;
  }
  // __syncthreads();

  MGARDX_EXEC bool LoopCondition2() { return r_rest > R - r_ghost; }

  MGARDX_EXEC void Operation11() {
    // while (r_rest > R - r_ghost) {
    r_main = Math<DeviceType>::Min(R, r_rest);
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

  MGARDX_EXEC void Operation12() {
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
            vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], am_sm[i],
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, R - 1, c_sm, f_sm)];
    }
  }
  // __syncthreads();

  MGARDX_EXEC void Operation13() {
    /* flush results to v */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < R; i++) {
        // if (FunctorBase<DeviceType>::GetBlockIdX() == 0 &&
        // FunctorBase<DeviceType>::GetBlockIdY() == 0 &&
        // FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
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

  MGARDX_EXEC void Operation14() {
    // /* Update unloaded col */
    r_rest -= r_main;

    /* Advance c */
    r_gl += R;

    /* Copy next ghost to main */
    r_ghost = Math<DeviceType>::Min(G, r_main - (R - G));
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

  MGARDX_EXEC void Operation15() {
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

  MGARDX_EXEC void Operation16() {
    /* Only 1 col remain */
    if (r_ghost + r_rest == 1) {
      if (c_sm < c_rest && f_sm < f_rest) {
        // #ifdef MGARD_X_FMA
        //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
        //       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2,
        //       0, c_sm, f_sm)]) * am_sm[0];
        // #else
        //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
        //       (vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] - dist_sm[0] *
        //       prev_vec_sm) / am_sm[0];
        // #endif
        // if (FunctorBase<DeviceType>::GetBlockIdX() == 0 &&
        // FunctorBase<DeviceType>::GetBlockIdY() == 0 &&
        // FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
        // FunctorBase<DeviceType>::GetThreadIdY() == 0) {
        //   printf("backward 1 (%f) %f %f %f %f\n",
        //   tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0],
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
        //       __fma_rn(dist_sm[0], prev_vec_sm, vec_sm[get_idx(ldsm1, ldsm2,
        //       0, c_sm, f_sm)]) * am_sm[0];
        // #else
        //       vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
        //       (vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] - dist_sm[0] *
        //       prev_vec_sm) / am_sm[0];
        // #endif
        // if (FunctorBase<DeviceType>::GetBlockIdX() == 0 &&
        // FunctorBase<DeviceType>::GetBlockIdY() == 0 &&
        // FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
        // FunctorBase<DeviceType>::GetThreadIdY() == 0) {
        //   printf("backward 1 (%f) %f %f %f %f\n",
        //   tridiag_backward(prev_vec_sm, dist_sm[0], am_sm[0],
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
          //       __fma_rn(dist_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i - 1,
          //       c_sm, f_sm)],
          //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]) * am_sm[i];
          // #else
          //         vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] =
          //         (vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] -
          //          dist_sm[i] * vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm,
          //          f_sm)]) / am_sm[i];
          // #endif
          //   if (FunctorBase<DeviceType>::GetBlockIdX() == 0 &&
          //   FunctorBase<DeviceType>::GetBlockIdY() == 0 &&
          //   FunctorBase<DeviceType>::GetThreadIdX() == 0 &&
          //   FunctorBase<DeviceType>::GetThreadIdY() == 0) { printf("backward
          //   R=%d (%f) %f %f %f %f\n", i,
          //   tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm,
          //   f_sm)],
          //    dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm,
          //    f_sm)]), vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
          //    dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm,
          //    f_sm)]);

          // }

          // if (debug) printf("compute sm[%d] %f <- %f %f %f %f\n", i,
          //             tridiag_backward(vec_sm[get_idx(ldsm1, ldsm2, i - 1,
          //             c_sm, f_sm)],
          //  dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm,
          //  f_sm)]),
          //             vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)],
          //  dist_sm[i], am_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm,
          //  f_sm)]);

          vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = tridiag_backward2(
              vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], am_sm[i],
              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
        }
      }
    }
  }

  MGARDX_EXEC void Operation17() {
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
    v.reset_offset();
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size = (C * F + 2) * (R + G) * sizeof(T);
    return size;
  }

private:
  // functor parameters
  DIM curr_dim_r, curr_dim_c, curr_dim_f;
  SubArray<1, T, DeviceType> am, bm;
  SubArray<D, T, DeviceType> v;

  // thread local variables
  SIZE threadId;

  T *vec_sm;
  SIZE ldsm1, ldsm2;
  T *am_sm;
  T *bm_sm;

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
class Ipk3ReoKernel : public Kernel {
public:
  constexpr static DIM NumDim = D;
  using DataType = T;
  constexpr static std::string_view Name = "ipk3_nd";
  constexpr static SIZE G = 2;
  MGARDX_CONT
  Ipk3ReoKernel(DIM curr_dim_r, DIM curr_dim_c, DIM curr_dim_f,
                SubArray<1, T, DeviceType> am, SubArray<1, T, DeviceType> bm,
                SubArray<D, T, DeviceType> v)
      : curr_dim_r(curr_dim_r), curr_dim_c(curr_dim_c), curr_dim_f(curr_dim_f),
        am(am), bm(bm), v(v) {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Ipk3ReoFunctor<D, T, R, C, F, G, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = Ipk3ReoFunctor<D, T, R, C, F, G, DeviceType>;
    FunctorType functor(curr_dim_r, curr_dim_c, curr_dim_f, am, bm, v);

    SIZE nc = 1, nf = 1;
    if (D >= 2)
      nc = v.shape(curr_dim_c);
    nf = v.shape(curr_dim_f);

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
        SIZE t = v.shape(d);
        gridx *= t;
      }
    }

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  DIM curr_dim_r, curr_dim_c, curr_dim_f;
  SubArray<1, T, DeviceType> am, bm;
  SubArray<D, T, DeviceType> v;
};

} // namespace multi_dimension

} // namespace data_refactoring

} // namespace mgard_x

#endif
