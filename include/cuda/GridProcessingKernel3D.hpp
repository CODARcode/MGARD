/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_GRID_PROCESSING_KERNEL_3D_TEMPLATE
#define MGRAD_CUDA_GRID_PROCESSING_KERNEL_3D_TEMPLATE

#include "CommonInternal.h"
#include "GPKFunctor.h"
#include "GridProcessingKernel3D.h"

namespace mgard_cuda {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F>
MGARDm_EXEC void
__gpk_reo_3d(IDX ngridz, IDX ngridy, IDX ngridx, IDX nblockz, IDX nblocky,
             IDX nblockx, IDX blockz, IDX blocky, IDX blockx, IDX threadz,
             IDX thready, IDX threadx, SIZE nr, SIZE nc, SIZE nf, SIZE nr_c,
             SIZE nc_c, SIZE nf_c, T *dratio_r, T *dratio_c, T *dratio_f, T *dv,
             SIZE lddv1, SIZE lddv2, T *dw, SIZE lddw1, SIZE lddw2, T *dwf,
             SIZE lddwf1, SIZE lddwf2, T *dwc, SIZE lddwc1, SIZE lddwc2, T *dwr,
             SIZE lddwr1, SIZE lddwr2, T *dwcf, SIZE lddwcf1, SIZE lddwcf2,
             T *dwrf, SIZE lddwrf1, SIZE lddwrf2, T *dwrc, SIZE lddwrc1,
             SIZE lddwrc2, T *dwrcf, SIZE lddwrcf1, SIZE lddwrcf2) {

  // // to be removed
  int TYPE = 1;
  bool INTERPOLATION = true;
  bool CALC_COEFF = true;
  bool in_next = false;
  bool skip = false;

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

  // r = blockIdx.z * blockDim.z;
  // c = blockIdx.y * blockDim.y;
  // f = blockIdx.x * blockDim.x;

  r = blockz * nblockz;
  c = blocky * nblocky;
  f = blockx * nblockx;

  rest_r = nr - r;
  rest_c = nc - c;
  rest_f = nf - f;

  nr_p = nr;
  nc_p = nc;
  nf_p = nf;

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

  // r_sm = threadIdx.z;
  // c_sm = threadIdx.y;
  // f_sm = threadIdx.x;

  r_sm = threadz;
  c_sm = thready;
  f_sm = threadx;

  r_sm_ex = R * 2;
  c_sm_ex = C * 2;
  f_sm_ex = F * 2;

  // threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
  //            (threadIdx.y * blockDim.x) + threadIdx.x;

  threadId = (threadz * (nblockx * nblocky)) + (thready * nblockx) + threadx;

  T *sm = SharedMemory<T>();
  SIZE ldsm1 = F * 2 + 1;
  SIZE ldsm2 = C * 2 + 1;
  T *v_sm = sm;
  T *ratio_f_sm = sm + (F * 2 + 1) * (C * 2 + 1) * (R * 2 + 1);
  T *ratio_c_sm = ratio_f_sm + F * 2;
  T *ratio_r_sm = ratio_c_sm + C * 2;

  r_gl = r + r_sm;
  r_gl_ex = r + R * 2;
  c_gl = c + c_sm;
  c_gl_ex = c + C * 2;
  f_gl = f + f_sm;
  f_gl_ex = f + F * 2;

  //  __syncthreads();
  // if (r_sm == 0 && c_sm == 0 && f_sm == 0) {
  //   //printf("setting zeros\n");
  //   for (int i = 0; i < R * 2 + 1; i++) {
  //     for (int j = 0; j < C * 2 + 1; j++) {
  //       for (int k = 0; k < F * 2 + 1; k++) {
  //         v_sm[get_idx(ldsm1, ldsm2, i, j, k)] = 0.0;
  //       }
  //     }
  //   }
  //   //printf("done zeros\n");
  // }
  //  __syncthreads();
  /* Load v */
  // loading extra rules
  // case 1: input = odd (non-padding required)
  //    case 1.a: block size < rest (need to load extra);
  //    case 1.b: block size > rest (NO need to load extra);
  // case 2: input = even (padding requried)
  //    case 2.a: block size < rest (need to load extra);
  //    case 2.b: block size >= rest (NO need to load extra, but need
  //    padding);

  // Load from dv
  if (r_sm < rest_r && c_sm < rest_c && f_sm < rest_f) {

    // load cubic
    // asm volatile("membar.cta;");
    // start = clock64();
    v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
        dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl)];
    // if (blockIdx.x==0 && blockIdx.y==0&&blockIdx.z==0) {
    //   printf("load (%d %d %d) %f <- %d+(%d %d %d) (ld: %d %d)\n",
    //           r_sm, c_sm, f_sm,
    //           dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl)],
    //           other_offset_v+r_gl, c_gl, f_gl, lddv1, lddv2);
    // }
    if (r_sm == 0) {
      if (rest_r > R * 2) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)] =
            dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl)];
      }
    }
    if (c_sm == 0) {
      if (rest_c > C * 2) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] =
            dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl)];
      }
    }
    if (f_sm == 0) {
      if (rest_f > F * 2) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)] =
            dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl_ex)];
      }
    }
    if (c_sm == 0 && f_sm == 0) {
      if (rest_c > C * 2 && rest_f > F * 2) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)] =
            dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)];
      }
    }
    if (r_sm == 0 && f_sm == 0) {
      if (rest_r > R * 2 && rest_f > F * 2) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)] =
            dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)];
      }
    }
    if (r_sm == 0 && c_sm == 0) {
      if (rest_r > R * 2 && rest_c > C * 2) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)] =
            dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)];
      }
    }
    if (r_sm == 0 && c_sm == 0 && f_sm == 0) {
      if (rest_r > R * 2 && rest_c > C * 2 && rest_f > F * 2) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm_ex)] =
            dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)];
      }
    }
  }

  __syncthreads();

  // apply padding is necessary
  if (r_sm < rest_r && c_sm < rest_c && f_sm < rest_f) {

    // printf("load main[%d %d %d]:%f --> [%d %d %d] (%d %d %d)\n", r_gl,
    // c_gl, f_gl,
    //     dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl)], r_sm, c_sm, f_sm, nr,
    //     nc, nf);

    // asm volatile("membar.cta;");
    // start = clock64() - start;
    // printf("[load main] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
    // blockIdx.y, blockIdx.x, start); start = clock64();

    // load extra surface

    if (r_sm == 0) {
      if (rest_r > R * 2) {
        // v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)] =
        //     dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl)];
        // printf("load-r[%d %d %d]:%f --> [%d %d %d]\n", r_gl_ex, c_gl, f_gl,
        //   dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl)], r_sm_ex, c_sm,
        //   f_sm);
      } else if (nr % 2 == 0) {
        // if (r == 16 && c == 0 && f == 0) {
        //   printf("padding (%d %d %d) %f <- (%f %f %f)\n", rest_r_p - 1,
        //   c_sm, f_sm,
        //         v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, f_sm)], rest_r
        //         - 1, c_sm, f_sm);
        //   padded = true;
        //   aa = v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, f_sm)];
        //   bb = v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm)];
        // }
        v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, f_sm)];
      }
    }

    if (c_sm == 0) {
      if (rest_c > C * 2) {
        // v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] =
        //     dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl)];
        // printf("load-c[%d %d %d]:%f --> [%d %d %d]\n", r_gl, c_gl_ex, f_gl,
        //   dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl)], r_sm, c_sm_ex,
        //   f_sm);
      } else if (nc % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, f_sm)];
      }
    }

    if (f_sm == 0) {
      if (rest_f > F * 2) {
        // v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)] =
        //     dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl_ex)];
        // printf("load-f[%d %d %d]:%f --> [%d %d %d]\n", r_gl, c_gl, f_gl_ex,
        //   dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl_ex)], r_sm, c_sm,
        //   f_sm_ex);
      } else if (nf % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f_p - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f - 1)];
      }
    }

    // load extra edges
    if (c_sm == 0 && f_sm == 0) {
      if (rest_c > C * 2 && rest_f > F * 2) {
        // v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)] =
        //     dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)];
        // printf("load-cf[%d %d %d]:%f --> [%d %d %d]\n", r_gl, c_gl_ex,
        // f_gl_ex, dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)], r_sm,
        // c_sm_ex, f_sm_ex);
      } else if (rest_c <= C * 2 && rest_f <= F * 2 && nc % 2 == 0 &&
                 nf % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, rest_f_p - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, rest_f - 1)];
      } else if (rest_c > C * 2 && rest_f <= F * 2 && nf % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, rest_f_p - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, rest_f - 1)];
      } else if (rest_c <= C * 2 && rest_f > F * 2 && nc % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, f_sm_ex)];
      }
    }

    if (r_sm == 0 && f_sm == 0) {
      if (rest_r > R * 2 && rest_f > F * 2) {
        // v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)] =
        //     dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)];
        // printf("load-rf[%d %d %d]:%f --> [%d %d %d]\n", r_gl_ex, c_gl,
        // f_gl_ex, dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)],
        // r_sm_ex, c_sm, f_sm_ex);
      } else if (rest_r <= R * 2 && rest_f <= F * 2 && nr % 2 == 0 &&
                 nf % 2 == 0) {
        // printf("padding (%d %d %d) <- (%d %d %d)\n", rest_r_p - 1, c_sm,
        // rest_f_p - 1, rest_r - 1, c_sm, rest_f - 1);
        v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, rest_f_p - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, rest_f - 1)];
      } else if (rest_r > R * 2 && rest_f <= F * 2 && nf % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, rest_f_p - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, rest_f - 1)];
      } else if (rest_r <= R * 2 && rest_f > F * 2 && nr % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, f_sm_ex)];
      }
    }

    if (r_sm == 0 && c_sm == 0) {
      if (rest_r > R * 2 && rest_c > C * 2) {
        // v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)] =
        //     dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)];
        // printf("load-rc[%d %d %d]:%f --> [%d %d %d]\n", r_gl_ex, c_gl_ex,
        // f_gl, dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)], r_sm_ex,
        // c_sm_ex, f_sm);
      } else if (rest_r <= R * 2 && rest_c <= C * 2 && nr % 2 == 0 &&
                 nc % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1, f_sm)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, f_sm)];
        // printf("padding (%d %d %d) <- (%d %d %d): %f\n", rest_r_p - 1,
        // rest_c_p - 1, f_sm, rest_r - 1, rest_c - 1, f_sm,
        // v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, f_sm)]);
      } else if (rest_r > R * 2 && rest_c <= C * 2 && nc % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c - 1, f_sm)];
      } else if (rest_r <= R * 2 && rest_c > C * 2 && nr % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm_ex, f_sm)];
      }
    }
    // load extra vertex

    if (r_sm == 0 && c_sm == 0 && f_sm == 0) {
      if (rest_r > R * 2 && rest_c > C * 2 && rest_f > F * 2) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm_ex)] =
            dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)];
        // printf("load-rcf[%d %d %d]:%f --> [%d %d %d]\n", r_gl_ex, c_gl_ex,
        // f_gl_ex, dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)],
        // r_sm_ex, c_sm_ex, f_sm_ex);
      } else if (rest_r <= R * 2 && rest_c <= C * 2 && rest_f <= F * 2 &&
                 nr % 2 == 0 && nc % 2 == 0 && nf % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1, rest_f_p - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, rest_f - 1)];
      } else if (rest_r > R * 2 && rest_c > C * 2 && rest_f <= F * 2 &&
                 nf % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, rest_f_p - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, rest_f - 1)];
      } else if (rest_r > R * 2 && rest_c <= C * 2 && rest_f > F * 2 &&
                 nc % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c - 1, f_sm_ex)];
      } else if (rest_r > R * 2 && rest_c <= C * 2 && rest_f <= F * 2 &&
                 nc % 2 == 0 && nf % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, rest_f_p - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c - 1, rest_f - 1)];
      } else if (rest_r <= R * 2 && rest_c > C * 2 && rest_f > F * 2 &&
                 nr % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm_ex, f_sm_ex)];
      } else if (rest_r <= R * 2 && rest_c > C * 2 && rest_f <= F * 2 &&
                 nr % 2 == 0 && nf % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, rest_f_p - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm_ex, rest_f - 1)];
      } else if (rest_r <= R * 2 && rest_c <= C * 2 && rest_f > F * 2 &&
                 nr % 2 == 0 && nc % 2 == 0) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1, f_sm_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, f_sm_ex)];
      }
    }

    // asm volatile("membar.cta;");
    // start = clock64() - start;
    // printf("[load extra] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
    // blockIdx.y, blockIdx.x, start); start = clock64();

    // load dist
    if (c_sm == 0 && f_sm == 0 && r_sm < rest_r_p - 2) {
      // printf("%d/%d load %f\n", r_sm, rest_r - 2, dratio_r[r + r_sm]);
      ratio_r_sm[r_sm] = dratio_r[r + r_sm];
      // if (nr % 2 == 0 && R * 2 + 1 >= rest_r_p && r_sm == 0) {
      //   ratio_r_sm[rest_r_p - 3] = 0.5;
      // }
    }
    if (r_sm == 0 && f_sm == 0 && c_sm < rest_c_p - 2) {
      ratio_c_sm[c_sm] = dratio_c[c + c_sm];
      // if (nc % 2 == 0 && C * 2 + 1 >= rest_c_p && c_sm == 0) {
      //   ratio_c_sm[rest_c_p - 3] = 0.5;
      // }
    }
    if (c_sm == 0 && r_sm == 0 && f_sm < rest_f_p - 2) {
      ratio_f_sm[f_sm] = dratio_f[f + f_sm];
      // if (nf % 2 == 0 && F * 2 + 1 >= rest_f_p && f_sm == 0) {
      //   ratio_f_sm[rest_f_p - 3] = 0.5;
      // }
    }

    // if (r == 0 && c == 0 && f == 0 && r_sm == 0 && c_sm == 0 && f_sm == 0)
    // {
    //   printf("ratio:");
    //   for (int i = 0; i < R * 2 + 1; i++) {
    //     printf("%2.2f ", ratio_r_sm[i]);
    //   }
    //   printf("\n");
    // }

  } // restrict boundary

  // asm volatile("membar.cta;");
  // start = clock64() - start;
  // printf("[load ratio] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
  // blockIdx.y, blockIdx.x, start); start = clock64();

  // __syncthreads();
  // // debug print
  // if (debug) {
  //   printf("in config: %d %d %d (%d %d %d)\n", R, C, F, r,c,f);
  //   printf("rest_p: %d %d %d\n", rest_r_p, rest_c_p, rest_f_p);
  //   bool print = false;
  //   for (int i = 0; i < R * 2 + 1; i++) {
  //     for (int j = 0; j < C * 2 + 1; j++) {
  //       for (int k = 0; k < F * 2 + 1; k++) {
  //         // if (abs(v_sm[get_idx(ldsm1, ldsm2, i, j, k)]) > 10000) {
  //           // print = true;
  //           // printf("(block %d %d %d) %2.2f \n", r,c,f,
  //           v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
  //         // printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
  //         // }
  //         printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
  //       }
  //       printf("\n");
  //     }
  //     printf("\n");
  //   }
  // }
  __syncthreads();

  if (dw && threadId < R * C * F) {
    r_sm = (threadId / (C * F)) * 2;
    c_sm = ((threadId % (C * F)) / F) * 2;
    f_sm = ((threadId % (C * F)) % F) * 2;
    r_gl = r / 2 + threadId / (C * F);
    c_gl = c / 2 + threadId % (C * F) / F;
    f_gl = f / 2 + threadId % (C * F) % F;
    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
      // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
      // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
      // r_sm, c_sm, f_sm);
    }
  }

  // asm volatile("membar.cta;");
  // start = clock64() - start;
  // printf("[store coarse] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
  // blockIdx.y, blockIdx.x, start); start = clock64();
  int base = 0;
  // printf("TYPE =%d \n", TYPE);
  // printf("%d == %d && %llu >= %d && %llu < %d\n", r + R * 2, nr_p - 1,
  // threadId, base, threadId, base + C * F);

  if (dw && r + R * 2 == nr_p - 1 && threadId >= base &&
      threadId < base + C * F) {
    r_sm = R * 2;
    c_sm = ((threadId - base) / F) * 2;
    f_sm = ((threadId - base) % F) * 2;
    r_gl = r / 2 + R;
    c_gl = c / 2 + (threadId - base) / F;
    f_gl = f / 2 + (threadId - base) % F;
    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
      // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
      // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
      // r_sm, c_sm, f_sm);
    }
  }

  base += C * F; // ROUND_UP_WARP(C * F) * WARP_SIZE;
  if (dw && c + C * 2 == nc_p - 1 && threadId >= base &&
      threadId < base + R * F) {
    r_sm = ((threadId - base) / F) * 2;
    c_sm = C * 2;
    f_sm = ((threadId - base) % F) * 2;
    r_gl = r / 2 + (threadId - base) / F;
    c_gl = c / 2 + C;
    f_gl = f / 2 + (threadId - base) % F;
    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
      // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
      // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
      // r_sm, c_sm, f_sm);
    }
    // printf("(%d %d %d) (%d %d %d) %f\n",
    //         r_sm, c_sm, f_sm, r_gl, c_gl, f_gl, dwork[get_idx(lddv1, lddv2,
    //         r_gl, c_gl, f_gl)]);
  }

  base += R * F; // ROUND_UP_WARP(R * F) * WARP_SIZE;
  // printf("%d %d\n", base,  threadId);
  if (dw && f + F * 2 == nf_p - 1 && threadId >= base &&
      threadId < base + R * C) {
    r_sm = ((threadId - base) / C) * 2;
    c_sm = ((threadId - base) % C) * 2;
    f_sm = F * 2;
    r_gl = r / 2 + (threadId - base) / C;
    c_gl = c / 2 + (threadId - base) % C;
    f_gl = f / 2 + F;
    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
      // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
      // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
      // r_sm, c_sm, f_sm);
    }
  }

  base += R * C; // ROUND_UP_WARP(R * C) * WARP_SIZE;
  // load extra edges
  if (dw && c + C * 2 == nc_p - 1 && f + F * 2 == nf_p - 1 &&
      threadId >= base && threadId < base + R) {
    r_sm = (threadId - base) * 2;
    c_sm = C * 2;
    f_sm = F * 2;
    r_gl = r / 2 + threadId - base;
    c_gl = c / 2 + C;
    f_gl = f / 2 + F;
    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
      // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
      // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
      // r_sm, c_sm, f_sm);
    }
  }

  base += R; // ROUND_UP_WARP(R) * WARP_SIZE;
  // if (TYPE == 2) printf("%d %d, %d, %llu, %d\n",dw == NULL, f + F * 2, nf_p
  // - 1, threadId, C);
  if (dw && r + R * 2 == nr_p - 1 && f + F * 2 == nf_p - 1 &&
      threadId >= base && threadId < base + C) {
    r_sm = R * 2;
    c_sm = (threadId - base) * 2;
    f_sm = F * 2;
    r_gl = r / 2 + R;
    c_gl = c / 2 + threadId - base;
    f_gl = f / 2 + F;
    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
      // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
      // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
      // r_sm, c_sm, f_sm);
    }
    // printf("store[%d %d %d]: %f\n", r_sm, c_sm, f_sm, v_sm[get_idx(ldsm1,
    // ldsm2, r_sm, c_sm, f_sm)]);
  }

  base += C; // ROUND_UP_WARP(C) * WARP_SIZE;
  if (dw && r + R * 2 == nr_p - 1 && c + C * 2 == nc_p - 1 &&
      threadId >= base && threadId < base + F) {
    r_sm = R * 2;
    c_sm = C * 2;
    f_sm = (threadId - base) * 2;
    r_gl = r / 2 + R;
    c_gl = c / 2 + C;
    f_gl = f / 2 + threadId - base;
    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
      // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
      // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
      // r_sm, c_sm, f_sm);
    }
  }
  base += F; // ROUND_UP_WARP(F) * WARP_SIZE;
  // // load extra vertex
  if (dw && r + R * 2 == nr_p - 1 && c + C * 2 == nc_p - 1 &&
      f + F * 2 == nf_p - 1 && threadId >= base && threadId < base + 1) {
    r_sm = R * 2;
    c_sm = C * 2;
    f_sm = F * 2;
    r_gl = r / 2 + R;
    c_gl = c / 2 + C;
    f_gl = f / 2 + F;
    res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
      // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
      // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
      // r_sm, c_sm, f_sm);
    }
  }

  // asm volatile("membar.cta;");
  // start = clock64() - start;
  // printf("[store extra] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
  // blockIdx.y, blockIdx.x, start); start = clock64();

  // start = clock64();

  if (dwf && threadId >= R * C * F && threadId < R * C * F * 2) {
    r_sm = ((threadId - R * C * F) / (C * F)) * 2;
    c_sm = (((threadId - R * C * F) % (C * F)) / F) * 2;
    f_sm = (((threadId - R * C * F) % (C * F)) % F) * 2 + 1;
    r_gl = r / 2 + (threadId - R * C * F) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F) % (C * F)) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf - nf_c) {
      res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                 v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                 ratio_f_sm[f_sm - 1]);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
      dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)] = res;
    }

    // if (nr == 70)
    // printf("f-store: (%d %d %d) <- %f (%d %d %d)\n", r_gl,
    // c_gl, f_gl, v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)], r_sm, c_sm,
    // f_sm);
    // asm volatile("membar.cta;");
    // start = clock64() - start;
    // printf("[F-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
    // blockIdx.y, blockIdx.x, start); start = clock64();
  }
  // asm volatile("membar.cta;");
  // start = clock64() - start;
  // printf("[F-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
  // blockIdx.y, blockIdx.x, start); start = clock64();

  // if (r_sm % 2 == 0 && c_sm % 2 != 0 && f_sm % 2 == 0) {

  if (dwc && threadId >= R * C * F * 2 && threadId < R * C * F * 3) {
    r_sm = ((threadId - R * C * F * 2) / (C * F)) * 2;
    c_sm = (((threadId - R * C * F * 2) % (C * F)) / F) * 2 + 1;
    f_sm = (((threadId - R * C * F * 2) % (C * F)) % F) * 2;
    r_gl = r / 2 + (threadId - R * C * F * 2) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 2) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 2) % (C * F)) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc - nc_c && f_gl < nf_c) {
      res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                 v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                 ratio_c_sm[c_sm - 1]);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
      dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)] = res;
    }
  }

  // asm volatile("membar.cta;");
  // start = clock64() - start;
  // printf("[C-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
  // blockIdx.y, blockIdx.x, start); start = clock64();

  // if (r_sm % 2 != 0 && c_sm % 2 == 0 && f_sm % 2 == 0) {
  if (dwr && threadId >= R * C * F * 3 && threadId < R * C * F * 4) {
    r_sm = ((threadId - R * C * F * 3) / (C * F)) * 2 + 1;
    c_sm = (((threadId - R * C * F * 3) % (C * F)) / F) * 2;
    f_sm = (((threadId - R * C * F * 3) % (C * F)) % F) * 2;
    r_gl = r / 2 + (threadId - R * C * F * 3) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 3) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 3) % (C * F)) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
        r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
      res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                 v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                 ratio_r_sm[r_sm - 1]);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
      dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)] = res;
    }
  }

  // asm volatile("membar.cta;");
  // start = clock64() - start;
  // printf("[R-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
  // blockIdx.y, blockIdx.x, start); start = clock64();
  __syncthreads();
  if (dwcf && threadId >= R * C * F * 4 && threadId < R * C * F * 5) {
    r_sm = ((threadId - R * C * F * 4) / (C * F)) * 2;
    c_sm = (((threadId - R * C * F * 4) % (C * F)) / F) * 2 + 1;
    f_sm = (((threadId - R * C * F * 4) % (C * F)) % F) * 2 + 1;
    r_gl = r / 2 + (threadId - R * C * F * 4) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 4) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 4) % (C * F)) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc - nc_c && f_gl < nf - nf_c) {
      T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      res = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
      dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)] = res;
    }
  }

  // asm volatile("membar.cta;");
  // start = clock64() - start;
  // printf("[CF-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
  // blockIdx.y, blockIdx.x, start); start = clock64();

  if (dwrf && threadId >= R * C * F * 5 && threadId < R * C * F * 6) {
    r_sm = ((threadId - R * C * F * 5) / (C * F)) * 2 + 1;
    c_sm = (((threadId - R * C * F * 5) % (C * F)) / F) * 2;
    f_sm = (((threadId - R * C * F * 5) % (C * F)) % F) * 2 + 1;
    r_gl = r / 2 + (threadId - R * C * F * 5) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 5) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 5) % (C * F)) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
        r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
      T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      res = lerp(f1, f2, ratio_r_sm[r_sm - 1]);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
      dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)] = res;
    }
  }

  if (dwrc && threadId >= R * C * F * 6 && threadId < R * C * F * 7) {
    r_sm = ((threadId - R * C * F * 6) / (C * F)) * 2 + 1;
    c_sm = (((threadId - R * C * F * 6) % (C * F)) / F) * 2 + 1;
    f_sm = (((threadId - R * C * F * 6) % (C * F)) % F) * 2;
    r_gl = r / 2 + (threadId - R * C * F * 6) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 6) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 6) % (C * F)) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
        r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
      T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                  ratio_c_sm[c_sm - 1]);
      T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                  ratio_c_sm[c_sm - 1]);
      res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
      dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)] = res;
    }
  }

  if (dwrcf && threadId >= R * C * F * 7 && threadId < R * C * F * 8) {
    r_sm = ((threadId - R * C * F * 7) / (C * F)) * 2 + 1;
    c_sm = (((threadId - R * C * F * 7) % (C * F)) / F) * 2 + 1;
    f_sm = (((threadId - R * C * F * 7) % (C * F)) % F) * 2 + 1;
    r_gl = r / 2 + (threadId - R * C * F * 7) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 7) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 7) % (C * F)) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
        r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
      T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      T f3 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      T f4 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);

      T fc1 = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
      T fc2 = lerp(f3, f4, ratio_c_sm[c_sm - 1]);

      res = lerp(fc1, fc2, ratio_r_sm[r_sm - 1]);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
      dwrcf[get_idx(lddwrcf1, lddwrcf2, r_gl, c_gl, f_gl)] = res;
    }
  }
  // end = clock64();

  // asm volatile("membar.cta;");
  // if (threadId < 256 && blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x ==
  // 0) printf("threadId %d elapsed %lu\n", threadId, end-start);
  if (r + R * 2 == nr_p - 1) {
    // printf("test\n");
    if (threadId < C * F) {
      // printf("test1\n");
      if (dwf) {
        // printf("test2\n");
        r_sm = R * 2;
        c_sm = (threadId / F) * 2;
        f_sm = (threadId % F) * 2 + 1;
        r_gl = r / 2 + R;
        c_gl = c / 2 + threadId / F;
        f_gl = f / 2 + threadId % F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
          res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                     ratio_f_sm[f_sm - 1]);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
          // printf("dwf (%d %d %d): %f<-(%f %f %f)\n", r_gl, c_gl, f_gl, res,
          //   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
          //                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
          //                ratio_f_sm[f_sm - 1]);
          dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)] = res;
        }
      }

      if (dwc) {
        r_sm = R * 2;
        c_sm = (threadId / F) * 2 + 1;
        f_sm = (threadId % F) * 2;
        r_gl = r / 2 + R;
        c_gl = c / 2 + threadId / F;
        f_gl = f / 2 + threadId % F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
          res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                     ratio_c_sm[c_sm - 1]);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
          dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)] = res;
        }
      }

      if (dwcf) {
        r_sm = R * 2;
        c_sm = (threadId / F) * 2 + 1;
        f_sm = (threadId % F) * 2 + 1;
        r_gl = r / 2 + R;
        c_gl = c / 2 + threadId / F;
        f_gl = f / 2 + threadId % F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
          T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
          T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
          res = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
          dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)] = res;
        }
      }
    }
  }

  if (c + C * 2 == nc_p - 1) {
    if (threadId >= R * C * F && threadId < R * C * F + R * F) {
      if (dwf) {
        r_sm = ((threadId - R * C * F) / F) * 2;
        c_sm = C * 2;
        f_sm = ((threadId - R * C * F) % F) * 2 + 1;
        r_gl = r / 2 + (threadId - R * C * F) / F;
        c_gl = c / 2 + C;
        f_gl = f / 2 + (threadId - R * C * F) % F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
          res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                     ratio_f_sm[f_sm - 1]);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
          dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)] = res;
        }
      }

      if (dwr) {
        r_sm = ((threadId - R * C * F) / F) * 2 + 1;
        c_sm = C * 2;
        f_sm = ((threadId - R * C * F) % F) * 2;
        r_gl = r / 2 + (threadId - R * C * F) / F;
        c_gl = c / 2 + C;
        f_gl = f / 2 + (threadId - R * C * F) % F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
          res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                     ratio_r_sm[r_sm - 1]);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
          dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)] = res;
        }
      }

      if (dwrf) {
        r_sm = ((threadId - R * C * F) / F) * 2 + 1;
        c_sm = C * 2;
        f_sm = ((threadId - R * C * F) % F) * 2 + 1;
        r_gl = r / 2 + (threadId - R * C * F) / F;
        c_gl = c / 2 + C;
        f_gl = f / 2 + (threadId - R * C * F) % F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
          T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
          T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
          res = lerp(f1, f2, ratio_r_sm[r_sm - 1]);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
          dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)] = res;
        }
      }
    }
  }

  if (f + F * 2 == nf_p - 1) {
    if (threadId >= R * C * F * 2 && threadId < R * C * F * 2 + R * C) {
      if (dwc) {
        r_sm = ((threadId - R * C * F * 2) / C) * 2;
        c_sm = ((threadId - R * C * F * 2) % C) * 2 + 1;
        f_sm = F * 2;
        r_gl = r / 2 + (threadId - R * C * F * 2) / C;
        c_gl = c / 2 + (threadId - R * C * F * 2) % C;
        f_gl = f / 2 + F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
          res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                     ratio_c_sm[c_sm - 1]);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
          dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)] = res;
        }
      }

      if (dwr) {
        r_sm = ((threadId - R * C * F * 2) / C) * 2 + 1;
        c_sm = ((threadId - R * C * F * 2) % C) * 2;
        f_sm = F * 2;
        r_gl = r / 2 + (threadId - R * C * F * 2) / C;
        c_gl = c / 2 + (threadId - R * C * F * 2) % C;
        f_gl = f / 2 + F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
          res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                     ratio_r_sm[r_sm - 1]);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
          dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)] = res;
        }
      }

      if (dwrc) {
        r_sm = ((threadId - R * C * F * 2) / C) * 2 + 1;
        c_sm = ((threadId - R * C * F * 2) % C) * 2 + 1;
        f_sm = F * 2;
        r_gl = r / 2 + (threadId - R * C * F * 2) / C;
        c_gl = c / 2 + (threadId - R * C * F * 2) % C;
        f_gl = f / 2 + F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
          T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                      ratio_c_sm[c_sm - 1]);
          T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                      ratio_c_sm[c_sm - 1]);
          res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
          dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)] = res;
        }
      }
    }
  }

  if (dwr && c + C * 2 == nc_p - 1 && f + F * 2 == nf_p - 1) {
    if (threadId >= R * C * F * 3 && threadId < R * C * F * 3 + R) {
      r_sm = (threadId - R * C * F * 3) * 2 + 1;
      c_sm = C * 2;
      f_sm = F * 2;
      r_gl = r / 2 + threadId - R * C * F * 3;
      c_gl = c / 2 + C;
      f_gl = f / 2 + F;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
        res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                   v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                   ratio_r_sm[r_sm - 1]);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
        dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)] = res;
      }
    }
  }

  if (dwc && r + R * 2 == nr_p - 1 && f + F * 2 == nf_p - 1) {
    if (threadId >= R * C * F * 4 && threadId < R * C * F * 4 + C) {
      r_sm = R * 2;
      c_sm = (threadId - R * C * F * 4) * 2 + 1;
      f_sm = F * 2;
      r_gl = r / 2 + R;
      c_gl = c / 2 + threadId - R * C * F * 4;
      f_gl = f / 2 + F;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
        res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                   ratio_c_sm[c_sm - 1]);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
        dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)] = res;
      }
    }
  }

  if (dwf && r + R * 2 == nr_p - 1 && c + C * 2 == nc_p - 1) {
    if (threadId >= R * C * F * 5 && threadId < R * C * F * 5 + F) {
      r_sm = R * 2;
      c_sm = C * 2;
      f_sm = (threadId - R * C * F * 5) * 2 + 1;
      r_gl = r / 2 + R;
      c_gl = c / 2 + C;
      f_gl = f / 2 + threadId - R * C * F * 5;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
        res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                   ratio_f_sm[f_sm - 1]);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
        dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)] = res;
        // printf("dwf(%d %d %d): %f\n", r_gl, c_gl, f_gl,
        // dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)]);
      }
    }
  }

  // if (r == 0 && c == 0 && f == 0 && threadId == 0) {
  //   printf("out config: %d %d %d (%d %d %d)\n", R, C, F, r,c,f);
  //   for (int i = 0; i < R * 2 + 1; i++) {
  //     for (int j = 0; j < C * 2 + 1; j++) {
  //       for (int k = 0; k < F * 2 + 1; k++) {
  //         printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
  //       }
  //       printf("\n");
  //     }
  //     printf("\n");
  //   }
  // }
}

template <DIM D, typename T, SIZE R, SIZE C, SIZE F>
MGARDm_KERL void
_gpk_reo_3d(SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
            T *dratio_r, T *dratio_c, T *dratio_f, T *dv, SIZE lddv1,
            SIZE lddv2, T *dw, SIZE lddw1, SIZE lddw2, T *dwf, SIZE lddwf1,
            SIZE lddwf2, T *dwc, SIZE lddwc1, SIZE lddwc2, T *dwr, SIZE lddwr1,
            SIZE lddwr2, T *dwcf, SIZE lddwcf1, SIZE lddwcf2, T *dwrf,
            SIZE lddwrf1, SIZE lddwrf2, T *dwrc, SIZE lddwrc1, SIZE lddwrc2,
            T *dwrcf, SIZE lddwrcf1, SIZE lddwrcf2) {

  __gpk_reo_3d<D, T, R, C, F>(
      gridDim.z, gridDim.y, gridDim.x, blockDim.z, blockDim.y, blockDim.x,
      blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
      nr, nc, nf, nr_c, nc_c, nf_c, dratio_r, dratio_c, dratio_f, dv, lddv1,
      lddv2, dw, lddw1, lddw2, dwf, lddwf1, lddwf2, dwc, lddwc1, lddwc2, dwr,
      lddwr1, lddwr2, dwcf, lddwcf1, lddwcf2, dwrf, lddwrf1, lddwrf2, dwrc,
      lddwrc1, lddwrc2, dwrcf, lddwrcf1, lddwrcf2);
}

template <DIM D, typename T, SIZE R, SIZE C, SIZE F>
void gpk_reo_3d_adaptive_launcher(
    Handle<D, T> &handle, SIZE nr, SIZE nc, SIZE nf, T *dratio_r, T *dratio_c,
    T *dratio_f, T *dv, SIZE lddv1, SIZE lddv2, T *dw, SIZE lddw1, SIZE lddw2,
    T *dwf, SIZE lddwf1, SIZE lddwf2, T *dwc, SIZE lddwc1, SIZE lddwc2, T *dwr,
    SIZE lddwr1, SIZE lddwr2, T *dwcf, SIZE lddwcf1, SIZE lddwcf2, T *dwrf,
    SIZE lddwrf1, SIZE lddwrf2, T *dwrc, SIZE lddwrc1, SIZE lddwrc2, T *dwrcf,
    SIZE lddwrcf1, SIZE lddwrcf2, int queue_idx) {

  SIZE nr_c = nr / 2 + 1;
  SIZE nc_c = nc / 2 + 1;
  SIZE nf_c = nf / 2 + 1;
  SIZE total_thread_z = std::max(nr - 1, (SIZE)1);
  SIZE total_thread_y = std::max(nc - 1, (SIZE)1);
  SIZE total_thread_x = std::max(nf - 1, (SIZE)1);

  SIZE tbx, tby, tbz, gridx, gridy, gridz;
  dim3 threadsPerBlock, blockPerGrid;
  size_t sm_size;
  // const int R = 4;
  // const int C = 4;
  // const int F = 16;
  // tbz = std::min(R, total_thread_z);
  // tby = std::min(C, total_thread_y);
  // tbx = std::min(F, total_thread_x);
  tbz = R;
  tby = C;
  tbx = F;
  sm_size = ((R + 1) * (C + 1) * (F + 1) + R + C + F) * sizeof(T);
  gridz = ceil((float)total_thread_z / tbz);
  gridy = ceil((float)total_thread_y / tby);
  gridx = ceil((float)total_thread_x / tbx);
  threadsPerBlock = dim3(tbx, tby, tbz);
  blockPerGrid = dim3(gridx, gridy, gridz);
  // printf("exec config (%d %d %d) (%d %d %d)\n", tbx, tby, tbz, gridx, gridy,
  // gridz);
  _gpk_reo_3d<D, T, R / 2, C / 2, F / 2>
      <<<blockPerGrid, threadsPerBlock, sm_size,
         *(cudaStream_t *)handle.get(queue_idx)>>>(
          nr, nc, nf, nr_c, nc_c, nf_c, dratio_r, dratio_c, dratio_f, dv, lddv1,
          lddv2, dw, lddw1, lddw2, dwf, lddwf1, lddwf2, dwc, lddwc1, lddwc2,
          dwr, lddwr1, lddwr2, dwcf, lddwcf1, lddwcf2, dwrf, lddwrf1, lddwrf2,
          dwrc, lddwrc1, lddwrc2, dwrcf, lddwrcf1, lddwrcf2);
  gpuErrchk(cudaGetLastError());
  if (handle.sync_and_check_all_kernels) {
    gpuErrchk(cudaDeviceSynchronize());
  }
}

template <DIM D, typename T>
void gpk_reo_3d(Handle<D, T> &handle, SIZE nr, SIZE nc, SIZE nf, T *dratio_r,
                T *dratio_c, T *dratio_f, T *dv, SIZE lddv1, SIZE lddv2, T *dw,
                SIZE lddw1, SIZE lddw2, T *dwf, SIZE lddwf1, SIZE lddwf2,
                T *dwc, SIZE lddwc1, SIZE lddwc2, T *dwr, SIZE lddwr1,
                SIZE lddwr2, T *dwcf, SIZE lddwcf1, SIZE lddwcf2, T *dwrf,
                SIZE lddwrf1, SIZE lddwrf2, T *dwrc, SIZE lddwrc1, SIZE lddwrc2,
                T *dwrcf, SIZE lddwrcf1, SIZE lddwrcf2, int queue_idx,
                int config) {

#define GPK(R, C, F)                                                           \
  {                                                                            \
    gpk_reo_3d_adaptive_launcher<D, T, R, C, F>(                               \
        handle, nr, nc, nf, dratio_r, dratio_c, dratio_f, dv, lddv1, lddv2,    \
        dw, lddw1, lddw2, dwf, lddwf1, lddwf2, dwc, lddwc1, lddwc2, dwr,       \
        lddwr1, lddwr2, dwcf, lddwcf1, lddwcf2, dwrf, lddwrf1, lddwrf2, dwrc,  \
        lddwrc1, lddwrc2, dwrcf, lddwrcf1, lddwrcf2, queue_idx);               \
  }
  bool profile = false;
  if (handle.profile_kernels) {
    profile = true;
  }
  if (D == 3) {
    if (profile || config == 6) {
      GPK(2, 2, 128)
    }
    if (profile || config == 5) {
      GPK(2, 2, 64)
    }
    if (profile || config == 4) {
      GPK(4, 4, 32)
    }
    if (profile || config == 3) {
      GPK(4, 4, 16)
    }
    if (profile || config == 2) {
      GPK(4, 4, 8)
    }
    if (profile || config == 1) {
      GPK(4, 4, 4)
    }
    if (profile || config == 0) {
      GPK(2, 2, 2)
    }
    // PI_QL(T, 4, 4, 4)
  } else if (D == 2) {
    if (profile || config == 6) {
      GPK(1, 2, 128)
    }
    if (profile || config == 5) {
      GPK(1, 2, 64)
    }
    if (profile || config == 4) {
      GPK(1, 4, 32)
    }
    if (profile || config == 3) {
      GPK(1, 4, 16)
    }
    if (profile || config == 2) {
      GPK(1, 4, 8)
    }
    if (profile || config == 1) {
      GPK(1, 4, 4)
    }
    if (profile || config == 0) {
      GPK(1, 2, 4)
    }
    // PI_QL(T, 1, 4, 4)
  } else if (D == 1) {
    if (profile || config == 6) {
      GPK(1, 1, 128)
    }
    if (profile || config == 5) {
      GPK(1, 1, 64)
    }
    if (profile || config == 4) {
      GPK(1, 1, 32)
    }
    if (profile || config == 3) {
      GPK(1, 1, 16)
    }
    if (profile || config == 2) {
      GPK(1, 1, 8)
    }
    if (profile || config == 1) {
      GPK(1, 1, 8)
    }
    if (profile || config == 0) {
      GPK(1, 1, 8)
    }
  }
#undef GPK
}

template <DIM D, typename T, SIZE R, SIZE C, SIZE F>
__global__ void
_gpk_rev_3d(SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
            T *dratio_r, T *dratio_c, T *dratio_f, T *dv, SIZE lddv1,
            SIZE lddv2, T *dw, SIZE lddw1, SIZE lddw2, T *dwf, SIZE lddwf1,
            SIZE lddwf2, T *dwc, SIZE lddwc1, SIZE lddwc2, T *dwr, SIZE lddwr1,
            SIZE lddwr2, T *dwcf, SIZE lddwcf1, SIZE lddwcf2, T *dwrf,
            SIZE lddwrf1, SIZE lddwrf2, T *dwrc, SIZE lddwrc1, SIZE lddwrc2,
            T *dwrcf, SIZE lddwrcf1, SIZE lddwrcf2, SIZE svr, SIZE svc,
            SIZE svf, SIZE nvr, SIZE nvc, SIZE nvf) {

  // to be removed
  // int TYPE = 1;
  // bool INTERPOLATION = true;
  // bool COEFF_RESTORE = true;
  // int in_next = false;
  // int skip = false;

  SIZE r = blockIdx.z * blockDim.z;
  SIZE c = blockIdx.y * blockDim.y;
  SIZE f = blockIdx.x * blockDim.x;

  SIZE r_sm = threadIdx.z;
  SIZE c_sm = threadIdx.y;
  SIZE f_sm = threadIdx.x;

  SIZE r_sm_ex = R * 2;
  SIZE c_sm_ex = C * 2;
  SIZE f_sm_ex = F * 2;

  SIZE r_gl;
  SIZE c_gl;
  SIZE f_gl;

  SIZE r_gl_ex;
  SIZE c_gl_ex;
  SIZE f_gl_ex;

  T res;

  LENGTH threadId;

  T *sm = SharedMemory<T>();

  // extern __shared__ double sm[]; // size: (blockDim.x + 1) * (blockDim.y + 1)
  // * (blockDim.z + 1)
  SIZE ldsm1 = F * 2 + 1;
  SIZE ldsm2 = C * 2 + 1;
  T *v_sm = sm;
  T *ratio_f_sm = sm + (F * 2 + 1) * (C * 2 + 1) * (R * 2 + 1);
  T *ratio_c_sm = ratio_f_sm + F * 2;
  T *ratio_r_sm = ratio_c_sm + C * 2;

  SIZE rest_r = nr - r;
  SIZE rest_c = nc - c;
  SIZE rest_f = nf - f;

  SIZE nr_p = nr;
  SIZE nc_p = nc;
  SIZE nf_p = nf;

  SIZE rest_r_p;
  SIZE rest_c_p;
  SIZE rest_f_p;

  rest_r_p = rest_r;
  rest_c_p = rest_c;
  rest_f_p = rest_f;

  threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
             (threadIdx.y * blockDim.x) + threadIdx.x;

  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);

  // // load dist
  // if (c_sm == 0 && f_sm == 0 && r + r_sm < nr) {
  //   ratio_r_sm[r_sm] = dratio_r[r + r_sm];
  // }
  // if (r_sm == 0 && f_sm == 0 && c + c_sm < nc) {
  //   ratio_c_sm[c_sm] = dratio_c[c + c_sm];
  // }
  // if (c_sm == 0 && r_sm == 0 && f + f_sm < nf) {
  //   ratio_f_sm[f_sm] = dratio_f[f + f_sm];
  // }

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

  // load dist
  if (c_sm == 0 && f_sm == 0 && r_sm < rest_r - 2) {
    ratio_r_sm[r_sm] = dratio_r[r + r_sm];
    if (nr % 2 == 0 && R * 2 + 1 >= rest_r_p && r_sm == 0) {
      ratio_r_sm[rest_r_p - 3] = 0.5;
    }
  }
  if (r_sm == 0 && f_sm == 0 && c_sm < rest_c - 2) {
    ratio_c_sm[c_sm] = dratio_c[c + c_sm];
    if (nc % 2 == 0 && C * 2 + 1 >= rest_c_p && c_sm == 0) {
      ratio_c_sm[rest_c_p - 3] = 0.5;
    }
  }
  if (c_sm == 0 && r_sm == 0 && f_sm < rest_f - 2) {
    ratio_f_sm[f_sm] = dratio_f[f + f_sm];
    if (nf % 2 == 0 && F * 2 + 1 >= rest_f_p && f_sm == 0) {
      ratio_f_sm[rest_f_p - 3] = 0.5;
    }
  }

  // if (threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
  //   for (int i = 0; i < R * 2 + 1; i++) {
  //     for (int j = 0; j < C * 2 + 1; j++) {
  //       for (int k = 0; k < F * 2 + 1; k++) {
  //         v_sm[get_idx(ldsm1, ldsm2, i, j, k)] = 71177117;
  //       }
  //     }
  //   }
  // }

  __syncthreads();

  if (dw && threadId < R * C * F) {
    r_sm = (threadId / (C * F)) * 2;
    c_sm = ((threadId % (C * F)) / F) * 2;
    f_sm = ((threadId % (C * F)) % F) * 2;
    r_gl = r / 2 + threadId / (C * F);
    c_gl = c / 2 + threadId % (C * F) / F;
    f_gl = f / 2 + threadId % (C * F) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
      // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
      // printf("block: (%d %d %d) thread: (%d %d %d) load0 (%d %d %d): %f
      // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
      // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
      //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
      //                 r_gl, c_gl, f_gl);
    }
  }

  int base = 0;
  if (dw && threadId >= base && threadId < base + C * F) {
    r_sm = R * 2;
    c_sm = ((threadId - base) / F) * 2;
    f_sm = ((threadId - base) % F) * 2;
    r_gl = r / 2 + R;
    c_gl = c / 2 + (threadId - base) / F;
    f_gl = f / 2 + (threadId - base) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
      // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
      // printf("block: (%d %d %d) thread: (%d %d %d) load1 (%d %d %d): %f
      // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
      // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
      //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
      //                 r_gl, c_gl, f_gl);
    }
  }
  base += C * F; // ROUND_UP_WARP(C * F) * WARP_SIZE;
  if (dw && threadId >= base && threadId < base + R * F) {
    r_sm = ((threadId - base) / F) * 2;
    c_sm = C * 2;
    f_sm = ((threadId - base) % F) * 2;
    r_gl = r / 2 + (threadId - base) / F;
    c_gl = c / 2 + C;
    f_gl = f / 2 + (threadId - base) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
      // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
      // printf("block: (%d %d %d) thread: (%d %d %d) load2 (%d %d %d): %f
      // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
      // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
      //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
      //                 r_gl, c_gl, f_gl);
    }
  }
  base += R * F; // ROUND_UP_WARP(R * F) * WARP_SIZE;
  if (dw && threadId >= base && threadId < base + R * C) {
    r_sm = ((threadId - base) / C) * 2;
    c_sm = ((threadId - base) % C) * 2;
    f_sm = F * 2;
    r_gl = r / 2 + (threadId - base) / C;
    c_gl = c / 2 + (threadId - base) % C;
    f_gl = f / 2 + F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
      // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
      // printf("block: (%d %d %d) thread: (%d %d %d) load3 (%d %d %d): %f
      // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
      // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
      //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
      //                 r_gl, c_gl, f_gl);
    }
  }
  base += R * C; // ROUND_UP_WARP(R * C) * WARP_SIZE;
  // load extra edges
  if (dw && threadId >= base && threadId < base + R) {
    r_sm = (threadId - base) * 2;
    c_sm = C * 2;
    f_sm = F * 2;
    r_gl = r / 2 + threadId - base;
    c_gl = c / 2 + C;
    f_gl = f / 2 + F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
      // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
      // printf("block: (%d %d %d) thread: (%d %d %d) load4 (%d %d %d): %f
      // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
      // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
      //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
      //                 r_gl, c_gl, f_gl);
    }
  }
  base += R; // ROUND_UP_WARP(R) * WARP_SIZE;
  if (dw && threadId >= base && threadId < base + C) {
    r_sm = R * 2;
    c_sm = (threadId - base) * 2;
    f_sm = F * 2;
    r_gl = r / 2 + R;
    c_gl = c / 2 + threadId - base;
    f_gl = f / 2 + F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
      // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
      // printf("block: (%d %d %d) thread: (%d %d %d) load5 (%d %d %d): %f
      // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
      // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
      //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
      //                 r_gl, c_gl, f_gl);
    }
  }
  base += C; // ROUND_UP_WARP(C) * WARP_SIZE;
  if (dw && threadId >= base && threadId < base + F) {
    r_sm = R * 2;
    c_sm = C * 2;
    f_sm = (threadId - base) * 2;
    r_gl = r / 2 + R;
    c_gl = c / 2 + C;
    f_gl = f / 2 + threadId - base;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
      // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
      // printf("block: (%d %d %d) thread: (%d %d %d) load6 (%d %d %d): %f
      // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
      // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
      //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
      //                 r_gl, c_gl, f_gl);
    }
  }
  base += F; // ROUND_UP_WARP(F) * WARP_SIZE;
  // // load extra vertex
  if (dw && threadId >= base && threadId < base + 1) {
    r_sm = R * 2;
    c_sm = C * 2;
    f_sm = F * 2;
    r_gl = r / 2 + R;
    c_gl = c / 2 + C;
    f_gl = f / 2 + F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf_c) {
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
      // if (c_gl == nc_c - 1 && f_gl == nf_c-1)
      // printf("block: (%d %d %d) thread: (%d %d %d) load7 (%d %d %d): %f
      // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
      // threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
      //               v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)],
      //                 r_gl, c_gl, f_gl);
    }
  }

  __syncthreads();

  // __syncthreads();
  // if (threadIdx.x == 0 && threadIdx.y == 0&& threadIdx.z == 0) {
  //   printf("rest_p: %u %u %u RCF\n", rest_r_p, rest_c_p, rest_f_p, R, C, F);
  //   for (int i = 0; i < min(rest_r_p, R * 2 + 1); i++) {
  //     for (int j = 0; j < min(rest_c_p, C * 2 + 1); j++) {
  //       for (int k = 0; k < min(rest_f_p, F * 2 + 1); k++) {
  //         printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
  //       }
  //       printf("\n");
  //     }
  //     printf("\n");
  //   }
  // }
  // __syncthreads();

  // __syncthreads();
  // if (debug) {
  //   printf("TYPE: %d %d %d %d\n", TYPE,
  //           min(rest_r_p, R * 2 + 1),
  //           min(rest_c_p, C * 2 + 1),
  //           min(rest_f_p, F * 2 + 1));
  //   for (int i = 0; i < min(rest_r_p, R * 2 + 1); i++) {
  //     for (int j = 0; j < min(rest_c_p, C * 2 + 1); j++) {
  //       for (int k = 0; k < min(rest_f_p, F * 2 + 1); k++) {
  //         printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
  //       }
  //       printf("\n");
  //     }
  //     printf("\n");
  //   }
  // }
  // __syncthreads();

  if (dwf && threadId >= R * C * F && threadId < R * C * F * 2) {

    r_sm = ((threadId - R * C * F) / (C * F)) * 2;
    c_sm = (((threadId - R * C * F) % (C * F)) / F) * 2;
    f_sm = (((threadId - R * C * F) % (C * F)) % F) * 2 + 1;
    r_gl = r / 2 + (threadId - R * C * F) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F) % (C * F)) % F;

    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc_c && f_gl < nf - nf_c) {

      res = dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
      res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
    }
  }

  if (dwc && threadId >= R * C * F * 2 && threadId < R * C * F * 3) {
    r_sm = ((threadId - R * C * F * 2) / (C * F)) * 2;
    c_sm = (((threadId - R * C * F * 2) % (C * F)) / F) * 2 + 1;
    f_sm = (((threadId - R * C * F * 2) % (C * F)) % F) * 2;
    r_gl = r / 2 + (threadId - R * C * F * 2) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 2) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 2) % (C * F)) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc - nc_c && f_gl < nf_c) {
      res = dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
      res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                  ratio_c_sm[c_sm - 1]);
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
    }
  }

  if (dwr && threadId >= R * C * F * 3 && threadId < R * C * F * 4) {
    r_sm = ((threadId - R * C * F * 3) / (C * F)) * 2 + 1;
    c_sm = (((threadId - R * C * F * 3) % (C * F)) / F) * 2;
    f_sm = (((threadId - R * C * F * 3) % (C * F)) % F) * 2;
    r_gl = r / 2 + (threadId - R * C * F * 3) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 3) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 3) % (C * F)) % F;

    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
        r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
      res = dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
      res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                  ratio_r_sm[r_sm - 1]);
      // if (c_gl == nc_c-1 && f_gl == nf_c - 1)
      //     printf("block: (%d %d %d) thread: (%d %d %d) calc_coeff0 (%d
      //     %d %d): %f <- %f %f\n", blockIdx.z, blockIdx.y, blockIdx.x,
      //     threadIdx.z, threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
      //             res, v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm,
      //             f_sm)],
      //               v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)]);
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
    }
  }

  if (dwcf && threadId >= R * C * F * 4 && threadId < R * C * F * 5) {
    r_sm = ((threadId - R * C * F * 4) / (C * F)) * 2;
    c_sm = (((threadId - R * C * F * 4) % (C * F)) / F) * 2 + 1;
    f_sm = (((threadId - R * C * F * 4) % (C * F)) % F) * 2 + 1;
    r_gl = r / 2 + (threadId - R * C * F * 4) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 4) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 4) % (C * F)) % F;

    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p && r_gl < nr_c &&
        c_gl < nc - nc_c && f_gl < nf - nf_c) {
      res = dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)];
      T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      res += lerp(f1, f2, ratio_c_sm[c_sm - 1]);
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
    }
  }

  if (dwrf && threadId >= R * C * F * 5 && threadId < R * C * F * 6) {
    r_sm = ((threadId - R * C * F * 5) / (C * F)) * 2 + 1;
    c_sm = (((threadId - R * C * F * 5) % (C * F)) / F) * 2;
    f_sm = (((threadId - R * C * F * 5) % (C * F)) % F) * 2 + 1;
    r_gl = r / 2 + (threadId - R * C * F * 5) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 5) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 5) % (C * F)) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
        r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
      res = dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)];
      T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      res += lerp(f1, f2, ratio_r_sm[r_sm - 1]);
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
    }
  }

  if (dwrc && threadId >= R * C * F * 6 && threadId < R * C * F * 7) {
    r_sm = ((threadId - R * C * F * 6) / (C * F)) * 2 + 1;
    c_sm = (((threadId - R * C * F * 6) % (C * F)) / F) * 2 + 1;
    f_sm = (((threadId - R * C * F * 6) % (C * F)) % F) * 2;
    r_gl = r / 2 + (threadId - R * C * F * 6) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 6) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 6) % (C * F)) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
        r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
      res = dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)];
      T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                  ratio_c_sm[c_sm - 1]);
      T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                  ratio_c_sm[c_sm - 1]);
      res += lerp(c1, c2, ratio_r_sm[r_sm - 1]);
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
    }
  }

  if (dwrcf && threadId >= R * C * F * 7 && threadId < R * C * F * 8) {
    r_sm = ((threadId - R * C * F * 7) / (C * F)) * 2 + 1;
    c_sm = (((threadId - R * C * F * 7) % (C * F)) / F) * 2 + 1;
    f_sm = (((threadId - R * C * F * 7) % (C * F)) % F) * 2 + 1;
    r_gl = r / 2 + (threadId - R * C * F * 7) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 7) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 7) % (C * F)) % F;
    if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
        r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
      res = dwrcf[get_idx(lddwrcf1, lddwrcf2, r_gl, c_gl, f_gl)];
      T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      T f3 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
      T f4 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);

      T fc1 = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
      T fc2 = lerp(f3, f4, ratio_c_sm[c_sm - 1]);

      res += lerp(fc1, fc2, ratio_r_sm[r_sm - 1]);
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
    }
  }

  if (r + R * 2 == nr_p - 1) {
    if (threadId < C * F) {
      if (dwf) {
        r_sm = R * 2;
        c_sm = (threadId / F) * 2;
        f_sm = (threadId % F) * 2 + 1;
        r_gl = r / 2 + R;
        c_gl = c / 2 + threadId / F;
        f_gl = f / 2 + threadId % F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
          res = dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
          res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
        }
      }

      if (dwc) {
        r_sm = R * 2;
        c_sm = (threadId / F) * 2 + 1;
        f_sm = (threadId % F) * 2;
        r_gl = r / 2 + R;
        c_gl = c / 2 + threadId / F;
        f_gl = f / 2 + threadId % F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
          res = dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
          res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                      ratio_c_sm[c_sm - 1]);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
        }
      }
      if (dwcf) {
        r_sm = R * 2;
        c_sm = (threadId / F) * 2 + 1;
        f_sm = (threadId % F) * 2 + 1;
        r_gl = r / 2 + R;
        c_gl = c / 2 + threadId / F;
        f_gl = f / 2 + threadId % F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
          res = dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)];
          T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
          T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
          res += lerp(f1, f2, ratio_c_sm[c_sm - 1]);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
        }
      }
    }
  }

  if (c + C * 2 == nc_p - 1) {
    if (threadId >= R * C * F && threadId < R * C * F + R * F) {
      if (dwf) {
        r_sm = ((threadId - R * C * F) / F) * 2;
        c_sm = C * 2;
        f_sm = ((threadId - R * C * F) % F) * 2 + 1;
        r_gl = r / 2 + (threadId - R * C * F) / F;
        c_gl = c / 2 + C;
        f_gl = f / 2 + (threadId - R * C * F) % F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
          res = dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
          res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
        }
      }
      if (dwr) {
        r_sm = ((threadId - R * C * F) / F) * 2 + 1;
        c_sm = C * 2;
        f_sm = ((threadId - R * C * F) % F) * 2;
        r_gl = r / 2 + (threadId - R * C * F) / F;
        c_gl = c / 2 + C;
        f_gl = f / 2 + (threadId - R * C * F) % F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
          res = dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
          res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                      ratio_r_sm[r_sm - 1]);
          // if (c_gl == nc_c-1 && f_gl == nf_c - 1)
          // printf("block: (%d %d %d) thread: (%d %d %d) calc_coeff1 (%d
          // %d %d): %f <- %f %f\n", blockIdx.z, blockIdx.y, blockIdx.x,
          // threadIdx.z, threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
          //         res, v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm,
          //         f_sm)],
          //           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)]);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
        }
      }
      if (dwrf) {
        r_sm = ((threadId - R * C * F) / F) * 2 + 1;
        c_sm = C * 2;
        f_sm = ((threadId - R * C * F) % F) * 2 + 1;
        r_gl = r / 2 + (threadId - R * C * F) / F;
        c_gl = c / 2 + C;
        f_gl = f / 2 + (threadId - R * C * F) % F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
          res = dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)];
          T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
          T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
          res += lerp(f1, f2, ratio_r_sm[r_sm - 1]);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
        }
      }
    }
  }

  if (f + F * 2 == nf_p - 1) {
    if (threadId >= R * C * F * 2 && threadId < R * C * F * 2 + R * C) {
      if (dwc) {
        r_sm = ((threadId - R * C * F * 2) / C) * 2;
        c_sm = ((threadId - R * C * F * 2) % C) * 2 + 1;
        f_sm = F * 2;
        r_gl = r / 2 + (threadId - R * C * F * 2) / C;
        c_gl = c / 2 + (threadId - R * C * F * 2) % C;
        f_gl = f / 2 + F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
          res = dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
          res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                      ratio_c_sm[c_sm - 1]);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
        }
      }

      if (dwr) {
        r_sm = ((threadId - R * C * F * 2) / C) * 2 + 1;
        c_sm = ((threadId - R * C * F * 2) % C) * 2;
        f_sm = F * 2;
        r_gl = r / 2 + (threadId - R * C * F * 2) / C;
        c_gl = c / 2 + (threadId - R * C * F * 2) % C;
        f_gl = f / 2 + F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
          res = dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
          res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                      ratio_r_sm[r_sm - 1]);
          // if (c_gl == nc_c-1 && f_gl == nf_c - 1)
          // printf("block: (%d %d %d) thread: (%d %d %d) calc_coeff2 (%d
          // %d %d): %f <- %f %f\n", blockIdx.z, blockIdx.y, blockIdx.x,
          // threadIdx.z, threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
          //         res, v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm,
          //         f_sm)],
          //           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)]);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
        }
      }

      if (dwrc) {
        r_sm = ((threadId - R * C * F * 2) / C) * 2 + 1;
        c_sm = ((threadId - R * C * F * 2) % C) * 2 + 1;
        f_sm = F * 2;
        r_gl = r / 2 + (threadId - R * C * F * 2) / C;
        c_gl = c / 2 + (threadId - R * C * F * 2) % C;
        f_gl = f / 2 + F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
          res = dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)];
          T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                      ratio_c_sm[c_sm - 1]);
          T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                      ratio_c_sm[c_sm - 1]);
          res += lerp(c1, c2, ratio_r_sm[r_sm - 1]);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
        }
      }
    }
  }

  if (c + C * 2 == nc_p - 1 && f + F * 2 == nf_p - 1) {
    if (threadId >= R * C * F * 3 && threadId < R * C * F * 3 + R) {
      if (dwr) {
        r_sm = (threadId - R * C * F * 3) * 2 + 1;
        c_sm = C * 2;
        f_sm = F * 2;
        r_gl = r / 2 + threadId - R * C * F * 3;
        c_gl = c / 2 + C;
        f_gl = f / 2 + F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
          res = dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
          res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                      ratio_r_sm[r_sm - 1]);
          // if (c_gl == nc_c-1 && f_gl == nf_c - 1)
          // printf("block: (%d %d %d) thread: (%d %d %d) calc_coeff3 (%d
          // %d %d): %f <- %f %f\n", blockIdx.z, blockIdx.y, blockIdx.x,
          // threadIdx.z, threadIdx.y, threadIdx.x, r_sm, c_sm, f_sm,
          //         res, v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm,
          //         f_sm)],
          //           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)]);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
        }
      }
    }
  }

  if (r + R * 2 == nr_p - 1 && f + F * 2 == nf_p - 1) {
    if (threadId >= R * C * F * 4 && threadId < R * C * F * 4 + C) {
      if (dwc) {
        r_sm = R * 2;
        c_sm = (threadId - R * C * F * 4) * 2 + 1;
        f_sm = F * 2;
        r_gl = r / 2 + R;
        c_gl = c / 2 + threadId - R * C * F * 4;
        f_gl = f / 2 + F;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
          res = dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
          res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                      ratio_c_sm[c_sm - 1]);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
        }
      }
    }
  }

  if (r + R * 2 == nr_p - 1 && c + C * 2 == nc_p - 1) {
    if (threadId >= R * C * F * 5 && threadId < R * C * F * 5 + F) {
      if (dwf) {
        r_sm = R * 2;
        c_sm = C * 2;
        f_sm = (threadId - R * C * F * 5) * 2 + 1;
        r_gl = r / 2 + R;
        c_gl = c / 2 + C;
        f_gl = f / 2 + threadId - R * C * F * 5;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
          res = dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
          res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
        }
      }
    }
  }

  // __syncthreads();
  // if (debug) {
  //   printf("TYPE: %d %d %d %d\n", TYPE,
  //           min(rest_r_p, R * 2 + 1),
  //           min(rest_c_p, C * 2 + 1),
  //           min(rest_f_p, F * 2 + 1));
  //   for (int i = 0; i < min(rest_r_p, R * 2 + 1); i++) {
  //     for (int j = 0; j < min(rest_c_p, C * 2 + 1); j++) {
  //       for (int k = 0; k < min(rest_f_p, F * 2 + 1); k++) {
  //         printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
  //       }
  //       printf("\n");
  //     }
  //     printf("\n");
  //   }
  // }
  // __syncthreads();

  __syncthreads();

  r_sm = threadIdx.z;
  c_sm = threadIdx.y;
  f_sm = threadIdx.x;

  r_sm_ex = blockDim.z;
  c_sm_ex = blockDim.y;
  f_sm_ex = blockDim.x;

  r_gl = r + r_sm;
  c_gl = c + c_sm;
  f_gl = f + f_sm;

  // r_gl_ex = r + R * 2;
  // c_gl_ex = c + C * 2;
  // f_gl_ex = f + F * 2;

  r_gl_ex = r + rest_r - 1;
  c_gl_ex = c + rest_c - 1;
  f_gl_ex = f + rest_f - 1;

  int unpadding_r = rest_r;
  int unpadding_c = rest_c;
  int unpadding_f = rest_f;
  if (nr % 2 == 0)
    unpadding_r -= 1;
  if (nc % 2 == 0)
    unpadding_c -= 1;
  if (nf % 2 == 0)
    unpadding_f -= 1;

  if (r_sm < unpadding_r && c_sm < unpadding_c && f_sm < unpadding_f) {

    // store extra rules
    // case 1: input = odd (non-padding required)
    //    case 1.a: block size + 1 == rest (need to store extra);
    //    case 1.b: block size + 1 != rest (No need to store extra);
    // case 2: input = even (un-padding requried)
    //    case 2.a: block size + 1 >= rest (No need to store extra, but need
    //    un-padding first); case 2.b: block size + 1 < rest (No need to store
    //    extra);

    if (D >= 3 && r_sm == 0) {
      if (nr % 2 != 0 && R * 2 + 1 == rest_r) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)];
      }
      if (nr % 2 == 0 && R * 2 + 1 >= rest_r_p) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, f_sm)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm)];
        // if ( v_sm[get_idx(ldsm1, ldsm2, rest_r_p-1, c_sm, f_sm)] == 71177117)
        // printf("un-padding0 error block: (%d %d %d) thread: (%d %d %d)
        // un-padding (%d %d %d) %f (%d %d %d)\n", blockIdx.z, blockIdx.y,
        // blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
        //   rest_r-1, c_sm, f_sm,
        //     v_sm[get_idx(ldsm1, ldsm2, rest_r_p-1, c_sm, f_sm)], rest_r_p-1,
        //     c_sm, f_sm);
      }
    }

    if (D >= 2 && c_sm == 0) {
      if (nc % 2 != 0 && C * 2 + 1 == rest_c) {
        dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)];
      }
      if (nc % 2 == 0 && C * 2 + 1 >= rest_c_p) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, f_sm)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm)];
        // if (v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm)] ==
        // 71177117)
        //   printf("un-padding1 error block: (%d %d %d) thread: (%d %d %d) "
        //          "un-padding (%d %d %d) %f (%d %d %d)\n",
        //          blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
        //          threadIdx.y, threadIdx.x, r_sm, rest_c - 1, f_sm,
        //          v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm)], r_sm,
        //          rest_c_p - 1, f_sm);
      }
    }

    if (D >= 1 && f_sm == 0) {
      if (nf % 2 != 0 && F * 2 + 1 == rest_f) {
        dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)];
      }
      if (nf % 2 == 0 && F * 2 + 1 >= rest_f_p) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f_p - 1)];
        // if ( v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f_p-1)] == 71177117)
        // printf("un-padding2 error block: (%d %d %d) thread: (%d %d %d)
        // un-padding (%d %d %d) %f (%d %d %d)\n", blockIdx.z, blockIdx.y,
        // blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
        //   r_sm, c_sm, rest_f-1,
        //     v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f_p-1)], r_sm, c_sm,
        //     rest_f_p-1);
      }
    }

    // load extra edges
    if (D >= 2 && c_sm == 0 && f_sm == 0) {
      if (nc % 2 != 0 && C * 2 + 1 == rest_c && nf % 2 != 0 &&
          F * 2 + 1 == rest_f) {
        dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)];
      }
      if (nc % 2 == 0 && nf % 2 == 0 && C * 2 + 1 >= rest_c_p &&
          F * 2 + 1 >= rest_f_p) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, rest_f - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, rest_f_p - 1)];
        // printf("block: (%d %d %d) thread: (%d %d %d) un-padding (%d %d %d) %f
        // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
        // threadIdx.y, threadIdx.x, r_sm, rest_c-1, rest_f-1,
        //     v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c-1, rest_f-1)], r_sm,
        //     rest_c_p-1, rest_f_p-1);
      }
      if (nc % 2 == 0 && nf % 2 != 0 && C * 2 + 1 >= rest_c_p &&
          F * 2 + 1 == rest_f) {
        dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm_ex)];
      }
      if (nc % 2 != 0 && nf % 2 == 0 && C * 2 + 1 == rest_c &&
          F * 2 + 1 >= rest_f_p) {
        dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, rest_f_p - 1)];
        // printf("(%d %d %d): %f <- (%d %d %d)\n",
        //         r_gl, c_gl_ex, f_gl_ex,
        //         dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)],
        //         r_sm, c_sm_ex, f_gl_ex);
      }
    }

    if (D >= 3 && r_sm == 0 && f_sm == 0) {
      if (nr % 2 != 0 && R * 2 + 1 == rest_r && nf % 2 != 0 &&
          F * 2 + 1 == rest_f) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)];
      }
      if (nr % 2 == 0 && nf % 2 == 0 && R * 2 + 1 >= rest_r_p &&
          F * 2 + 1 >= rest_f_p) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, rest_f - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, rest_f_p - 1)];
        // if ( v_sm[get_idx(ldsm1, ldsm2, rest_r_p-1, c_sm, rest_f_p-1)] ==
        // 71177117) printf("un-padding3 error block: (%d %d %d) thread: (%d %d
        // %d) un-padding (%d %d %d) %f (%d %d %d)\n", blockIdx.z, blockIdx.y,
        // blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
        //   rest_r-1, c_sm, rest_f-1,
        //     v_sm[get_idx(ldsm1, ldsm2, rest_r_p-1, c_sm, rest_f_p-1)],
        //     rest_r_p-1, c_sm, rest_f_p-1);
      }
      if (nr % 2 == 0 && nf % 2 != 0 && R * 2 + 1 >= rest_r_p &&
          F * 2 + 1 == rest_f) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm_ex)];
      }
      if (nr % 2 != 0 && nf % 2 == 0 && R * 2 + 1 == rest_r &&
          F * 2 + 1 >= rest_f_p) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, rest_f_p - 1)];
        // printf("(%d %d %d): %f <- (%d %d %d)\n",
        //         r_gl_ex, c_gl, rest_f-1,
        //         dv[get_idx(lddv1, lddv2, r_gl_ex-1, c_gl, f_gl_ex)],
        //         r_sm_ex, c_sm, rest_f_p-1);
      }
    }

    if (D >= 3 && r_sm == 0 && c_sm == 0) {
      if (nr % 2 != 0 && R * 2 + 1 == rest_r && nc % 2 != 0 &&
          C * 2 + 1 == rest_c) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)];
      }
      if (nr % 2 == 0 && nc % 2 == 0 && R * 2 + 1 >= rest_r_p &&
          C * 2 + 1 >= rest_c_p) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, f_sm)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1, f_sm)];
        // if ( v_sm[get_idx(ldsm1, ldsm2, rest_r_p-1, rest_c_p-1, f_sm)] ==
        // 71177117) printf("un-padding4 error block: (%d %d %d) thread: (%d %d
        // %d) un-padding (%d %d %d) %f (%d %d %d)\n", blockIdx.z, blockIdx.y,
        // blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
        //   rest_r-1, rest_c-1, f_sm,
        //     v_sm[get_idx(ldsm1, ldsm2, rest_r_p-1, rest_c_p-1, f_sm)],
        //     rest_r_p-1, rest_c_p-1, f_sm);
      }
      if (nr % 2 == 0 && nc % 2 != 0 && R * 2 + 1 >= rest_r_p &&
          C * 2 + 1 == rest_c) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm)];
      }
      if (nr % 2 != 0 && nc % 2 == 0 && R * 2 + 1 == rest_r &&
          C * 2 + 1 >= rest_c_p) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm)];
      }
    }
    // load extra vertex

    if (D >= 3 && r_sm == 0 && c_sm == 0 && f_sm == 0) {
      if (nr % 2 != 0 && R * 2 + 1 == rest_r && nc % 2 != 0 &&
          C * 2 + 1 == rest_c && nf % 2 != 0 && F * 2 + 1 == rest_f) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm_ex)];
      }

      if (nr % 2 == 0 && nc % 2 == 0 && nf % 2 == 0 && R * 2 + 1 >= rest_r_p &&
          C * 2 + 1 >= rest_c_p && F * 2 + 1 >= rest_f_p) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, rest_f - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1,
                         rest_f_p - 1)];

        // printf("block: (%d %d %d) thread: (%d %d %d) un-padding (%d %d %d) %f
        // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
        // threadIdx.y, threadIdx.x, rest_r-1, rest_c-1, rest_f-1,
        //     v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c-1, rest_f-1)],
        //     rest_r_p-1, rest_c_p-1, rest_f_p-1);
      }
      if (nr % 2 == 0 && nc % 2 == 0 && nf % 2 != 0 && R * 2 + 1 >= rest_r_p &&
          C * 2 + 1 >= rest_c_p && F * 2 + 1 == rest_f) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1, f_sm_ex)];
      }
      if (nr % 2 == 0 && nc % 2 != 0 && nf % 2 == 0 && R * 2 + 1 >= rest_r_p &&
          C * 2 + 1 == rest_c && F * 2 + 1 >= rest_f_p) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, rest_f_p - 1)];
      }
      if (nr % 2 != 0 && nc % 2 == 0 && nf % 2 == 0 && R * 2 + 1 == rest_r &&
          C * 2 + 1 >= rest_c_p && F * 2 + 1 >= rest_f_p) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, rest_f_p - 1)];
      }
      if (nr % 2 == 0 && nc % 2 != 0 && nf % 2 != 0 && R * 2 + 1 >= rest_r_p &&
          C * 2 + 1 == rest_c && F * 2 + 1 == rest_f) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm_ex)];
      }
      if (nr % 2 != 0 && nc % 2 == 0 && nf % 2 != 0 && R * 2 + 1 == rest_r &&
          C * 2 + 1 >= rest_c_p && F * 2 + 1 == rest_f) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm_ex)];
      }
      if (nr % 2 != 0 && nc % 2 != 0 && nf % 2 == 0 && R * 2 + 1 == rest_r &&
          C * 2 + 1 == rest_c && F * 2 + 1 >= rest_f_p) {
        dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, rest_f_p - 1)];
      }
    }
  }

  __syncthreads();

  if (r_sm < rest_r && c_sm < rest_c && f_sm < rest_f) {
    if (r_gl >= svr && r_gl < svr + nvr && c_gl >= svc && c_gl < svc + nvc &&
        f_gl >= svf && f_gl < svf + nvf) {
      dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl)] =
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];

      // if (c_gl == nc - 1 && f_gl == nf - 1) {
      //   printf("block: (%d %d %d) thread: (%d %d %d) store (%d %d %d) %f
      //   (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
      //   threadIdx.y, threadIdx.x, r_gl, c_gl, f_gl,
      //     v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)], r_sm, c_sm, f_sm);
      // }
    }
  }
}

template <DIM D, typename T, SIZE R, SIZE C, SIZE F>
void gpk_rev_3d_adaptive_launcher(
    Handle<D, T> &handle, SIZE nr, SIZE nc, SIZE nf, T *dratio_r, T *dratio_c,
    T *dratio_f, T *dv, SIZE lddv1, SIZE lddv2, T *dw, SIZE lddw1, SIZE lddw2,
    T *dwf, SIZE lddwf1, SIZE lddwf2, T *dwc, SIZE lddwc1, SIZE lddwc2, T *dwr,
    SIZE lddwr1, SIZE lddwr2, T *dwcf, SIZE lddwcf1, SIZE lddwcf2, T *dwrf,
    SIZE lddwrf1, SIZE lddwrf2, T *dwrc, SIZE lddwrc1, SIZE lddwrc2, T *dwrcf,
    SIZE lddwrcf1, SIZE lddwrcf2, SIZE svr, SIZE svc, SIZE svf, SIZE nvr,
    SIZE nvc, SIZE nvf, int queue_idx) {
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  SIZE nr_c = nr / 2 + 1;
  SIZE nc_c = nc / 2 + 1;
  SIZE nf_c = nf / 2 + 1;
  SIZE total_thread_z = std::max(nr - 1, (SIZE)1);
  SIZE total_thread_y = std::max(nc - 1, (SIZE)1);
  SIZE total_thread_x = std::max(nf - 1, (SIZE)1);

  SIZE tbx, tby, tbz, gridx, gridy, gridz;
  dim3 threadsPerBlock, blockPerGrid;
  size_t sm_size;

  // tbz = std::min(R, total_thread_z);
  // tby = std::min(C, total_thread_y);
  // tbx = std::min(F, total_thread_x);
  tbz = R;
  tby = C;
  tbx = F;
  sm_size = ((R + 1) * (C + 1) * (F + 1) + R + C + F) * sizeof(T);
  gridz = ceil((float)total_thread_z / tbz);
  gridy = ceil((float)total_thread_y / tby);
  gridx = ceil((float)total_thread_x / tbx);
  threadsPerBlock = dim3(tbx, tby, tbz);
  blockPerGrid = dim3(gridx, gridy, gridz);
  // printf("prolongate exec: %d %d %d %d %d %d\n", tbx, tby, tbz, gridx, gridy,
  // gridz);
  _gpk_rev_3d<D, T, R / 2, C / 2, F / 2>
      <<<blockPerGrid, threadsPerBlock, sm_size,
         *(cudaStream_t *)handle.get(queue_idx)>>>(
          nr, nc, nf, nr_c, nc_c, nf_c, dratio_r, dratio_c, dratio_f, dv, lddv1,
          lddv2, dw, lddw1, lddw2, dwf, lddwf1, lddwf2, dwc, lddwc1, lddwc2,
          dwr, lddwr1, lddwr2, dwcf, lddwcf1, lddwcf2, dwrf, lddwrf1, lddwrf2,
          dwrc, lddwrc1, lddwrc2, dwrcf, lddwrcf1, lddwrcf2, svr, svc, svf, nvr,
          nvc, nvf);
  gpuErrchk(cudaGetLastError());
  if (handle.sync_and_check_all_kernels) {
    gpuErrchk(cudaDeviceSynchronize());
  }
}

template <DIM D, typename T>
void gpk_rev_3d(Handle<D, T> &handle, SIZE nr, SIZE nc, SIZE nf, T *dratio_r,
                T *dratio_c, T *dratio_f, T *dv, SIZE lddv1, SIZE lddv2, T *dw,
                SIZE lddw1, SIZE lddw2, T *dwf, SIZE lddwf1, SIZE lddwf2,
                T *dwc, SIZE lddwc1, SIZE lddwc2, T *dwr, SIZE lddwr1,
                SIZE lddwr2, T *dwcf, SIZE lddwcf1, SIZE lddwcf2, T *dwrf,
                SIZE lddwrf1, SIZE lddwrf2, T *dwrc, SIZE lddwrc1, SIZE lddwrc2,
                T *dwrcf, SIZE lddwrcf1, SIZE lddwrcf2, SIZE svr, SIZE svc,
                SIZE svf, SIZE nvr, SIZE nvc, SIZE nvf, int queue_idx,
                int config) {

#define GPK(R, C, F)                                                           \
  {                                                                            \
    gpk_rev_3d_adaptive_launcher<D, T, R, C, F>(                               \
        handle, nr, nc, nf, dratio_r, dratio_c, dratio_f, dv, lddv1, lddv2,    \
        dw, lddw1, lddw2, dwf, lddwf1, lddwf2,\ 
                            dwc,                                               \
        lddwc1, lddwc2, dwr, lddwr1, lddwr2, dwcf, lddwcf1, lddwcf2,\ 
                            dwrf,                                              \
        lddwrf1, lddwrf2, dwrc, lddwrc1, lddwrc2, dwrcf, lddwrcf1, lddwrcf2,   \
        svr, svc, svf, nvr, nvc, nvf, queue_idx);                              \
  }
  bool profile = false;
  if (handle.profile_kernels) {
    profile = true;
  }
  if (D == 3) {
    if (profile || config == 6) {
      GPK(2, 2, 128)
    }
    if (profile || config == 5) {
      GPK(2, 2, 64)
    }
    if (profile || config == 4) {
      GPK(4, 4, 32)
    }
    if (profile || config == 3) {
      GPK(4, 4, 16)
    }
    if (profile || config == 2) {
      GPK(4, 4, 8)
    }
    if (profile || config == 1) {
      GPK(4, 4, 4)
    }
    if (profile || config == 0) {
      GPK(2, 2, 2)
    }
  } else if (D == 2) {
    if (profile || config == 6) {
      GPK(1, 2, 128)
    }
    if (profile || config == 5) {
      GPK(1, 2, 64)
    }
    if (profile || config == 4) {
      GPK(1, 4, 32)
    }
    if (profile || config == 3) {
      GPK(1, 4, 16)
    }
    if (profile || config == 2) {
      GPK(1, 4, 8)
    }
    if (profile || config == 1) {
      GPK(1, 4, 4)
    }
    if (profile || config == 0) {
      GPK(1, 2, 4)
    }
  } else if (D == 1) {
    if (profile || config == 6) {
      GPK(1, 1, 128)
    }
    if (profile || config == 5) {
      GPK(1, 1, 64)
    }
    if (profile || config == 4) {
      GPK(1, 1, 32)
    }
    if (profile || config == 3) {
      GPK(1, 1, 16)
    }
    if (profile || config == 2) {
      GPK(1, 1, 8)
    }
    if (profile || config == 1) {
      GPK(1, 1, 8)
    }
    if (profile || config == 0) {
      GPK(1, 1, 8)
    }
  }
#undef GPK
}

} // namespace mgard_cuda

#endif