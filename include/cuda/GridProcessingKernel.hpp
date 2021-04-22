/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_GRID_PROCESSING_KERNEL_TEMPLATE
#define MGRAD_CUDA_GRID_PROCESSING_KERNEL_TEMPLATE

#include "GridProcessingKernel.h"

namespace mgard_cuda {

template <typename T, int D_GLOBAL, int D_LOCAL, int R, int C, int F,
          bool INTERPOLATION, bool CALC_COEFF, int TYPE>
__global__ void
_gpk_reo(int *shape, int *shape_c, int *ldvs, int *ldws, int unprocessed_n,
         int *unprocessed_dims, int curr_dim_r, int curr_dim_c, int curr_dim_f,
         T *dratio_r, T *dratio_c, T *dratio_f, T *dv, int lddv1, int lddv2,
         T *dw, int lddw1, int lddw2, T *dwf, int lddwf1, int lddwf2, T *dwc,
         int lddwc1, int lddwc2, T *dwr, int lddwr1, int lddwr2, T *dwcf,
         int lddwcf1, int lddwcf2, T *dwrf, int lddwrf1, int lddwrf2, T *dwrc,
         int lddwrc1, int lddwrc2, T *dwrcf, int lddwrcf1, int lddwrcf2) {

  // bool debug = false;
  // if (blockIdx.x == 0 && blockIdx.y ==0 && blockIdx.z == 0 &&
  //     threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) debug =
  //     false;

  // volatile clock_t start = 0;
  // volatile clock_t end = 0;
  // volatile unsigned long long sum_time = 0;

  size_t threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
                    (threadIdx.y * blockDim.x) + threadIdx.x;

  int nr, nc, nf;
  int nr_c, nc_c, nf_c;
  int r, c, f;
  int rest_r, rest_c, rest_f;
  int nr_p, nc_p, nf_p;
  int rest_r_p, rest_c_p, rest_f_p;
  int r_sm, c_sm, f_sm;
  int r_sm_ex, c_sm_ex, f_sm_ex;
  int r_gl, c_gl, f_gl;
  int r_gl_ex, c_gl_ex, f_gl_ex;
  T res;
  bool in_next = true;

  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);

  T *sm = SharedMemory<T>();

  // extern __shared__ double sm[]; // size: (blockDim.x + 1) * (blockDim.y + 1)
  // * (blockDim.z + 1)
  int ldsm1 = F * 2 + 1;
  int ldsm2 = C * 2 + 1;
  T *v_sm = sm;
  T *ratio_f_sm = sm + (F * 2 + 1) * (C * 2 + 1) * (R * 2 + 1);
  T *ratio_c_sm = ratio_f_sm + F * 2;
  T *ratio_r_sm = ratio_c_sm + C * 2;
  int *shape_sm = (int *)(ratio_r_sm + R * 2);
  int *shape_c_sm = shape_sm + D_GLOBAL;
  int *unprocessed_dims_sm = shape_c_sm + D_GLOBAL;
  int *ldvs_sm = unprocessed_dims_sm + D_GLOBAL;
  int *ldws_sm = ldvs_sm + D_GLOBAL;
  int idx[D_GLOBAL];
  if (threadId < D_GLOBAL) {
    shape_sm[threadId] = shape[threadId];
    shape_c_sm[threadId] = shape_c[threadId];
    ldvs_sm[threadId] = ldvs[threadId];
    ldws_sm[threadId] = ldws[threadId];
  }

  if (threadId < unprocessed_n) {
    unprocessed_dims_sm[threadId] = unprocessed_dims[threadId];
  }
  __syncthreads();

  for (int d = 0; d < D_GLOBAL; d++)
    idx[d] = 0;

  nr = shape_sm[curr_dim_r];
  nc = shape_sm[curr_dim_c];
  nf = shape_sm[curr_dim_f];

  nr_c = shape_c_sm[curr_dim_r];
  nc_c = shape_c_sm[curr_dim_c];
  nf_c = shape_c_sm[curr_dim_f];

  if (D_LOCAL < 3) {
    nr = 1;
    nr_c = 1;
  }
  if (D_LOCAL < 2) {
    nc = 1;
    nc_c = 1;
  }

  r = blockIdx.z * blockDim.z;
  c = blockIdx.y * blockDim.y;
  int bidx = blockIdx.x;
  int firstD = div_roundup(shape_sm[0] - 1, blockDim.x);
  f = (bidx % firstD) * blockDim.x;

  bidx /= firstD;

  // if (debug) printf("n: %d %d %d rcf: %d %d %d\n", nr, nc, nf, r, c, f);
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

  for (int d = 0; d < D_GLOBAL; d++) {
    if (D_LOCAL == 3 && d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
      idx[d] = bidx % shape_sm[d];
      bidx /= shape_sm[d];
      if (idx[d] >= shape_c_sm[d])
        in_next = false;
    }
    if (D_LOCAL == 2 && d != curr_dim_c && d != curr_dim_f) {
      idx[d] = bidx % shape_sm[d];
      bidx /= shape_sm[d];
      if (idx[d] >= shape_c_sm[d])
        in_next = false;
    }
  }

  int skip = 0;
#pragma unroll 1
  for (int t = 0; t < D_GLOBAL; t++) {
    for (int k = 0; k < unprocessed_n; k++) {
      if (t == unprocessed_dims_sm[k] &&
          (shape_sm[t] % 2 == 1 && idx[t] % 2 == 1 ||
           shape_sm[t] % 2 == 0 && idx[t] % 2 == 1 &&
               idx[t] != shape_sm[t] - 1)) {
        skip = 1;
      }
    }
  }

  // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
  // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
  //   printf("total_idx_sm: %d %d %d %d (skip: %d)\n", idx[3], idx[2], idx[1],
  //   idx[0], skip);
  // }
  // }

  size_t other_offset_v = get_idx<D_GLOBAL>(ldvs_sm, idx);
  size_t other_offset_w = get_idx<D_GLOBAL>(ldws_sm, idx);

  dv = dv + other_offset_v;
  dw = dw + other_offset_w;
  dwr = dwr + other_offset_w;
  dwc = dwc + other_offset_w;
  dwf = dwf + other_offset_w;
  dwrf = dwrf + other_offset_w;
  dwrc = dwrc + other_offset_w;
  dwcf = dwcf + other_offset_w;
  dwrcf = dwrcf + other_offset_w;

  if (TYPE == 2) {
    dwf = dw;
    dwcf = dwc;
    dwrf = dwr;
    dwrcf = dwrc;
  }
  __syncthreads();
  // if (!skip)
  {
    r_sm = threadIdx.z;
    c_sm = threadIdx.y;
    f_sm = threadIdx.x;

    r_sm_ex = R * 2;
    c_sm_ex = C * 2;
    f_sm_ex = F * 2;

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
          v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1,
                       rest_f_p - 1)] =
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
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
      } else if (TYPE == 2) {
        f_gl *= 2;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              ;
            }
            if (CALC_COEFF) {
              if (in_next && f_gl < nf_c) {
                ;
              } else {
                res -= dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
              }
            }
          }
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
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
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
      } else if (TYPE == 2) {
        f_gl *= 2;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              ;
            }
            if (CALC_COEFF) {
              if (in_next && f_gl < nf_c) {
                ;
              } else {
                res -= dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
              }
            }
          }
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
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
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
      } else if (TYPE == 2) {
        f_gl *= 2;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              ;
            }
            if (CALC_COEFF) {
              if (in_next && f_gl < nf_c) {
                ;
              } else {
                res -= dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
              }
            }
          }
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
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
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
      } else if (TYPE == 2) {
        f_gl *= 2;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              ;
            }
            if (CALC_COEFF) {
              if (in_next && f_gl < nf_c) {
                ;
              } else {
                res -= dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
              }
            }
          }
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
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
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
      } else if (TYPE == 2) {
        f_gl *= 2;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              ;
            }
            if (CALC_COEFF) {
              if (in_next && f_gl < nf_c) {
                ;
              } else {
                res -= dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
              }
            }
          }
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
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
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
      } else if (TYPE == 2) {
        f_gl *= 2;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              ;
            }
            if (CALC_COEFF) {
              if (in_next && f_gl < nf_c) {
                ;
              } else {
                res -= dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
              }
            }
          }
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
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
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
      } else if (TYPE == 2) {
        f_gl *= 2;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              ;
            }
            if (CALC_COEFF) {
              if (in_next && f_gl < nf_c) {
                ;
              } else {
                res -= dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
              }
            }
          }
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
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
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
      } else if (TYPE == 2) {
        f_gl *= 2;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              ;
            }
            if (CALC_COEFF) {
              if (in_next && f_gl < nf_c) {
                ;
              } else {
                res -= dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
              }
            }
          }
          dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)] = res;
          // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
          // r_gl, c_gl, f_gl, dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)],
          // r_sm, c_sm, f_sm);
        }
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
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
          if (!skip) {
            if (INTERPOLATION) {
              res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                         ratio_f_sm[f_sm - 1]);
            }
            if (INTERPOLATION && CALC_COEFF) { // fused
              res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            }
            if (!INTERPOLATION && CALC_COEFF) { // calc_coeff only
              res -= dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
            }
          }
          dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)] = res;
        }
      } else if (TYPE == 2) {
        f_gl = 2 * f_gl + 1;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              ;
            }
            if (CALC_COEFF) {
              if (in_next && f_gl < nf_c) {
                ;
              } else {
                res -= dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
              }
            }
          }
          dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)] = res;
        }
      }

      // if (nr == 70) printf("f-store: (%d %d %d) <- %f (%d %d %d)\n", r_gl,
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
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
          if (!skip) {
            if (INTERPOLATION) {
              res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                         ratio_c_sm[c_sm - 1]);
            }
            if (INTERPOLATION && CALC_COEFF) {
              res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            }
            if (!INTERPOLATION && CALC_COEFF) {
              res -= dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
            }
          }
          dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)] = res;
        }
      } else if (TYPE == 2) {
        f_gl *= 2;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                         ratio_c_sm[c_sm - 1]);
            }
            if (CALC_COEFF) { // no need to test in_next
              res -= dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
            }
          }
          dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)] = res;
        }
        // if (nr == 70) printf("c-store: (%d %d %d) <- %f (%d %d %d)\n", r_gl,
        // c_gl, f_gl, v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)], r_sm,
        // c_sm, f_sm);
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
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
          if (!skip) {
            if (INTERPOLATION) {
              res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                         ratio_r_sm[r_sm - 1]);
            }
            if (INTERPOLATION && CALC_COEFF) {
              res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            }
            if (!INTERPOLATION && CALC_COEFF) {
              res -= dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
            }
          }
          dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)] = res;
        }
      } else if (TYPE == 2) {
        f_gl *= 2;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                         ratio_r_sm[r_sm - 1]);
            }
            if (CALC_COEFF) { // no need to test if in_next
              res -= dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
            }
          }
          dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)] = res;
        }
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
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
          if (!skip) {
            if (INTERPOLATION) {
              T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                          ratio_f_sm[f_sm - 1]);
              T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                          ratio_f_sm[f_sm - 1]);
              T tmp = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
              res = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
            }
            if (INTERPOLATION && CALC_COEFF) {
              res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            }
            if (!INTERPOLATION && CALC_COEFF) {
              res -= dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)];
            }
          }
          dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)] = res;
        }
      } else if (TYPE == 2) {
        f_gl = 2 * f_gl + 1;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                         ratio_c_sm[c_sm - 1]);
            }
            if (CALC_COEFF) { // not need to test if in_next
              res -= dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)];
            }
          }
          dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)] = res;
        }
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
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
          if (!skip) {
            if (INTERPOLATION) {
              T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                          v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                          ratio_f_sm[f_sm - 1]);
              T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                          v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                          ratio_f_sm[f_sm - 1]);
              res = lerp(f1, f2, ratio_r_sm[r_sm - 1]);
            }
            if (INTERPOLATION && CALC_COEFF) {
              res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            }
            if (!INTERPOLATION && CALC_COEFF) {
              res -= dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)];
            }
          }
          dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)] = res;
        }
      } else if (TYPE == 2) {
        f_gl = 2 * f_gl + 1;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                         ratio_r_sm[r_sm - 1]);
            }
            if (CALC_COEFF) { // no need to test if in_next
              res -= dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)];
            }
          }
          dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)] = res;
        }
      }
    }

    if (dwrc && threadId >= R * C * F * 6 && threadId < R * C * F * 7) {
      r_sm = ((threadId - R * C * F * 6) / (C * F)) * 2 + 1;
      c_sm = (((threadId - R * C * F * 6) % (C * F)) / F) * 2 + 1;
      f_sm = (((threadId - R * C * F * 6) % (C * F)) % F) * 2;
      r_gl = r / 2 + (threadId - R * C * F * 6) / (C * F);
      c_gl = c / 2 + ((threadId - R * C * F * 6) % (C * F)) / F;
      f_gl = f / 2 + ((threadId - R * C * F * 6) % (C * F)) % F;
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
          if (!skip) {
            if (INTERPOLATION) {
              T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                          v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                          ratio_c_sm[c_sm - 1]);
              T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                          v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                          ratio_c_sm[c_sm - 1]);
              res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
            }
            if (INTERPOLATION && CALC_COEFF) {
              res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            }
            if (!INTERPOLATION && CALC_COEFF) {
              res -= dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)];
            }
          }
          dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)] = res;
        }
      } else if (TYPE == 2) {
        f_gl *= 2;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                          v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                          ratio_c_sm[c_sm - 1]);
              T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                          v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                          ratio_c_sm[c_sm - 1]);
              res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
            }
            if (CALC_COEFF) { // no need to test if in_next
              res -= dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)];
            }
          }
          dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)] = res;
        }
      }
    }

    if (dwrcf && threadId >= R * C * F * 7 && threadId < R * C * F * 8) {
      r_sm = ((threadId - R * C * F * 7) / (C * F)) * 2 + 1;
      c_sm = (((threadId - R * C * F * 7) % (C * F)) / F) * 2 + 1;
      f_sm = (((threadId - R * C * F * 7) % (C * F)) % F) * 2 + 1;
      r_gl = r / 2 + (threadId - R * C * F * 7) / (C * F);
      c_gl = c / 2 + ((threadId - R * C * F * 7) % (C * F)) / F;
      f_gl = f / 2 + ((threadId - R * C * F * 7) % (C * F)) % F;
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (TYPE == 1) {
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
          if (!skip) {
            if (INTERPOLATION) {
              T f1 = lerp(
                  v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
              T f2 = lerp(
                  v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
              T f3 = lerp(
                  v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);
              T f4 = lerp(
                  v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm - 1)],
                  v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm + 1)],
                  ratio_f_sm[f_sm - 1]);

              T fc1 = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
              T fc2 = lerp(f3, f4, ratio_c_sm[c_sm - 1]);

              res = lerp(fc1, fc2, ratio_r_sm[r_sm - 1]);
            }
            if (INTERPOLATION && CALC_COEFF) {
              res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            }
            if (!INTERPOLATION && CALC_COEFF) {
              res -= dwrcf[get_idx(lddwrcf1, lddwrcf2, r_gl, c_gl, f_gl)];
            }
          }
          dwrcf[get_idx(lddwrcf1, lddwrcf2, r_gl, c_gl, f_gl)] = res;
        }
      } else if (TYPE == 2) {
        f_gl = 2 * f_gl + 1;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf) {
          if (!skip) {
            if (INTERPOLATION) {
              T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                          v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                          ratio_c_sm[c_sm - 1]);
              T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                          v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                          ratio_c_sm[c_sm - 1]);
              res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
            }
            if (CALC_COEFF) { // no need to test if in_next
              res -= dwrcf[get_idx(lddwrcf1, lddwrcf2, r_gl, c_gl, f_gl)];
            }
          }
          dwrcf[get_idx(lddwrcf1, lddwrcf2, r_gl, c_gl, f_gl)] = res;
        }
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
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
          if (TYPE == 1) {
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
              // printf("test3\n");
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                             ratio_f_sm[f_sm - 1]);
                }
                if (INTERPOLATION && CALC_COEFF) {
                  res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                }
                if (!INTERPOLATION && CALC_COEFF) {
                  res -= dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
                }
              }
              // printf("dwf (%d %d %d): %f\n", r_gl, c_gl, f_gl, res);
              dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)] = res;
            }
          } else if (TYPE == 2) {
            f_gl = 2 * f_gl + 1;
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
              if (!skip) {
                if (INTERPOLATION) {
                  ;
                }
                if (CALC_COEFF) { // need to test if in_next
                  if (in_next && f_gl < nf_c) {
                    ;
                  } // in_next
                  else {
                    res -= dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
                  }
                }
              }
              dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)] = res;
            }
          }
        }

        if (dwc) {
          r_sm = R * 2;
          c_sm = (threadId / F) * 2 + 1;
          f_sm = (threadId % F) * 2;
          r_gl = r / 2 + R;
          c_gl = c / 2 + threadId / F;
          f_gl = f / 2 + threadId % F;
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
          if (TYPE == 1) {
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                             ratio_c_sm[c_sm - 1]);
                }
                if (INTERPOLATION && CALC_COEFF) {
                  res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                }
                if (!INTERPOLATION && CALC_COEFF) {
                  res -= dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
                }
              }
              dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)] = res;
            }
          } else if (TYPE == 2) {
            f_gl *= 2;
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                             ratio_c_sm[c_sm - 1]);
                }
                if (CALC_COEFF) { // no need to test if in_next
                  res -= dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
                }
              }
              dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)] = res;
            }
          }
        }

        // printf("(%d %d %d) (%d %d %d) %f\n",
        //         r_sm, c_sm, f_sm, r_gl, c_gl, f_gl, v_sm[get_idx(ldsm1,
        //         ldsm2, r_sm, c_sm, f_sm)]);
        if (dwcf) {
          r_sm = R * 2;
          c_sm = (threadId / F) * 2 + 1;
          f_sm = (threadId % F) * 2 + 1;
          r_gl = r / 2 + R;
          c_gl = c / 2 + threadId / F;
          f_gl = f / 2 + threadId % F;
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
          if (TYPE == 1) {
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
              if (!skip) {
                if (INTERPOLATION) {
                  T f1 = lerp(
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
                  T f2 = lerp(
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
                  res = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
                }
                if (INTERPOLATION && CALC_COEFF) {
                  res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                }
                if (!INTERPOLATION && CALC_COEFF) {
                  res -= dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)];
                }
              }
              dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)] = res;
            }
          } else if (TYPE == 2) {
            f_gl = 2 * f_gl + 1;
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                             ratio_c_sm[c_sm - 1]);
                }
                if (CALC_COEFF) {
                  res -= dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)];
                }
              }
              dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)] = res;
            }
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
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
          if (TYPE == 1) {
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                             ratio_f_sm[f_sm - 1]);
                }
                if (INTERPOLATION && CALC_COEFF) {
                  res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                }
                if (!INTERPOLATION && CALC_COEFF) {
                  res -= dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
                }
              }
              dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)] = res;
            }
          } else if (TYPE == 2) {
            f_gl = 2 * f_gl + 1;
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
              if (!skip) {
                if (INTERPOLATION) {
                  ;
                }
                if (CALC_COEFF) { // need to test if in_next
                  if (in_next && f_gl < nf_c) {
                    ;
                  } // in_next
                  else {
                    res -= dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
                  }
                }
              }
              dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)] = res;
            }
          }
        }

        if (dwr) {
          r_sm = ((threadId - R * C * F) / F) * 2 + 1;
          c_sm = C * 2;
          f_sm = ((threadId - R * C * F) % F) * 2;
          r_gl = r / 2 + (threadId - R * C * F) / F;
          c_gl = c / 2 + C;
          f_gl = f / 2 + (threadId - R * C * F) % F;
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
          if (TYPE == 1) {
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                             ratio_r_sm[r_sm - 1]);
                }
                if (INTERPOLATION && CALC_COEFF) {
                  res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                }
                if (!INTERPOLATION && CALC_COEFF) {
                  res -= dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
                }
              }
              dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)] = res;
            }
          } else if (TYPE == 2) {
            f_gl *= 2;
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                             ratio_r_sm[r_sm - 1]);
                }
                if (CALC_COEFF) {
                  res -= dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
                }
              }
              dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)] = res;
            }
          }
        }

        if (dwrf) {
          r_sm = ((threadId - R * C * F) / F) * 2 + 1;
          c_sm = C * 2;
          f_sm = ((threadId - R * C * F) % F) * 2 + 1;
          r_gl = r / 2 + (threadId - R * C * F) / F;
          c_gl = c / 2 + C;
          f_gl = f / 2 + (threadId - R * C * F) % F;
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
          if (TYPE == 1) {
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
              if (!skip) {
                if (INTERPOLATION) {
                  T f1 = lerp(
                      v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
                  T f2 = lerp(
                      v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                      ratio_f_sm[f_sm - 1]);
                  res = lerp(f1, f2, ratio_r_sm[r_sm - 1]);
                }
                if (INTERPOLATION && CALC_COEFF) {
                  res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                }
                if (!INTERPOLATION && CALC_COEFF) {
                  res -= dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)];
                }
              }
              dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)] = res;
            }
          } else if (TYPE == 2) {
            f_gl = 2 * f_gl + 1;
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                             ratio_r_sm[r_sm - 1]);
                }
                if (CALC_COEFF) { // no need to test if in_next
                  res -= dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)];
                }
              }
              dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)] = res;
            }
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
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
          if (TYPE == 1) {
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                             ratio_c_sm[c_sm - 1]);
                }
                if (INTERPOLATION && CALC_COEFF) {
                  res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                }
                if (!INTERPOLATION && CALC_COEFF) {
                  res -= dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
                }
              }
              dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)] = res;
            }
          } else if (TYPE == 2) {
            f_gl *= 2;
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                             ratio_c_sm[c_sm - 1]);
                }
                if (CALC_COEFF) {
                  res -= dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
                }
              }
              dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)] = res;
            }
          }
        }

        if (dwr) {
          r_sm = ((threadId - R * C * F * 2) / C) * 2 + 1;
          c_sm = ((threadId - R * C * F * 2) % C) * 2;
          f_sm = F * 2;
          r_gl = r / 2 + (threadId - R * C * F * 2) / C;
          c_gl = c / 2 + (threadId - R * C * F * 2) % C;
          f_gl = f / 2 + F;
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
          if (TYPE == 1) {
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                             ratio_r_sm[r_sm - 1]);
                }
                if (INTERPOLATION && CALC_COEFF) {
                  res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                }
                if (!INTERPOLATION && CALC_COEFF) {
                  res -= dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
                }
              }
              dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)] = res;
            }
          } else if (TYPE == 2) {
            f_gl *= 2;
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
              if (!skip) {
                if (INTERPOLATION) {
                  res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                             v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                             ratio_r_sm[r_sm - 1]);
                }
                if (CALC_COEFF) {
                  res -= dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
                }
              }
              dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)] = res;
            }
          }
        }

        if (dwrc) {
          r_sm = ((threadId - R * C * F * 2) / C) * 2 + 1;
          c_sm = ((threadId - R * C * F * 2) % C) * 2 + 1;
          f_sm = F * 2;
          r_gl = r / 2 + (threadId - R * C * F * 2) / C;
          c_gl = c / 2 + (threadId - R * C * F * 2) % C;
          f_gl = f / 2 + F;
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
          if (TYPE == 1) {
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
              if (!skip) {
                if (INTERPOLATION) {
                  T c1 = lerp(
                      v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                      ratio_c_sm[c_sm - 1]);
                  T c2 = lerp(
                      v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                      ratio_c_sm[c_sm - 1]);
                  res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
                }
                if (INTERPOLATION && CALC_COEFF) {
                  res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
                }
                if (!INTERPOLATION && CALC_COEFF) {
                  res -= dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)];
                }
              }
              dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)] = res;
            }
          } else if (TYPE == 2) {
            f_gl *= 2;
            if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
                r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf) {
              if (!skip) {
                if (INTERPOLATION) {
                  T c1 = lerp(
                      v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                      ratio_c_sm[c_sm - 1]);
                  T c2 = lerp(
                      v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                      v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                      ratio_c_sm[c_sm - 1]);
                  res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
                }
                if (CALC_COEFF) {
                  res -= dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)];
                }
              }
              dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)] = res;
            }
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
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                           ratio_r_sm[r_sm - 1]);
              }
              if (INTERPOLATION && CALC_COEFF) {
                res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
              }
              if (!INTERPOLATION && CALC_COEFF) {
                res -= dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
              }
            }
            dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)] = res;
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                           ratio_r_sm[r_sm - 1]);
              }
              if (CALC_COEFF) {
                res -= dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
              }
            }
            dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)] = res;
          }
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
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                           ratio_c_sm[c_sm - 1]);
              }
              if (INTERPOLATION && CALC_COEFF) {
                res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
              }
              if (!INTERPOLATION && CALC_COEFF) {
                res -= dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
              }
            }
            dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)] = res;
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                           ratio_c_sm[c_sm - 1]);
              }
              if (CALC_COEFF) {
                res -= dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
              }
            }
            dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)] = res;
          }
        }
      }
    }

    // printf("test1\n");
    if (dwf && r + R * 2 == nr_p - 1 && c + C * 2 == nc_p - 1) {
      // printf("test2\n");
      if (threadId >= R * C * F * 5 && threadId < R * C * F * 5 + F) {
        // printf("test3\n");
        r_sm = R * 2;
        c_sm = C * 2;
        f_sm = (threadId - R * C * F * 5) * 2 + 1;
        r_gl = r / 2 + R;
        c_gl = c / 2 + C;
        f_gl = f / 2 + threadId - R * C * F * 5;
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
            // printf("test4\n");
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                           ratio_f_sm[f_sm - 1]);
              }
              if (INTERPOLATION && CALC_COEFF) {
                res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
              }
              if (!INTERPOLATION && CALC_COEFF) {
                res -= dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
              }
            }
            dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)] = res;
            // printf("dwf(%d %d %d): %f\n", r_gl, c_gl, f_gl,
            // dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)]);
          }
        } else if (TYPE == 2) {
          f_gl = 2 * f_gl + 1;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
            if (!skip) {
              if (INTERPOLATION) {
                ;
              }
              if (CALC_COEFF) { // do need to test in_next
                if (in_next && f_gl < nf_c) {
                  ;
                } // in_next
                else {
                  res -= dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
                }
              }
            }
            dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)] = res;
          }
        }
      }
    }

  } // skip

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

template <typename T, int D_GLOBAL, int D_LOCAL, int R, int C, int F,
          bool INTERPOLATION, bool CALC_COEFF, int TYPE>
void gpk_reo_adaptive_launcher(
    Handle<T, D_GLOBAL> &handle, thrust::device_vector<int> shape,
    thrust::device_vector<int> shape_c, thrust::device_vector<int> ldvs,
    thrust::device_vector<int> ldws,
    thrust::device_vector<int> unprocessed_dims, int curr_dim_r, int curr_dim_c,
    int curr_dim_f, T *dratio_r, T *dratio_c, T *dratio_f, T *dv, int lddv1,
    int lddv2, T *dw, int lddw1, int lddw2, T *dwf, int lddwf1, int lddwf2,
    T *dwc, int lddwc1, int lddwc2, T *dwr, int lddwr1, int lddwr2, T *dwcf,
    int lddwcf1, int lddwcf2, T *dwrf, int lddwrf1, int lddwrf2, T *dwrc,
    int lddwrc1, int lddwrc2, T *dwrcf, int lddwrcf1, int lddwrcf2,
    int queue_idx) {

  int nr = shape[curr_dim_r];
  int nc = shape[curr_dim_c];
  int nf = shape[curr_dim_f];
  if (D_LOCAL == 2) {
    nr = 1;
  }
  int total_thread_z = std::max(nr - 1, 1);
  int total_thread_y = std::max(nc - 1, 1);
  int total_thread_x = std::max(nf - 1, 1);

  int tbx, tby, tbz, gridx, gridy, gridz;
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
  sm_size += (D_GLOBAL * 5) * sizeof(int);

  gridz = ceil((float)total_thread_z / tbz);
  gridy = ceil((float)total_thread_y / tby);
  gridx = ceil((float)total_thread_x / tbx);
  for (int d = 0; d < D_GLOBAL; d++) {
    if (D_LOCAL == 3 && d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
      gridx *= shape[d];
    }
    if (D_LOCAL == 2 && d != curr_dim_f && d != curr_dim_c) {
      gridx *= shape[d];
    }
  }
  threadsPerBlock = dim3(tbx, tby, tbz);
  blockPerGrid = dim3(gridx, gridy, gridz);

  // printf("_gpk_reo exec config (%d %d %d) (%d %d %d)\n", tbx, tby, tbz,
  // gridx, gridy, gridz);
  _gpk_reo<T, D_GLOBAL, D_LOCAL, R / 2, C / 2, F / 2, INTERPOLATION, CALC_COEFF,
           TYPE><<<blockPerGrid, threadsPerBlock, sm_size,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
      thrust::raw_pointer_cast(shape.data()),
      thrust::raw_pointer_cast(shape_c.data()),
      thrust::raw_pointer_cast(ldvs.data()),
      thrust::raw_pointer_cast(ldws.data()), unprocessed_dims.size(),
      thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r, curr_dim_c,
      curr_dim_f, dratio_r, dratio_c, dratio_f, dv, lddv1, lddv2, dw, lddw1,
      lddw2, dwf, lddwf1, lddwf2, dwc, lddwc1, lddwc2, dwr, lddwr1, lddwr2,
      dwcf, lddwcf1, lddwcf2, dwrf, lddwrf1, lddwrf2, dwrc, lddwrc1, lddwrc2,
      dwrcf, lddwrcf1, lddwrcf2);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template <typename T, int D_GLOBAL, int D_LOCAL, bool INTERPOLATION,
          bool CALC_COEFF, int TYPE>
void gpk_reo(Handle<T, D_GLOBAL> &handle, thrust::device_vector<int> shape,
             thrust::device_vector<int> shape_c,
             thrust::device_vector<int> ldvs, thrust::device_vector<int> ldws,
             thrust::device_vector<int> unprocessed_dims, int curr_dim_r,
             int curr_dim_c, int curr_dim_f, T *dratio_r, T *dratio_c,
             T *dratio_f, T *dv, int lddv1, int lddv2, T *dw, int lddw1,
             int lddw2, T *dwf, int lddwf1, int lddwf2, T *dwc, int lddwc1,
             int lddwc2, T *dwr, int lddwr1, int lddwr2, T *dwcf, int lddwcf1,
             int lddwcf2, T *dwrf, int lddwrf1, int lddwrf2, T *dwrc,
             int lddwrc1, int lddwrc2, T *dwrcf, int lddwrcf1, int lddwrcf2,
             int queue_idx, int config) {

#define GPK(R, C, F)                                                           \
  {                                                                            \
    gpk_reo_adaptive_launcher<T, D_GLOBAL, D_LOCAL, R, C, F, INTERPOLATION,    \
                              CALC_COEFF, TYPE>(                               \
        handle, shape, shape_c, ldvs, ldws, unprocessed_dims, curr_dim_r,      \
        curr_dim_c, curr_dim_f, dratio_r, dratio_c, dratio_f, dv, lddv1,       \
        lddv2, dw, lddw1, lddw2, dwf, lddwf1, lddwf2, dwc, lddwc1, lddwc2,     \
        dwr, lddwr1, lddwr2, dwcf, lddwcf1, lddwcf2, dwrf, lddwrf1, lddwrf2,   \
        dwrc, lddwrc1, lddwrc2, dwrcf, lddwrcf1, lddwrcf2, queue_idx);         \
  }
  bool profile = false;
#ifdef MGARD_CUDA_KERNEL_PROFILE
  profile = true;
#endif
  if (D_LOCAL == 3) {
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
    // GPK(T, 4, 4, 4)
  } else if (D_LOCAL == 2) {
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
    // GPK(T, 1, 4, 4)
  } else if (D_LOCAL == 1) {
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

template <typename T, int D_GLOBAL, int D_LOCAL, int R, int C, int F,
          bool INTERPOLATION, bool CALC_COEFF, int TYPE>
void gpk_reo_adaptive_launcher(
    Handle<T, D_GLOBAL> &handle, int *shape_h, int *shape_d, int *shape_c_d,
    int *ldvs, int *ldws, int unprocessed_n, int *unprocessed_dims,
    int curr_dim_r, int curr_dim_c, int curr_dim_f, T *dratio_r, T *dratio_c,
    T *dratio_f, T *dv, int lddv1, int lddv2, T *dw, int lddw1, int lddw2,
    T *dwf, int lddwf1, int lddwf2, T *dwc, int lddwc1, int lddwc2, T *dwr,
    int lddwr1, int lddwr2, T *dwcf, int lddwcf1, int lddwcf2, T *dwrf,
    int lddwrf1, int lddwrf2, T *dwrc, int lddwrc1, int lddwrc2, T *dwrcf,
    int lddwrcf1, int lddwrcf2, int queue_idx) {

  int nr = shape_h[curr_dim_r];
  int nc = shape_h[curr_dim_c];
  int nf = shape_h[curr_dim_f];
  if (D_LOCAL == 2) {
    nr = 1;
  }
  int total_thread_z = std::max(nr - 1, 1);
  int total_thread_y = std::max(nc - 1, 1);
  int total_thread_x = std::max(nf - 1, 1);

  int tbx, tby, tbz, gridx, gridy, gridz;
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
  sm_size += (D_GLOBAL * 5) * sizeof(int);

  gridz = ceil((float)total_thread_z / tbz);
  gridy = ceil((float)total_thread_y / tby);
  gridx = ceil((float)total_thread_x / tbx);
  for (int d = 0; d < D_GLOBAL; d++) {
    if (D_LOCAL == 3 && d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
      gridx *= shape_h[d];
    }
    if (D_LOCAL == 2 && d != curr_dim_f && d != curr_dim_c) {
      gridx *= shape_h[d];
    }
  }
  threadsPerBlock = dim3(tbx, tby, tbz);
  blockPerGrid = dim3(gridx, gridy, gridz);

  // printf("_gpk_reo exec config (%d %d %d) (%d %d %d)\n", tbx, tby, tbz,
  // gridx, gridy, gridz);

  // high_resolution_clock::time_point t1 = high_resolution_clock::now();
  _gpk_reo<T, D_GLOBAL, D_LOCAL, R / 2, C / 2, F / 2, INTERPOLATION, CALC_COEFF,
           TYPE><<<blockPerGrid, threadsPerBlock, sm_size,
                   *(cudaStream_t *)handle.get(queue_idx)>>>(
      shape_d, shape_c_d, ldvs, ldws, unprocessed_n, unprocessed_dims,
      curr_dim_r, curr_dim_c, curr_dim_f, dratio_r, dratio_c, dratio_f, dv,
      lddv1, lddv2, dw, lddw1, lddw2, dwf, lddwf1, lddwf2, dwc, lddwc1, lddwc2,
      dwr, lddwr1, lddwr2, dwcf, lddwcf1, lddwcf2, dwrf, lddwrf1, lddwrf2, dwrc,
      lddwrc1, lddwrc2, dwrcf, lddwrcf1, lddwrcf2);

  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
  // high_resolution_clock::time_point t2 = high_resolution_clock::now();
  // duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("gpk_reo_kernel time: %.6f s\n", time_span.count());
}

template <typename T, int D_GLOBAL, int D_LOCAL, bool INTERPOLATION,
          bool CALC_COEFF, int TYPE>
void gpk_reo(Handle<T, D_GLOBAL> &handle, int *shape_h, int *shape_d,
             int *shape_c_d, int *ldvs, int *ldws, int unprocessed_n,
             int *unprocessed_dims, int curr_dim_r, int curr_dim_c,
             int curr_dim_f, T *dratio_r, T *dratio_c, T *dratio_f, T *dv,
             int lddv1, int lddv2, T *dw, int lddw1, int lddw2, T *dwf,
             int lddwf1, int lddwf2, T *dwc, int lddwc1, int lddwc2, T *dwr,
             int lddwr1, int lddwr2, T *dwcf, int lddwcf1, int lddwcf2, T *dwrf,
             int lddwrf1, int lddwrf2, T *dwrc, int lddwrc1, int lddwrc2,
             T *dwrcf, int lddwrcf1, int lddwrcf2, int queue_idx, int config) {

#define GPK(R, C, F)                                                           \
  {                                                                            \
    gpk_reo_adaptive_launcher<T, D_GLOBAL, D_LOCAL, R, C, F, INTERPOLATION,    \
                              CALC_COEFF, TYPE>(                               \
        handle, shape_h, shape_d, shape_c_d, ldvs, ldws, unprocessed_n,        \
        unprocessed_dims, curr_dim_r, curr_dim_c, curr_dim_f, dratio_r,        \
        dratio_c, dratio_f, dv, lddv1, lddv2, dw, lddw1, lddw2, dwf, lddwf1,   \
        lddwf2, dwc, lddwc1, lddwc2, dwr, lddwr1, lddwr2, dwcf, lddwcf1,       \
        lddwcf2, dwrf, lddwrf1, lddwrf2, dwrc, lddwrc1, lddwrc2, dwrcf,        \
        lddwrcf1, lddwrcf2, queue_idx);                                        \
  }
  bool profile = false;
#ifdef MGARD_CUDA_KERNEL_PROFILE
  profile = true;
#endif
  if (D_LOCAL == 3) {
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
    // GPK(T, 4, 4, 4)
  } else if (D_LOCAL == 2) {
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
      GPK(1, 2, 2)
    }
    // GPK(T, 1, 4, 4)
  } else if (D_LOCAL == 1) {
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
      GPK(1, 1, 4)
    }
    if (profile || config == 0) {
      GPK(1, 1, 2)
    }
  }
#undef GPK
}

template <typename T, int D_GLOBAL, int D_LOCAL, int R, int C, int F,
          bool INTERPOLATION, bool COEFF_RESTORE, int TYPE>
__global__ void
_gpk_rev(int *shape, int *shape_c, int *ldvs, int *ldws, int unprocessed_n,
         int *unprocessed_dims, int curr_dim_r, int curr_dim_c, int curr_dim_f,
         T *dratio_r, T *dratio_c, T *dratio_f, T *dv, int lddv1, int lddv2,
         T *dw, int lddw1, int lddw2, T *dwf, int lddwf1, int lddwf2, T *dwc,
         int lddwc1, int lddwc2, T *dwr, int lddwr1, int lddwr2, T *dwcf,
         int lddwcf1, int lddwcf2, T *dwrf, int lddwrf1, int lddwrf2, T *dwrc,
         int lddwrc1, int lddwrc2, T *dwrcf, int lddwrcf1, int lddwrcf2,
         int svr, int svc, int svf, int nvr, int nvc, int nvf) {

  bool debug = false;
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    debug = false;

  bool debug2 = false;
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
    debug2 = false;

  size_t threadId = (threadIdx.z * (blockDim.x * blockDim.y)) +
                    (threadIdx.y * blockDim.x) + threadIdx.x;

  int nr, nc, nf;
  int nr_c, nc_c, nf_c;
  int r, c, f;
  int rest_r, rest_c, rest_f;
  int nr_p, nc_p, nf_p;
  int rest_r_p, rest_c_p, rest_f_p;
  int r_sm, c_sm, f_sm;
  int r_sm_ex, c_sm_ex, f_sm_ex;
  int r_gl, c_gl, f_gl;
  int r_gl_ex, c_gl_ex, f_gl_ex;
  T res;
  bool in_next = true;

  // extern __shared__ __align__(sizeof(T)) unsigned char smem[];
  // T * sm = reinterpret_cast<T *>(smem);

  T *sm = SharedMemory<T>();

  // extern __shared__ double sm[]; // size: (blockDim.x + 1) * (blockDim.y + 1)
  // * (blockDim.z + 1)
  int ldsm1 = F * 2 + 1;
  int ldsm2 = C * 2 + 1;
  T *v_sm = sm;
  T *ratio_f_sm = sm + (F * 2 + 1) * (C * 2 + 1) * (R * 2 + 1);
  T *ratio_c_sm = ratio_f_sm + F * 2;
  T *ratio_r_sm = ratio_c_sm + C * 2;
  int *shape_sm = (int *)(ratio_r_sm + R * 2);
  int *shape_c_sm = shape_sm + D_GLOBAL;
  int *unprocessed_dims_sm = shape_c_sm + D_GLOBAL;
  int *ldvs_sm = unprocessed_dims_sm + D_GLOBAL;
  int *ldws_sm = ldvs_sm + D_GLOBAL;
  int idx[D_GLOBAL];
  if (threadId < D_GLOBAL) {
    shape_sm[threadId] = shape[threadId];
    shape_c_sm[threadId] = shape_c[threadId];
    ldvs_sm[threadId] = ldvs[threadId];
    ldws_sm[threadId] = ldws[threadId];
  }

  if (threadId < unprocessed_n) {
    unprocessed_dims_sm[threadId] = unprocessed_dims[threadId];
  }
  __syncthreads();
  for (int d = 0; d < D_GLOBAL; d++)
    idx[d] = 0;

  nr = shape_sm[curr_dim_r];
  nc = shape_sm[curr_dim_c];
  nf = shape_sm[curr_dim_f];

  nr_c = shape_c_sm[curr_dim_r];
  nc_c = shape_c_sm[curr_dim_c];
  nf_c = shape_c_sm[curr_dim_f];

  if (D_LOCAL < 3) {
    nr = 1;
    nr_c = 1;
  }
  if (D_LOCAL < 2) {
    nc = 1;
    nc_c = 1;
  }

  r = blockIdx.z * blockDim.z;
  c = blockIdx.y * blockDim.y;
  int bidx = blockIdx.x;
  int firstD = div_roundup(shape_sm[0] - 1, blockDim.x);
  f = (bidx % firstD) * blockDim.x;

  bidx /= firstD;

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

  for (int d = 0; d < D_GLOBAL; d++) {
    if (D_LOCAL == 3 && d != curr_dim_r && d != curr_dim_c && d != curr_dim_f) {
      idx[d] = bidx % shape_sm[d];
      bidx /= shape_sm[d];
      if ((shape_sm[d] % 2 == 1 && idx[d] % 2 != 0) ||
          shape_sm[d] % 2 == 0 &&
              (idx[d] % 2 != 0 && idx[d] != shape_sm[d] - 1))
        in_next = false;
    }
    if (D_LOCAL == 2 && d != curr_dim_c && d != curr_dim_f) {
      idx[d] = bidx % shape_sm[d];
      bidx /= shape_sm[d];
      if ((shape_sm[d] % 2 == 1 && idx[d] % 2 != 0) ||
          shape_sm[d] % 2 == 0 &&
              (idx[d] % 2 != 0 && idx[d] != shape_sm[d] - 1))
        in_next = false;
    }
  }

  int skip = 0;
#pragma unroll 1
  for (int t = 0; t < D_GLOBAL; t++) {
    for (int k = 0; k < unprocessed_n; k++) {
      if (t == unprocessed_dims_sm[k] && idx[t] >= shape_c_sm[t]) {
        skip = 1;
      }
    }
  }

  // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
  // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
  //   printf("TYPE %d total_idx_sm: %d %d %d %d (skip: %d)\n", TYPE, idx[3],
  //   idx[2], idx[1], idx[0], skip);
  // }
  // }

  size_t other_offset_v = get_idx<D_GLOBAL>(ldvs_sm, idx);
  size_t other_offset_w = get_idx<D_GLOBAL>(ldws_sm, idx);

  dv = dv + other_offset_v;
  dw = dw + other_offset_w;
  dwr = dwr + other_offset_w;
  dwc = dwc + other_offset_w;
  dwf = dwf + other_offset_w;
  dwrf = dwrf + other_offset_w;
  dwrc = dwrc + other_offset_w;
  dwcf = dwcf + other_offset_w;
  dwrcf = dwrcf + other_offset_w;

  if (TYPE == 2) {
    dwf = dw;
    dwcf = dwc;
    dwrf = dwr;
    dwrcf = dwrc;
  }
  __syncthreads();

  r_sm = threadIdx.z;
  c_sm = threadIdx.y;
  f_sm = threadIdx.x;

  r_sm_ex = R * 2;
  c_sm_ex = C * 2;
  f_sm_ex = F * 2;

  r_gl = r + r_sm;
  r_gl_ex = r + R * 2;
  c_gl = c + c_sm;
  c_gl_ex = c + C * 2;
  f_gl = f + f_sm;
  f_gl_ex = f + F * 2;

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

  if (threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
    for (int i = 0; i < R * 2 + 1; i++) {
      for (int j = 0; j < C * 2 + 1; j++) {
        for (int k = 0; k < F * 2 + 1; k++) {
          v_sm[get_idx(ldsm1, ldsm2, i, j, k)] = 0.0;
        }
      }
    }
  }

  __syncthreads();

  if (dw && threadId < R * C * F) {
    r_sm = (threadId / (C * F)) * 2;
    c_sm = ((threadId % (C * F)) / F) * 2;
    f_sm = ((threadId % (C * F)) % F) * 2;
    r_gl = r / 2 + threadId / (C * F);
    c_gl = c / 2 + threadId % (C * F) / F;
    f_gl = f / 2 + threadId % (C * F) % F;
    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
        } else {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
              dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        }
      }
    } else if (TYPE == 2) {
      f_gl *= 2;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (debug2)
          printf("(%d %d %d) %f <- (%d %d %d)\n", r_sm, c_sm, f_sm,
                 dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }

      f_gl += 1;
      f_sm += 1;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (debug2)
          printf("(%d %d %d) %f <- (%d %d %d)\n", r_sm, c_sm, f_sm,
                 dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }
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
    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
        } else {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
              dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        }
      }
    } else if (TYPE == 2) {
      f_gl *= 2;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            // printf("nf: %d, f_gl: %d, in_next: %d, f_in_next: %d\n", nf,
            // f_gl, in_next, f_in_next);
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }

      f_gl += 1;
      f_sm += 1;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (debug2)
          printf("(%d %d %d) %f <- (%d %d %d)\n", r_sm, c_sm, f_sm,
                 dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            // printf("nf: %d, f_gl: %d, in_next: %d, f_in_next: %d\n", nf,
            // f_gl, in_next, f_in_next);
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }
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
    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
        } else {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
              dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        }
      }
    } else if (TYPE == 2) {
      f_gl *= 2;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }

      f_gl += 1;
      f_sm += 1;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (debug2)
          printf("(%d %d %d) %f <- (%d %d %d)\n", r_sm, c_sm, f_sm,
                 dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }
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
    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
        } else {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
              dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        }
      }
    } else if (TYPE == 2) {
      f_gl *= 2;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }

      f_gl += 1;
      f_sm += 1;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (debug2)
          printf("(%d %d %d) %f <- (%d %d %d)\n", r_sm, c_sm, f_sm,
                 dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }
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
    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
        } else {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
              dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        }
      }
    } else if (TYPE == 2) {
      f_gl *= 2;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }

      f_gl += 1;
      f_sm += 1;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (debug2)
          printf("(%d %d %d) %f <- (%d %d %d)\n", r_sm, c_sm, f_sm,
                 dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }
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
    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
        } else {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
              dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        }
      }
    } else if (TYPE == 2) {
      f_gl *= 2;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }

      f_gl += 1;
      f_sm += 1;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (debug2)
          printf("(%d %d %d) %f <- (%d %d %d)\n", r_sm, c_sm, f_sm,
                 dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }
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
    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
        } else {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
              dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        }
      }
    } else if (TYPE == 2) {
      f_gl *= 2;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }

      f_gl += 1;
      f_sm += 1;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (debug2)
          printf("(%d %d %d) %f <- (%d %d %d)\n", r_sm, c_sm, f_sm,
                 dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }
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
    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
        } else {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
              dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        }
      }
    } else if (TYPE == 2) {
      f_gl *= 2;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }

      f_gl += 1;
      f_sm += 1;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] =
            dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)];
        if (debug2)
          printf("(%d %d %d) %f <- (%d %d %d)\n", r_sm, c_sm, f_sm,
                 dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
          if (COEFF_RESTORE) {
            bool f_in_next = (nf % 2 == 1 && f_gl % 2 == 0) ||
                             (nf % 2 == 0 && (f_gl % 2 == 0 || f_gl == nf - 1));
            if (in_next && f_in_next) {
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = 0.0;
            } else {
              ;
            }
          }
        }
      }
    }
  }

  __syncthreads();

  __syncthreads();
  if (debug) {
    printf("TYPE: %d %d %d %d\n", TYPE, min(rest_r_p, R * 2 + 1),
           min(rest_c_p, C * 2 + 1), min(rest_f_p, F * 2 + 1));
    for (int i = 0; i < min(rest_r_p, R * 2 + 1); i++) {
      for (int j = 0; j < min(rest_c_p, C * 2 + 1); j++) {
        for (int k = 0; k < min(rest_f_p, F * 2 + 1); k++) {
          printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
        }
        printf("\n");
      }
      printf("\n");
    }
  }
  __syncthreads();

  if (dwf && threadId >= R * C * F && threadId < R * C * F * 2) {

    r_sm = ((threadId - R * C * F) / (C * F)) * 2;
    c_sm = (((threadId - R * C * F) % (C * F)) / F) * 2;
    f_sm = (((threadId - R * C * F) % (C * F)) % F) * 2 + 1;
    r_gl = r / 2 + (threadId - R * C * F) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F) % (C * F)) % F;

    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {

        res = dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION && COEFF_RESTORE) { // fused
            res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
          } else if (INTERPOLATION && !COEFF_RESTORE) {
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                       ratio_f_sm[f_sm - 1]);
          }
        }
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
    } else if (TYPE == 2) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
        f_gl = 2 * f_gl + 1;
        // res = dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            ;
          }
        }
        // v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
    }
  }

  if (dwc && threadId >= R * C * F * 2 && threadId < R * C * F * 3) {
    r_sm = ((threadId - R * C * F * 2) / (C * F)) * 2;
    c_sm = (((threadId - R * C * F * 2) % (C * F)) / F) * 2 + 1;
    f_sm = (((threadId - R * C * F * 2) % (C * F)) % F) * 2;
    r_gl = r / 2 + (threadId - R * C * F * 2) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 2) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 2) % (C * F)) % F;
    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
        res = dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION && COEFF_RESTORE) {
            res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
          } else if (INTERPOLATION && !COEFF_RESTORE) {
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                       ratio_c_sm[c_sm - 1]);
          }
        }
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
    } else if (TYPE == 2) {
      f_gl *= 2;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
        res = dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                       ratio_c_sm[c_sm - 1]);
          }
        }
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
    }
  }

  if (dwr && threadId >= R * C * F * 3 && threadId < R * C * F * 4) {
    r_sm = ((threadId - R * C * F * 3) / (C * F)) * 2 + 1;
    c_sm = (((threadId - R * C * F * 3) % (C * F)) / F) * 2;
    f_sm = (((threadId - R * C * F * 3) % (C * F)) % F) * 2;
    r_gl = r / 2 + (threadId - R * C * F * 3) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 3) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 3) % (C * F)) % F;

    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
        res = dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION && COEFF_RESTORE) {
            res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                        ratio_r_sm[r_sm - 1]);
          } else if (INTERPOLATION && !COEFF_RESTORE) {
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                       ratio_r_sm[r_sm - 1]);
          }
        }
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
    } else if (TYPE == 2) {
      f_gl *= 2;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
        res = dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                       ratio_r_sm[r_sm - 1]);
          }
        }
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
    }
  }

  if (dwcf && threadId >= R * C * F * 4 && threadId < R * C * F * 5) {
    r_sm = ((threadId - R * C * F * 4) / (C * F)) * 2;
    c_sm = (((threadId - R * C * F * 4) % (C * F)) / F) * 2 + 1;
    f_sm = (((threadId - R * C * F * 4) % (C * F)) % F) * 2 + 1;
    r_gl = r / 2 + (threadId - R * C * F * 4) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 4) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 4) % (C * F)) % F;

    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
        res = dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION && COEFF_RESTORE) {
            T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            res += lerp(f1, f2, ratio_c_sm[c_sm - 1]);
          } else if (INTERPOLATION && !COEFF_RESTORE) {
            T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            res = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
          }
        }
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
    } else if (TYPE == 2) {
      f_gl = 2 * f_gl + 1;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
        res = dwcf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                       ratio_c_sm[c_sm - 1]);
          }
        }
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
    }
  }

  if (dwrf && threadId >= R * C * F * 5 && threadId < R * C * F * 6) {
    r_sm = ((threadId - R * C * F * 5) / (C * F)) * 2 + 1;
    c_sm = (((threadId - R * C * F * 5) % (C * F)) / F) * 2;
    f_sm = (((threadId - R * C * F * 5) % (C * F)) % F) * 2 + 1;
    r_gl = r / 2 + (threadId - R * C * F * 5) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 5) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 5) % (C * F)) % F;

    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf - nf_c) {

        res = dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION && COEFF_RESTORE) {
            T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);

            res += lerp(f1, f2, ratio_r_sm[r_sm - 1]);
          } else if (INTERPOLATION && !COEFF_RESTORE) {
            T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);

            res = lerp(f1, f2, ratio_r_sm[r_sm - 1]);
          }
        }
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
    } else if (TYPE == 2) {
      f_gl = 2 * f_gl + 1;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
        res = dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                       ratio_r_sm[r_sm - 1]);
          }
        }
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
    }
  }

  if (dwrc && threadId >= R * C * F * 6 && threadId < R * C * F * 7) {
    r_sm = ((threadId - R * C * F * 6) / (C * F)) * 2 + 1;
    c_sm = (((threadId - R * C * F * 6) % (C * F)) / F) * 2 + 1;
    f_sm = (((threadId - R * C * F * 6) % (C * F)) % F) * 2;
    r_gl = r / 2 + (threadId - R * C * F * 6) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 6) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 6) % (C * F)) % F;

    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
        res = dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION && COEFF_RESTORE) {
            T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            res += lerp(c1, c2, ratio_r_sm[r_sm - 1]);
          } else if (INTERPOLATION && !COEFF_RESTORE) {
            T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
          }
        }
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
    } else if (TYPE == 2) {
      f_gl *= 2;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf) {
        res = dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
          }
        }
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
    }
  }

  if (dwrcf && threadId >= R * C * F * 7 && threadId < R * C * F * 8) {
    r_sm = ((threadId - R * C * F * 7) / (C * F)) * 2 + 1;
    c_sm = (((threadId - R * C * F * 7) % (C * F)) / F) * 2 + 1;
    f_sm = (((threadId - R * C * F * 7) % (C * F)) % F) * 2 + 1;
    r_gl = r / 2 + (threadId - R * C * F * 7) / (C * F);
    c_gl = c / 2 + ((threadId - R * C * F * 7) % (C * F)) / F;
    f_gl = f / 2 + ((threadId - R * C * F * 7) % (C * F)) % F;

    if (TYPE == 1) {
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
        res = dwrcf[get_idx(lddwrcf1, lddwrcf2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION && COEFF_RESTORE) {
            T f1 =
                lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm - 1)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm + 1)],
                     ratio_f_sm[f_sm - 1]);
            T f2 =
                lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm - 1)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm + 1)],
                     ratio_f_sm[f_sm - 1]);
            T f3 =
                lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm - 1)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm + 1)],
                     ratio_f_sm[f_sm - 1]);
            T f4 =
                lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm - 1)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm + 1)],
                     ratio_f_sm[f_sm - 1]);

            T fc1 = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
            T fc2 = lerp(f3, f4, ratio_c_sm[c_sm - 1]);

            res += lerp(fc1, fc2, ratio_r_sm[r_sm - 1]);
          } else if (INTERPOLATION && !COEFF_RESTORE) {
            T f1 =
                lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm - 1)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm + 1)],
                     ratio_f_sm[f_sm - 1]);
            T f2 =
                lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm - 1)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm + 1)],
                     ratio_f_sm[f_sm - 1]);
            T f3 =
                lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm - 1)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm + 1)],
                     ratio_f_sm[f_sm - 1]);
            T f4 =
                lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm - 1)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm + 1)],
                     ratio_f_sm[f_sm - 1]);

            T fc1 = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
            T fc2 = lerp(f3, f4, ratio_c_sm[c_sm - 1]);

            res = lerp(fc1, fc2, ratio_r_sm[r_sm - 1]);
          }
        }
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
    } else if (TYPE == 2) {
      f_gl = 2 * f_gl + 1;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf) {
        res = dwrcf[get_idx(lddwrcf1, lddwrcf2, r_gl, c_gl, f_gl)];
        if (!skip) {
          if (INTERPOLATION) {
            T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
          }
        }
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
      }
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
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
            res = dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION && COEFF_RESTORE) {
                res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                            ratio_f_sm[f_sm - 1]);
              } else if (INTERPOLATION && !COEFF_RESTORE) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                           ratio_f_sm[f_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        } else if (TYPE == 2) {
          f_gl = 2 * f_gl + 1;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
            // res = dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION) {
                ;
              }
            }
            // v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        }
      }

      if (dwc) {
        r_sm = R * 2;
        c_sm = (threadId / F) * 2 + 1;
        f_sm = (threadId % F) * 2;
        r_gl = r / 2 + R;
        c_gl = c / 2 + threadId / F;
        f_gl = f / 2 + threadId % F;

        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
            res = dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION && COEFF_RESTORE) {
                res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                            ratio_c_sm[c_sm - 1]);
              } else if (INTERPOLATION && !COEFF_RESTORE) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                           ratio_c_sm[c_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
            res = dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                           ratio_c_sm[c_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        }
      }
      if (dwcf) {
        r_sm = R * 2;
        c_sm = (threadId / F) * 2 + 1;
        f_sm = (threadId % F) * 2 + 1;
        r_gl = r / 2 + R;
        c_gl = c / 2 + threadId / F;
        f_gl = f / 2 + threadId % F;
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf - nf_c) {
            res = dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION && COEFF_RESTORE) {
                T f1 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                         ratio_f_sm[f_sm - 1]);
                T f2 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                         ratio_f_sm[f_sm - 1]);
                res += lerp(f1, f2, ratio_c_sm[c_sm - 1]);
              } else if (INTERPOLATION && !COEFF_RESTORE) {
                T f1 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                         ratio_f_sm[f_sm - 1]);
                T f2 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                         ratio_f_sm[f_sm - 1]);
                res = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
              }
            }

            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        } else if (TYPE == 2) {
          f_gl = 2 * f_gl + 1;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
            res = dwcf[get_idx(lddwcf1, lddwcf2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                           ratio_c_sm[c_sm - 1]);
                // if (idx[1] ==0 && idx[2] == 0) {
                //   printf("%f(%d %d %d) %f(%d %d %d) -> %f\n",
                //           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                //           r_sm, c_sm - 1, f_sm, v_sm[get_idx(ldsm1, ldsm2,
                //           r_sm, c_sm + 1, f_sm)], r_sm, c_sm + 1, f_sm, res);
                // }
              }
            }

            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
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

        if (TYPE) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
            res = dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION && COEFF_RESTORE) {
                res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                            ratio_f_sm[f_sm - 1]);
              } else if (INTERPOLATION && !COEFF_RESTORE) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                           ratio_f_sm[f_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        } else if (TYPE == 2) {
          f_gl = 2 * f_gl + 1;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
            // res = dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION) {
                ;
              }
            }
            // v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        }
      }
      if (dwr) {
        r_sm = ((threadId - R * C * F) / F) * 2 + 1;
        c_sm = C * 2;
        f_sm = ((threadId - R * C * F) % F) * 2;
        r_gl = r / 2 + (threadId - R * C * F) / F;
        c_gl = c / 2 + C;
        f_gl = f / 2 + (threadId - R * C * F) % F;
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
            res = dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION && COEFF_RESTORE) {
                res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                            ratio_r_sm[r_sm - 1]);
              } else if (INTERPOLATION && !COEFF_RESTORE) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                           ratio_r_sm[r_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
            res = dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                           ratio_r_sm[r_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        }
      }
      if (dwrf) {
        r_sm = ((threadId - R * C * F) / F) * 2 + 1;
        c_sm = C * 2;
        f_sm = ((threadId - R * C * F) % F) * 2 + 1;
        r_gl = r / 2 + (threadId - R * C * F) / F;
        c_gl = c / 2 + C;
        f_gl = f / 2 + (threadId - R * C * F) % F;

        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
            res = dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION && COEFF_RESTORE) {
                T f1 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                         ratio_f_sm[f_sm - 1]);
                T f2 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                         ratio_f_sm[f_sm - 1]);
                res += lerp(f1, f2, ratio_r_sm[r_sm - 1]);
              } else if (INTERPOLATION && !COEFF_RESTORE) {
                T f1 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                         ratio_f_sm[f_sm - 1]);
                T f2 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                         ratio_f_sm[f_sm - 1]);
                res = lerp(f1, f2, ratio_r_sm[r_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        } else if (TYPE == 2) {
          f_gl = 2 * f_gl + 1;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
            res = dwrf[get_idx(lddwrf1, lddwrf2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                           ratio_r_sm[r_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
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
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
            res = dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION && COEFF_RESTORE) {
                res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                            ratio_c_sm[c_sm - 1]);
              } else if (INTERPOLATION && !COEFF_RESTORE) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                           ratio_c_sm[c_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
            res = dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                           ratio_c_sm[c_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        }
      }

      if (dwr) {
        r_sm = ((threadId - R * C * F * 2) / C) * 2 + 1;
        c_sm = ((threadId - R * C * F * 2) % C) * 2;
        f_sm = F * 2;
        r_gl = r / 2 + (threadId - R * C * F * 2) / C;
        c_gl = c / 2 + (threadId - R * C * F * 2) % C;
        f_gl = f / 2 + F;
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
            res = dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION && COEFF_RESTORE) {
                res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                            ratio_r_sm[r_sm - 1]);
              } else if (INTERPOLATION && !COEFF_RESTORE) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                           ratio_r_sm[r_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
            res = dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                           ratio_r_sm[r_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        }
      }

      if (dwrc) {
        r_sm = ((threadId - R * C * F * 2) / C) * 2 + 1;
        c_sm = ((threadId - R * C * F * 2) % C) * 2 + 1;
        f_sm = F * 2;
        r_gl = r / 2 + (threadId - R * C * F * 2) / C;
        c_gl = c / 2 + (threadId - R * C * F * 2) % C;
        f_gl = f / 2 + F;

        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
            res = dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION && COEFF_RESTORE) {
                T c1 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                         ratio_c_sm[c_sm - 1]);
                T c2 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                         ratio_c_sm[c_sm - 1]);
                res += lerp(c1, c2, ratio_r_sm[r_sm - 1]);
              } else if (INTERPOLATION && !COEFF_RESTORE) {
                T c1 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                         ratio_c_sm[c_sm - 1]);
                T c2 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                         ratio_c_sm[c_sm - 1]);
                res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc - nc_c && f_gl < nf) {
            res = dwrc[get_idx(lddwrc1, lddwrc2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION) {
                T c1 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                         ratio_c_sm[c_sm - 1]);
                T c2 =
                    lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                         v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                         ratio_c_sm[c_sm - 1]);
                res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
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
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
            res = dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION && COEFF_RESTORE) {
                res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                            ratio_r_sm[r_sm - 1]);
              } else if (INTERPOLATION && !COEFF_RESTORE) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                           ratio_r_sm[r_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf) {
            res = dwr[get_idx(lddwr1, lddwr2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                           ratio_r_sm[r_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
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
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
            res = dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION && COEFF_RESTORE) {
                res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                            ratio_c_sm[c_sm - 1]);
              } else if (INTERPOLATION && !COEFF_RESTORE) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                           ratio_c_sm[c_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        } else if (TYPE == 2) {
          f_gl *= 2;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf) {
            res = dwc[get_idx(lddwc1, lddwc2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                           ratio_c_sm[c_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
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
        if (TYPE == 1) {
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
            res = dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION && COEFF_RESTORE) {
                res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                            ratio_f_sm[f_sm - 1]);
              } else if (INTERPOLATION && !COEFF_RESTORE) {
                res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                           v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                           ratio_f_sm[f_sm - 1]);
              }
            }
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        } else if (TYPE == 2) {
          f_gl = 2 * f_gl + 1;
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf) {
            // res = dwf[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)];
            if (!skip) {
              if (INTERPOLATION) {
                ;
              }
            }
            // v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
          }
        }
      }
    }
  }

  __syncthreads();
  if (debug) {
    printf("TYPE: %d %d %d %d\n", TYPE, min(rest_r_p, R * 2 + 1),
           min(rest_c_p, C * 2 + 1), min(rest_f_p, F * 2 + 1));
    for (int i = 0; i < min(rest_r_p, R * 2 + 1); i++) {
      for (int j = 0; j < min(rest_c_p, C * 2 + 1); j++) {
        for (int k = 0; k < min(rest_f_p, F * 2 + 1); k++) {
          printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
        }
        printf("\n");
      }
      printf("\n");
    }
  }
  __syncthreads();

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
  if (TYPE == 1 && nf % 2 == 0)
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

    if (D_LOCAL >= 3 && r_sm == 0) {
      if (nr % 2 != 0 && R * 2 + 1 == rest_r) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)];
        }
      }
      if (nr % 2 == 0 && R * 2 + 1 >= rest_r_p) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, f_sm)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm)];
      }
    }

    if (D_LOCAL >= 2 && c_sm == 0) {
      if (nc % 2 != 0 && C * 2 + 1 == rest_c) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)];
        }
      }
      if (nc % 2 == 0 && C * 2 + 1 >= rest_c_p) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, f_sm)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm)];
      }
    }

    if (D_LOCAL >= 1 && f_sm == 0) {
      if (nf % 2 != 0 && F * 2 + 1 == rest_f) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)];
        }
      }
      if (nf % 2 == 0 && F * 2 + 1 >= rest_f_p && TYPE == 1) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f_p - 1)];
      }
    }

    // load extra edges
    if (D_LOCAL >= 2 && c_sm == 0 && f_sm == 0) {
      if (nc % 2 != 0 && C * 2 + 1 == rest_c && nf % 2 != 0 &&
          F * 2 + 1 == rest_f) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)];
        }
      }
      if (nc % 2 == 0 && nf % 2 == 0 && C * 2 + 1 >= rest_c_p &&
          F * 2 + 1 >= rest_f_p && TYPE == 1) {
        v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, rest_f - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, rest_f_p - 1)];
      }
      if (nc % 2 == 0 && nf % 2 != 0 && C * 2 + 1 >= rest_c_p &&
          F * 2 + 1 == rest_f) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm_ex)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm_ex)];
        }
      }
      if (nc % 2 != 0 && nf % 2 == 0 && C * 2 + 1 == rest_c &&
          F * 2 + 1 >= rest_f_p && TYPE == 1) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, rest_f_p - 1)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, rest_f_p - 1)];
          // printf("(%d %d %d): %f <- (%d %d %d)\n",
          //         r_gl, c_gl_ex, f_gl_ex,
          //         dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)],
          //         r_sm, c_sm_ex, f_gl_ex);
        }
      }
    }

    if (D_LOCAL >= 3 && r_sm == 0 && f_sm == 0) {
      if (nr % 2 != 0 && R * 2 + 1 == rest_r && nf % 2 != 0 &&
          F * 2 + 1 == rest_f) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)];
        }
      }
      if (nr % 2 == 0 && nf % 2 == 0 && R * 2 + 1 >= rest_r_p &&
          F * 2 + 1 >= rest_f_p && TYPE == 1) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, rest_f - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, rest_f_p - 1)];
      }
      if (nr % 2 == 0 && nf % 2 != 0 && R * 2 + 1 >= rest_r_p &&
          F * 2 + 1 == rest_f) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm_ex)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm_ex)];
        }
      }
      if (nr % 2 != 0 && nf % 2 == 0 && R * 2 + 1 == rest_r &&
          F * 2 + 1 >= rest_f_p && TYPE == 1) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, rest_f_p - 1)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, rest_f_p - 1)];
          // printf("(%d %d %d): %f <- (%d %d %d)\n",
          //         r_gl_ex, c_gl, rest_f-1,
          //         dv[get_idx(lddv1, lddv2, r_gl_ex-1, c_gl, f_gl_ex)],
          //         r_sm_ex, c_sm, rest_f_p-1);
        }
      }
    }

    if (D_LOCAL >= 3 && r_sm == 0 && c_sm == 0) {
      if (nr % 2 != 0 && R * 2 + 1 == rest_r && nc % 2 != 0 &&
          C * 2 + 1 == rest_c) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)];
        }
      }
      if (nr % 2 == 0 && nc % 2 == 0 && R * 2 + 1 >= rest_r_p &&
          C * 2 + 1 >= rest_c_p) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, f_sm)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1, f_sm)];
      }
      if (nr % 2 == 0 && nc % 2 != 0 && R * 2 + 1 >= rest_r_p &&
          C * 2 + 1 == rest_c) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)] +=
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm)];
        }
      }
      if (nr % 2 != 0 && nc % 2 == 0 && R * 2 + 1 == rest_r &&
          C * 2 + 1 >= rest_c_p) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm)];
        }
      }
    }
    // load extra vertex

    if (D_LOCAL >= 3 && r_sm == 0 && c_sm == 0 && f_sm == 0) {
      if (nr % 2 != 0 && R * 2 + 1 == rest_r && nc % 2 != 0 &&
          C * 2 + 1 == rest_c && nf % 2 != 0 && F * 2 + 1 == rest_f) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm_ex)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm_ex)];
        }
      }

      if (nr % 2 == 0 && nc % 2 == 0 && nf % 2 == 0 && R * 2 + 1 >= rest_r_p &&
          C * 2 + 1 >= rest_c_p && F * 2 + 1 >= rest_f_p && TYPE == 1) {
        v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, rest_f - 1)] =
            v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1,
                         rest_f_p - 1)];
      }
      if (nr % 2 == 0 && nc % 2 == 0 && nf % 2 != 0 && R * 2 + 1 >= rest_r_p &&
          C * 2 + 1 >= rest_c_p && F * 2 + 1 == rest_f) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1, f_sm_ex)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1, f_sm_ex)];
        }
      }
      if (nr % 2 == 0 && nc % 2 != 0 && nf % 2 == 0 && R * 2 + 1 >= rest_r_p &&
          C * 2 + 1 == rest_c && F * 2 + 1 >= rest_f_p && TYPE == 1) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, rest_f_p - 1)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, rest_f_p - 1)];
        }
      }
      if (nr % 2 != 0 && nc % 2 == 0 && nf % 2 == 0 && R * 2 + 1 == rest_r &&
          C * 2 + 1 >= rest_c_p && F * 2 + 1 >= rest_f_p && TYPE == 1) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, rest_f_p - 1)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, rest_f_p - 1)];
        }
      }
      if (nr % 2 == 0 && nc % 2 != 0 && nf % 2 != 0 && R * 2 + 1 >= rest_r_p &&
          C * 2 + 1 == rest_c && F * 2 + 1 == rest_f) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm_ex)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm_ex)];
        }
      }
      if (nr % 2 != 0 && nc % 2 == 0 && nf % 2 != 0 && R * 2 + 1 == rest_r &&
          C * 2 + 1 >= rest_c_p && F * 2 + 1 == rest_f) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm_ex)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm_ex)];
        }
      }
      if (nr % 2 != 0 && nc % 2 != 0 && nf % 2 == 0 && R * 2 + 1 == rest_r &&
          C * 2 + 1 == rest_c && F * 2 + 1 >= rest_f_p && TYPE == 1) {
        if (!INTERPOLATION && COEFF_RESTORE) {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] +=
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, rest_f_p - 1)];
        } else {
          dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, rest_f_p - 1)];
        }
      }
    }
  }

  __syncthreads();

  if (r_sm < rest_r && c_sm < rest_c && f_sm < rest_f) {
    if (r_gl >= svr && r_gl < svr + nvr && c_gl >= svc && c_gl < svc + nvc &&
        f_gl >= svf && f_gl < svf + nvf) {
      if (!INTERPOLATION && COEFF_RESTORE) {
        dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl)] +=
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      } else {
        dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl)] =
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      }
    }
  }
}

template <typename T, int D_GLOBAL, int D_LOCAL, int R, int C, int F,
          bool INTERPOLATION, bool COEFF_RESTORE, int TYPE>
void gpk_rev_adaptive_launcher(
    Handle<T, D_GLOBAL> &handle, thrust::device_vector<int> shape,
    thrust::device_vector<int> shape_c, thrust::device_vector<int> ldvs,
    thrust::device_vector<int> ldws,
    thrust::device_vector<int> unprocessed_dims, int curr_dim_r, int curr_dim_c,
    int curr_dim_f, T *dratio_r, T *dratio_c, T *dratio_f, T *dv, int lddv1,
    int lddv2, T *dw, int lddw1, int lddw2, T *dwf, int lddwf1, int lddwf2,
    T *dwc, int lddwc1, int lddwc2, T *dwr, int lddwr1, int lddwr2, T *dwcf,
    int lddwcf1, int lddwcf2, T *dwrf, int lddwrf1, int lddwrf2, T *dwrc,
    int lddwrc1, int lddwrc2, T *dwrcf, int lddwrcf1, int lddwrcf2, int svr,
    int svc, int svf, int nvr, int nvc, int nvf, int queue_idx) {

  int nr = shape[curr_dim_r];
  int nc = shape[curr_dim_c];
  int nf = shape[curr_dim_f];
  if (D_LOCAL == 2) {
    nr = 1;
  }
  int total_thread_z = std::max(nr - 1, 1);
  int total_thread_y = std::max(nc - 1, 1);
  int total_thread_x = std::max(nf - 1, 1);

  int tbx, tby, tbz, gridx, gridy, gridz;
  dim3 threadsPerBlock, blockPerGrid;
  size_t sm_size;

  // tbz = std::min(R, total_thread_z);
  // tby = std::min(C, total_thread_y);
  // tbx = std::min(F, total_thread_x);
  tbz = R;
  tby = C;
  tbx = F;
  sm_size = ((R + 1) * (C + 1) * (F + 1) + R + C + F) * sizeof(T);
  sm_size += (D_GLOBAL * 5 + 1) * sizeof(int);

  gridz = ceil((float)total_thread_z / tbz);
  gridy = ceil((float)total_thread_y / tby);
  gridx = ceil((float)total_thread_x / tbx);
  for (int d = 0; d < D_GLOBAL; d++) {
    if (D_LOCAL == 3 && d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
      gridx *= shape[d];
    }
    if (D_LOCAL == 2 && d != curr_dim_f && d != curr_dim_c) {
      gridx *= shape[d];
    }
  }

  threadsPerBlock = dim3(tbx, tby, tbz);
  blockPerGrid = dim3(gridx, gridy, gridz);

  // printf("gpk_rev exec: %d %d %d %d %d %d\n", tbx, tby, tbz, gridx, gridy,
  // gridz);
  _gpk_rev<T, D_GLOBAL, D_LOCAL, R / 2, C / 2, F / 2, INTERPOLATION,
           COEFF_RESTORE, TYPE><<<blockPerGrid, threadsPerBlock, sm_size,
                                  *(cudaStream_t *)handle.get(queue_idx)>>>(
      thrust::raw_pointer_cast(shape.data()),
      thrust::raw_pointer_cast(shape_c.data()),
      thrust::raw_pointer_cast(ldvs.data()),
      thrust::raw_pointer_cast(ldws.data()), unprocessed_dims.size(),
      thrust::raw_pointer_cast(unprocessed_dims.data()), curr_dim_r, curr_dim_c,
      curr_dim_f, dratio_r, dratio_c, dratio_f, dv, lddv1, lddv2, dw, lddw1,
      lddw2, dwf, lddwf1, lddwf2, dwc, lddwc1, lddwc2, dwr, lddwr1, lddwr2,
      dwcf, lddwcf1, lddwcf2, dwrf, lddwrf1, lddwrf2, dwrc, lddwrc1, lddwrc2,
      dwrcf, lddwrcf1, lddwrcf2, svr, svc, svf, nvr, nvc, nvf);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template <typename T, int D_GLOBAL, int D_LOCAL, bool INTERPOLATION,
          bool COEFF_RESTORE, int TYPE>
void gpk_rev(Handle<T, D_GLOBAL> &handle, thrust::device_vector<int> shape,
             thrust::device_vector<int> shape_c,
             thrust::device_vector<int> ldvs, thrust::device_vector<int> ldws,
             thrust::device_vector<int> unprocessed_dims, int curr_dim_r,
             int curr_dim_c, int curr_dim_f, T *dratio_r, T *dratio_c,
             T *dratio_f, T *dv, int lddv1, int lddv2, T *dw, int lddw1,
             int lddw2, T *dwf, int lddwf1, int lddwf2, T *dwc, int lddwc1,
             int lddwc2, T *dwr, int lddwr1, int lddwr2, T *dwcf, int lddwcf1,
             int lddwcf2, T *dwrf, int lddwrf1, int lddwrf2, T *dwrc,
             int lddwrc1, int lddwrc2, T *dwrcf, int lddwrcf1, int lddwrcf2,
             int svr, int svc, int svf, int nvr, int nvc, int nvf,
             int queue_idx, int config) {

#define GPK(R, C, F)                                                           \
  {                                                                            \
    gpk_rev_adaptive_launcher<T, D_GLOBAL, D_LOCAL, R, C, F, INTERPOLATION,    \
                              COEFF_RESTORE, TYPE>(                            \
        handle, shape, shape_c, ldvs, ldws, unprocessed_dims, curr_dim_r,      \
        curr_dim_c, curr_dim_f, dratio_r, dratio_c, dratio_f, dv, lddv1,       \
        lddv2, dw, lddw1, lddw2, dwf, lddwf1, lddwf2, dwc, lddwc1, lddwc2,     \
        dwr, lddwr1, lddwr2, dwcf, lddwcf1, lddwcf2, dwrf, lddwrf1, lddwrf2,   \
        dwrc, lddwrc1, lddwrc2, dwrcf, lddwrcf1, lddwrcf2, svr, svc, svf, nvr, \
        nvc, nvf, queue_idx);                                                  \
  }
  bool profile = false;
#ifdef MGARD_CUDA_KERNEL_PROFILE
  profile = true;
#endif
  if (D_LOCAL == 3) {
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
      GPK(4, 4, 4)
    }
  } else if (D_LOCAL == 2) {
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
      GPK(1, 2, 2)
    }
  } else if (D_LOCAL == 1) {
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
      GPK(1, 1, 4)
    }
    if (profile || config == 0) {
      GPK(1, 1, 2)
    }
  }
#undef GPK
}

template <typename T, int D_GLOBAL, int D_LOCAL, int R, int C, int F,
          bool INTERPOLATION, bool COEFF_RESTORE, int TYPE>
void gpk_rev_adaptive_launcher(
    Handle<T, D_GLOBAL> &handle, int *shape_h, int *shape_d, int *shape_c_d,
    int *ldvs, int *ldws, int unprocessed_n, int *unprocessed_dims,
    int curr_dim_r, int curr_dim_c, int curr_dim_f, T *dratio_r, T *dratio_c,
    T *dratio_f, T *dv, int lddv1, int lddv2, T *dw, int lddw1, int lddw2,
    T *dwf, int lddwf1, int lddwf2, T *dwc, int lddwc1, int lddwc2, T *dwr,
    int lddwr1, int lddwr2, T *dwcf, int lddwcf1, int lddwcf2, T *dwrf,
    int lddwrf1, int lddwrf2, T *dwrc, int lddwrc1, int lddwrc2, T *dwrcf,
    int lddwrcf1, int lddwrcf2, int svr, int svc, int svf, int nvr, int nvc,
    int nvf, int queue_idx) {

  int nr = shape_h[curr_dim_r];
  int nc = shape_h[curr_dim_c];
  int nf = shape_h[curr_dim_f];
  if (D_LOCAL == 2) {
    nr = 1;
  }
  int total_thread_z = std::max(nr - 1, 1);
  int total_thread_y = std::max(nc - 1, 1);
  int total_thread_x = std::max(nf - 1, 1);

  int tbx, tby, tbz, gridx, gridy, gridz;
  dim3 threadsPerBlock, blockPerGrid;
  size_t sm_size;

  // tbz = std::min(R, total_thread_z);
  // tby = std::min(C, total_thread_y);
  // tbx = std::min(F, total_thread_x);
  tbz = R;
  tby = C;
  tbx = F;
  sm_size = ((R + 1) * (C + 1) * (F + 1) + R + C + F) * sizeof(T);
  sm_size += (D_GLOBAL * 5 + 1) * sizeof(int);

  gridz = ceil((float)total_thread_z / tbz);
  gridy = ceil((float)total_thread_y / tby);
  gridx = ceil((float)total_thread_x / tbx);
  for (int d = 0; d < D_GLOBAL; d++) {
    if (D_LOCAL == 3 && d != curr_dim_f && d != curr_dim_c && d != curr_dim_r) {
      gridx *= shape_h[d];
    }
    if (D_LOCAL == 2 && d != curr_dim_f && d != curr_dim_c) {
      gridx *= shape_h[d];
    }
  }

  threadsPerBlock = dim3(tbx, tby, tbz);
  blockPerGrid = dim3(gridx, gridy, gridz);

  // printf("gpk_rev exec: %d %d %d %d %d %d\n", tbx, tby, tbz, gridx, gridy,
  // gridz);
  _gpk_rev<T, D_GLOBAL, D_LOCAL, R / 2, C / 2, F / 2, INTERPOLATION,
           COEFF_RESTORE, TYPE><<<blockPerGrid, threadsPerBlock, sm_size,
                                  *(cudaStream_t *)handle.get(queue_idx)>>>(
      shape_d, shape_c_d, ldvs, ldws, unprocessed_n, unprocessed_dims,
      curr_dim_r, curr_dim_c, curr_dim_f, dratio_r, dratio_c, dratio_f, dv,
      lddv1, lddv2, dw, lddw1, lddw2, dwf, lddwf1, lddwf2, dwc, lddwc1, lddwc2,
      dwr, lddwr1, lddwr2, dwcf, lddwcf1, lddwcf2, dwrf, lddwrf1, lddwrf2, dwrc,
      lddwrc1, lddwrc2, dwrcf, lddwrcf1, lddwrcf2, svr, svc, svf, nvr, nvc,
      nvf);
  gpuErrchk(cudaGetLastError());
#ifdef MGARD_CUDA_DEBUG
  gpuErrchk(cudaDeviceSynchronize());
#endif
}

template <typename T, int D_GLOBAL, int D_LOCAL, bool INTERPOLATION,
          bool COEFF_RESTORE, int TYPE>
void gpk_rev(Handle<T, D_GLOBAL> &handle, int *shape_h, int *shape_d,
             int *shape_c_d, int *ldvs, int *ldws, int unprocessed_n,
             int *unprocessed_dims, int curr_dim_r, int curr_dim_c,
             int curr_dim_f, T *dratio_r, T *dratio_c, T *dratio_f, T *dv,
             int lddv1, int lddv2, T *dw, int lddw1, int lddw2, T *dwf,
             int lddwf1, int lddwf2, T *dwc, int lddwc1, int lddwc2, T *dwr,
             int lddwr1, int lddwr2, T *dwcf, int lddwcf1, int lddwcf2, T *dwrf,
             int lddwrf1, int lddwrf2, T *dwrc, int lddwrc1, int lddwrc2,
             T *dwrcf, int lddwrcf1, int lddwrcf2, int svr, int svc, int svf,
             int nvr, int nvc, int nvf, int queue_idx, int config) {

#define GPK(R, C, F)                                                           \
  {                                                                            \
    gpk_rev_adaptive_launcher<T, D_GLOBAL, D_LOCAL, R, C, F, INTERPOLATION,    \
                              COEFF_RESTORE, TYPE>(                            \
        handle, shape_h, shape_d, shape_c_d, ldvs, ldws, unprocessed_n,        \
        unprocessed_dims, curr_dim_r, curr_dim_c, curr_dim_f, dratio_r,        \
        dratio_c, dratio_f, dv, lddv1, lddv2, dw, lddw1, lddw2, dwf, lddwf1,   \
        lddwf2, dwc, lddwc1, lddwc2, dwr, lddwr1, lddwr2, dwcf, lddwcf1,       \
        lddwcf2, dwrf, lddwrf1, lddwrf2, dwrc, lddwrc1, lddwrc2, dwrcf,        \
        lddwrcf1, lddwrcf2, svr, svc, svf, nvr, nvc, nvf, queue_idx);          \
  }
  bool profile = false;
#ifdef MGARD_CUDA_KERNEL_PROFILE
  profile = true;
#endif
  if (D_LOCAL == 3) {
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
      GPK(4, 4, 4)
    }
  } else if (D_LOCAL == 2) {
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
  } else if (D_LOCAL == 1) {
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