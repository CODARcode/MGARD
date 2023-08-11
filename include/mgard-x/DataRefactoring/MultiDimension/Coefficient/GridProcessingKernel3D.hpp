/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_GRID_PROCESSING_KERNEL_3D_TEMPLATE
#define MGARD_X_GRID_PROCESSING_KERNEL_3D_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"
#include "GPKFunctor.h"

namespace mgard_x {

namespace data_refactoring {

namespace multi_dimension {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class GpkReo3DFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT GpkReo3DFunctor() {}
  MGARDX_CONT GpkReo3DFunctor(
      SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
      SubArray<1, T, DeviceType> ratio_r, SubArray<1, T, DeviceType> ratio_c,
      SubArray<1, T, DeviceType> ratio_f, SubArray<D, T, DeviceType> v,
      SubArray<D, T, DeviceType> w, SubArray<D, T, DeviceType> wf,
      SubArray<D, T, DeviceType> wc, SubArray<D, T, DeviceType> wr,
      SubArray<D, T, DeviceType> wcf, SubArray<D, T, DeviceType> wrf,
      SubArray<D, T, DeviceType> wrc, SubArray<D, T, DeviceType> wrcf)
      : nr(nr), nc(nc), nf(nf), nr_c(nr_c), nc_c(nc_c), nf_c(nf_c),
        ratio_r(ratio_r), ratio_c(ratio_c), ratio_f(ratio_f), v(v), w(w),
        wf(wf), wc(wc), wr(wr), wcf(wcf), wrf(wrf), wrc(wrc), wrcf(wrcf) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {

    sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = (F / 2) * 2 + 1;
    ldsm2 = (C / 2) * 2 + 1;
    v_sm = sm;
    ratio_f_sm = sm + ((F / 2) * 2 + 1) * ((C / 2) * 2 + 1) * ((R / 2) * 2 + 1);
    ratio_c_sm = ratio_f_sm + (F / 2) * 2;
    ratio_r_sm = ratio_c_sm + (C / 2) * 2;

    r = FunctorBase<DeviceType>::GetBlockIdZ() *
        FunctorBase<DeviceType>::GetBlockDimZ();
    c = FunctorBase<DeviceType>::GetBlockIdY() *
        FunctorBase<DeviceType>::GetBlockDimY();
    f = FunctorBase<DeviceType>::GetBlockIdX() *
        FunctorBase<DeviceType>::GetBlockDimX();

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

    r_sm = FunctorBase<DeviceType>::GetThreadIdZ();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();

    r_sm_ex = (R / 2) * 2;
    c_sm_ex = (C / 2) * 2;
    f_sm_ex = (F / 2) * 2;

    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                (FunctorBase<DeviceType>::GetBlockDimX() *
                 FunctorBase<DeviceType>::GetBlockDimY())) +
               (FunctorBase<DeviceType>::GetThreadIdY() *
                FunctorBase<DeviceType>::GetBlockDimX()) +
               FunctorBase<DeviceType>::GetThreadIdX();

    r_gl = r + r_sm;
    r_gl_ex = r + (R / 2) * 2;
    c_gl = c + c_sm;
    c_gl_ex = c + (C / 2) * 2;
    f_gl = f + f_sm;
    f_gl_ex = f + (F / 2) * 2;

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
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = *v(r_gl, c_gl, f_gl);
      // if (blockIdx.x==0 && blockIdx.y==0&&blockIdx.z==0) {
      //   printf("load (%d %d %d) %f <- %d+(%d %d %d) (ld: %d %d)\n",
      //           r_sm, c_sm, f_sm,
      //           dv[get_idx(lddv1, lddv2, r_gl, c_gl, f_gl)],
      //           other_offset_v+r_gl, c_gl, f_gl, lddv1, lddv2);
      // }
      if (r_sm == 0) {
        if (rest_r > (R / 2) * 2) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)] =
              *v(r_gl_ex, c_gl, f_gl);
        }
      }
      if (c_sm == 0) {
        if (rest_c > (C / 2) * 2) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] =
              *v(r_gl, c_gl_ex, f_gl);
        }
      }
      if (f_sm == 0) {
        if (rest_f > (F / 2) * 2) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)] =
              *v(r_gl, c_gl, f_gl_ex);
        }
      }
      if (c_sm == 0 && f_sm == 0) {
        if (rest_c > (C / 2) * 2 && rest_f > (F / 2) * 2) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)] =
              *v(r_gl, c_gl_ex, f_gl_ex);
        }
      }
      if (r_sm == 0 && f_sm == 0) {
        if (rest_r > (R / 2) * 2 && rest_f > (F / 2) * 2) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)] =
              *v(r_gl_ex, c_gl, f_gl_ex);
        }
      }
      if (r_sm == 0 && c_sm == 0) {
        if (rest_r > (R / 2) * 2 && rest_c > (C / 2) * 2) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)] =
              *v(r_gl_ex, c_gl_ex, f_gl);
        }
      }
      if (r_sm == 0 && c_sm == 0 && f_sm == 0) {
        if (rest_r > (R / 2) * 2 && rest_c > (C / 2) * 2 &&
            rest_f > (F / 2) * 2) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm_ex)] =
              *v(r_gl_ex, c_gl_ex, f_gl_ex);
        }
      }
    }
  }

  MGARDX_EXEC void Operation2() {
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
        if (rest_r > (R / 2) * 2) {
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
        if (rest_c > (C / 2) * 2) {
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
        if (rest_f > (F / 2) * 2) {
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
        if (rest_c > (C / 2) * 2 && rest_f > (F / 2) * 2) {
          // v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)] =
          //     dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)];
          // printf("load-cf[%d %d %d]:%f --> [%d %d %d]\n", r_gl, c_gl_ex,
          // f_gl_ex, dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)], r_sm,
          // c_sm_ex, f_sm_ex);
        } else if (rest_c <= (C / 2) * 2 && rest_f <= (F / 2) * 2 &&
                   nc % 2 == 0 && nf % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, rest_f_p - 1)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, rest_f - 1)];
        } else if (rest_c > (C / 2) * 2 && rest_f <= (F / 2) * 2 &&
                   nf % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, rest_f_p - 1)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, rest_f - 1)];
        } else if (rest_c <= (C / 2) * 2 && rest_f > (F / 2) * 2 &&
                   nc % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, f_sm_ex)];
        }
      }

      if (r_sm == 0 && f_sm == 0) {
        if (rest_r > (R / 2) * 2 && rest_f > (F / 2) * 2) {
          // v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)] =
          //     dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)];
          // printf("load-rf[%d %d %d]:%f --> [%d %d %d]\n", r_gl_ex, c_gl,
          // f_gl_ex, dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl, f_gl_ex)],
          // r_sm_ex, c_sm, f_sm_ex);
        } else if (rest_r <= (R / 2) * 2 && rest_f <= (F / 2) * 2 &&
                   nr % 2 == 0 && nf % 2 == 0) {
          // printf("padding (%d %d %d) <- (%d %d %d)\n", rest_r_p - 1, c_sm,
          // rest_f_p - 1, rest_r - 1, c_sm, rest_f - 1);
          v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, rest_f_p - 1)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, rest_f - 1)];
        } else if (rest_r > (R / 2) * 2 && rest_f <= (F / 2) * 2 &&
                   nf % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, rest_f_p - 1)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, rest_f - 1)];
        } else if (rest_r <= (R / 2) * 2 && rest_f > (F / 2) * 2 &&
                   nr % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, f_sm_ex)];
        }
      }

      if (r_sm == 0 && c_sm == 0) {
        if (rest_r > (R / 2) * 2 && rest_c > (C / 2) * 2) {
          // v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)] =
          //     dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)];
          // printf("load-rc[%d %d %d]:%f --> [%d %d %d]\n", r_gl_ex, c_gl_ex,
          // f_gl, dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl)], r_sm_ex,
          // c_sm_ex, f_sm);
        } else if (rest_r <= (R / 2) * 2 && rest_c <= (C / 2) * 2 &&
                   nr % 2 == 0 && nc % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1, f_sm)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, f_sm)];
          // printf("padding (%d %d %d) <- (%d %d %d): %f\n", rest_r_p - 1,
          // rest_c_p - 1, f_sm, rest_r - 1, rest_c - 1, f_sm,
          // v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, f_sm)]);
        } else if (rest_r > (R / 2) * 2 && rest_c <= (C / 2) * 2 &&
                   nc % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c - 1, f_sm)];
        } else if (rest_r <= (R / 2) * 2 && rest_c > (C / 2) * 2 &&
                   nr % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm_ex, f_sm)];
        }
      }
      // load extra vertex

      if (r_sm == 0 && c_sm == 0 && f_sm == 0) {
        if (rest_r > (R / 2) * 2 && rest_c > (C / 2) * 2 &&
            rest_f > (F / 2) * 2) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm_ex)] =
              *v(r_gl_ex, c_gl_ex, f_gl_ex);
          // printf("load-rcf[%d %d %d]:%f --> [%d %d %d]\n", r_gl_ex, c_gl_ex,
          // f_gl_ex, dv[get_idx(lddv1, lddv2, r_gl_ex, c_gl_ex, f_gl_ex)],
          // r_sm_ex, c_sm_ex, f_sm_ex);
        } else if (rest_r <= (R / 2) * 2 && rest_c <= (C / 2) * 2 &&
                   rest_f <= (F / 2) * 2 && nr % 2 == 0 && nc % 2 == 0 &&
                   nf % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1,
                       rest_f_p - 1)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, rest_f - 1)];
        } else if (rest_r > (R / 2) * 2 && rest_c > (C / 2) * 2 &&
                   rest_f <= (F / 2) * 2 && nf % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, rest_f_p - 1)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, rest_f - 1)];
        } else if (rest_r > (R / 2) * 2 && rest_c <= (C / 2) * 2 &&
                   rest_f > (F / 2) * 2 && nc % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c - 1, f_sm_ex)];
        } else if (rest_r > (R / 2) * 2 && rest_c <= (C / 2) * 2 &&
                   rest_f <= (F / 2) * 2 && nc % 2 == 0 && nf % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, rest_f_p - 1)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c - 1, rest_f - 1)];
        } else if (rest_r <= (R / 2) * 2 && rest_c > (C / 2) * 2 &&
                   rest_f > (F / 2) * 2 && nr % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm_ex)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm_ex, f_sm_ex)];
        } else if (rest_r <= (R / 2) * 2 && rest_c > (C / 2) * 2 &&
                   rest_f <= (F / 2) * 2 && nr % 2 == 0 && nf % 2 == 0) {
          v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, rest_f_p - 1)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm_ex, rest_f - 1)];
        } else if (rest_r <= (R / 2) * 2 && rest_c <= (C / 2) * 2 &&
                   rest_f > (F / 2) * 2 && nr % 2 == 0 && nc % 2 == 0) {
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
        ratio_r_sm[r_sm] = *ratio_r(r + r_sm);
        // if (nr % 2 == 0 && (R/2) * 2 + 1 >= rest_r_p && r_sm == 0) {
        //   ratio_r_sm[rest_r_p - 3] = 0.5;
        // }
      }
      if (r_sm == 0 && f_sm == 0 && c_sm < rest_c_p - 2) {
        ratio_c_sm[c_sm] = *ratio_c(c + c_sm);
        // if (nc % 2 == 0 && (C/2) * 2 + 1 >= rest_c_p && c_sm == 0) {
        //   ratio_c_sm[rest_c_p - 3] = 0.5;
        // }
      }
      if (c_sm == 0 && r_sm == 0 && f_sm < rest_f_p - 2) {
        ratio_f_sm[f_sm] = *ratio_f(f + f_sm);
        // if (nf % 2 == 0 && (F/2) * 2 + 1 >= rest_f_p && f_sm == 0) {
        //   ratio_f_sm[rest_f_p - 3] = 0.5;
        // }
      }

      // if (r == 0 && c == 0 && f == 0 && r_sm == 0 && c_sm == 0 && f_sm == 0)
      // {
      //   printf("ratio:");
      //   for (int i = 0; i < (R/2) * 2 + 1; i++) {
      //     printf("%2.2f ", ratio_r_sm[i]);
      //   }
      //   printf("\n");
      // }

    } // restrict boundary

    // asm volatile("membar.cta;");
    // start = clock64() - start;
    // printf("[load ratio] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
    // blockIdx.y, blockIdx.x, start); start = clock64();
    // bool debug = false;
    // if (threadx == 0 && thready == 0 && threadz == 0) {
    //   debug = true;
    // }

    // __syncthreads();
    // // debug print
    // if (debug) {
    //   printf("in config: %d %d %d (%d %d %d)\n", (R/2), (C/2), (F/2), r,c,f);
    //   printf("rest_p: %d %d %d\n", rest_r_p, rest_c_p, rest_f_p);
    //   for (int i = 0; i < (R/2) * 2 + 1; i++) {
    //     for (int j = 0; j < (C/2) * 2 + 1; j++) {
    //       for (int k = 0; k < (F/2) * 2 + 1; k++) {
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
    if (!w.isNull() && threadId < (R / 2) * (C / 2) * (F / 2)) {
      r_sm = (threadId / ((C / 2) * (F / 2))) * 2;
      c_sm = ((threadId % ((C / 2) * (F / 2))) / (F / 2)) * 2;
      f_sm = ((threadId % ((C / 2) * (F / 2))) % (F / 2)) * 2;
      r_gl = r / 2 + threadId / ((C / 2) * (F / 2));
      c_gl = c / 2 + threadId % ((C / 2) * (F / 2)) / (F / 2);
      f_gl = f / 2 + threadId % ((C / 2) * (F / 2)) % (F / 2);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = res;
        // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
        // r_gl, c_gl, f_gl, !w.isNull()[get_idx(lddw1, lddw2, r_gl, c_gl,
        // f_gl)], r_sm, c_sm, f_sm);
      }
    }

    // asm volatile("membar.cta;");
    // start = clock64() - start;
    // printf("[store coarse] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
    // blockIdx.y, blockIdx.x, start); start = clock64();
    int base = 0;
    // printf("TYPE =%d \n", TYPE);
    // printf("%d == %d && %llu >= %d && %llu < %d\n", r + (R/2) * 2, nr_p - 1,
    // threadId, base, threadId, base + (C/2) * (F/2));

    if (!w.isNull() && r + (R / 2) * 2 == nr_p - 1 && threadId >= base &&
        threadId < base + (C / 2) * (F / 2)) {
      r_sm = (R / 2) * 2;
      c_sm = ((threadId - base) / (F / 2)) * 2;
      f_sm = ((threadId - base) % (F / 2)) * 2;
      r_gl = r / 2 + (R / 2);
      c_gl = c / 2 + (threadId - base) / (F / 2);
      f_gl = f / 2 + (threadId - base) % (F / 2);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = res;
        // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
        // r_gl, c_gl, f_gl, !w.isNull()[get_idx(lddw1, lddw2, r_gl, c_gl,
        // f_gl)], r_sm, c_sm, f_sm);
      }
    }

    base += (C / 2) * (F / 2); // ROUND_UP_WARP((C/2) * (F/2)) * WARP_SIZE;
    if (!w.isNull() && c + (C / 2) * 2 == nc_p - 1 && threadId >= base &&
        threadId < base + (R / 2) * (F / 2)) {
      r_sm = ((threadId - base) / (F / 2)) * 2;
      c_sm = (C / 2) * 2;
      f_sm = ((threadId - base) % (F / 2)) * 2;
      r_gl = r / 2 + (threadId - base) / (F / 2);
      c_gl = c / 2 + (C / 2);
      f_gl = f / 2 + (threadId - base) % (F / 2);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = res;
        // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
        // r_gl, c_gl, f_gl, !w.isNull()[get_idx(lddw1, lddw2, r_gl, c_gl,
        // f_gl)], r_sm, c_sm, f_sm);
      }
      // printf("(%d %d %d) (%d %d %d) %f\n",
      //         r_sm, c_sm, f_sm, r_gl, c_gl, f_gl, dwork[get_idx(lddv1, lddv2,
      //         r_gl, c_gl, f_gl)]);
    }

    base += (R / 2) * (F / 2); // ROUND_UP_WARP((R/2) * (F/2)) * WARP_SIZE;
    // printf("%d %d\n", base,  threadId);
    if (!w.isNull() && f + (F / 2) * 2 == nf_p - 1 && threadId >= base &&
        threadId < base + (R / 2) * (C / 2)) {
      r_sm = ((threadId - base) / (C / 2)) * 2;
      c_sm = ((threadId - base) % (C / 2)) * 2;
      f_sm = (F / 2) * 2;
      r_gl = r / 2 + (threadId - base) / (C / 2);
      c_gl = c / 2 + (threadId - base) % (C / 2);
      f_gl = f / 2 + (F / 2);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = res;
        // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
        // r_gl, c_gl, f_gl, !w.isNull()[get_idx(lddw1, lddw2, r_gl, c_gl,
        // f_gl)], r_sm, c_sm, f_sm);
      }
    }

    base += (R / 2) * (C / 2); // ROUND_UP_WARP((R/2) * (C/2)) * WARP_SIZE;
    // load extra edges
    if (!w.isNull() && c + (C / 2) * 2 == nc_p - 1 &&
        f + (F / 2) * 2 == nf_p - 1 && threadId >= base &&
        threadId < base + (R / 2)) {
      r_sm = (threadId - base) * 2;
      c_sm = (C / 2) * 2;
      f_sm = (F / 2) * 2;
      r_gl = r / 2 + threadId - base;
      c_gl = c / 2 + (C / 2);
      f_gl = f / 2 + (F / 2);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = res;
        // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
        // r_gl, c_gl, f_gl, !w.isNull()[get_idx(lddw1, lddw2, r_gl, c_gl,
        // f_gl)], r_sm, c_sm, f_sm);
      }
    }

    base += (R / 2); // ROUND_UP_WARP((R/2)) * WARP_SIZE;
    // if (TYPE == 2) printf("%d %d, %d, %llu, %d\n",!w.isNull() == NULL, f +
    // (F/2) * 2, nf_p
    // - 1, threadId, (C/2));
    if (!w.isNull() && r + (R / 2) * 2 == nr_p - 1 &&
        f + (F / 2) * 2 == nf_p - 1 && threadId >= base &&
        threadId < base + (C / 2)) {
      r_sm = (R / 2) * 2;
      c_sm = (threadId - base) * 2;
      f_sm = (F / 2) * 2;
      r_gl = r / 2 + (R / 2);
      c_gl = c / 2 + threadId - base;
      f_gl = f / 2 + (F / 2);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = res;
        // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
        // r_gl, c_gl, f_gl, !w.isNull()[get_idx(lddw1, lddw2, r_gl, c_gl,
        // f_gl)], r_sm, c_sm, f_sm);
      }
      // printf("store[%d %d %d]: %f\n", r_sm, c_sm, f_sm, v_sm[get_idx(ldsm1,
      // ldsm2, r_sm, c_sm, f_sm)]);
    }

    base += (C / 2); // ROUND_UP_WARP((C/2)) * WARP_SIZE;
    if (!w.isNull() && r + (R / 2) * 2 == nr_p - 1 &&
        c + (C / 2) * 2 == nc_p - 1 && threadId >= base &&
        threadId < base + (F / 2)) {
      r_sm = (R / 2) * 2;
      c_sm = (C / 2) * 2;
      f_sm = (threadId - base) * 2;
      r_gl = r / 2 + (R / 2);
      c_gl = c / 2 + (C / 2);
      f_gl = f / 2 + threadId - base;
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = res;
        // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
        // r_gl, c_gl, f_gl, !w.isNull()[get_idx(lddw1, lddw2, r_gl, c_gl,
        // f_gl)], r_sm, c_sm, f_sm);
      }
    }
    base += (F / 2); // ROUND_UP_WARP((F/2)) * WARP_SIZE;
    // // load extra vertex
    if (!w.isNull() && r + (R / 2) * 2 == nr_p - 1 &&
        c + (C / 2) * 2 == nc_p - 1 && f + (F / 2) * 2 == nf_p - 1 &&
        threadId >= base && threadId < base + 1) {
      r_sm = (R / 2) * 2;
      c_sm = (C / 2) * 2;
      f_sm = (F / 2) * 2;
      r_gl = r / 2 + (R / 2);
      c_gl = c / 2 + (C / 2);
      f_gl = f / 2 + (F / 2);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = res;
        // printf("w-store: %d+(%d %d %d) <- %f (%d %d %d)\n", other_offset_w,
        // r_gl, c_gl, f_gl, !w.isNull()[get_idx(lddw1, lddw2, r_gl, c_gl,
        // f_gl)], r_sm, c_sm, f_sm);
      }
    }

    // asm volatile("membar.cta;");
    // start = clock64() - start;
    // printf("[store extra] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
    // blockIdx.y, blockIdx.x, start); start = clock64();

    // start = clock64();

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
        res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                   ratio_f_sm[f_sm - 1]);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
        *wf(r_gl, c_gl, f_gl) = res;
      }

      // if (nr == 70)
      // printf("f-store: (%d %d %d) <- %f (%d %d %d)\n", r_gl,
      // c_gl, f_gl, v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)], r_sm, c_sm,
      // f_sm);
      // asm volatile("membar.cta;");
      // start = clock64() - start;
      // printf("[(F/2)-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
      // blockIdx.y, blockIdx.x, start); start = clock64();
    }
    // asm volatile("membar.cta;");
    // start = clock64() - start;
    // printf("[(F/2)-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
    // blockIdx.y, blockIdx.x, start); start = clock64();

    // if (r_sm % 2 == 0 && c_sm % 2 != 0 && f_sm % 2 == 0) {

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
        res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                   ratio_c_sm[c_sm - 1]);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
        *wc(r_gl, c_gl, f_gl) = res;
      }
    }

    // asm volatile("membar.cta;");
    // start = clock64() - start;
    // printf("[(C/2)-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
    // blockIdx.y, blockIdx.x, start); start = clock64();

    // if (r_sm % 2 != 0 && c_sm % 2 == 0 && f_sm % 2 == 0) {
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
        res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                   v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                   ratio_r_sm[r_sm - 1]);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
        *wr(r_gl, c_gl, f_gl) = res;
      }
    }

    // asm volatile("membar.cta;");
    // start = clock64() - start;
    // printf("[(R/2)-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
    // blockIdx.y, blockIdx.x, start); start = clock64();
    // __syncthreads();
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
        T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                    v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                    ratio_f_sm[f_sm - 1]);
        T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                    v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                    ratio_f_sm[f_sm - 1]);
        res = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
        *wcf(r_gl, c_gl, f_gl) = res;
      }
    }

    // asm volatile("membar.cta;");
    // start = clock64() - start;
    // printf("[CF-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
    // blockIdx.y, blockIdx.x, start); start = clock64();

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
        T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                    v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                    ratio_f_sm[f_sm - 1]);
        T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                    v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                    ratio_f_sm[f_sm - 1]);
        res = lerp(f1, f2, ratio_r_sm[r_sm - 1]);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
        *wrf(r_gl, c_gl, f_gl) = res;
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
        T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                    v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                    ratio_c_sm[c_sm - 1]);
        T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                    v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                    ratio_c_sm[c_sm - 1]);
        res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
        *wrc(r_gl, c_gl, f_gl) = res;
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
        *wrcf(r_gl, c_gl, f_gl) = res;
      }
    }
    // end = clock64();

    // asm volatile("membar.cta;");
    // if (threadId < 256 && blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x ==
    // 0) printf("threadId %d elapsed %lu\n", threadId, end-start);
    if (r + (R / 2) * 2 == nr_p - 1) {
      // printf("test\n");
      if (threadId < (C / 2) * (F / 2)) {
        // printf("test1\n");
        if (!wf.isNull()) {
          // printf("test2\n");
          r_sm = (R / 2) * 2;
          c_sm = (threadId / (F / 2)) * 2;
          f_sm = (threadId % (F / 2)) * 2 + 1;
          r_gl = r / 2 + (R / 2);
          c_gl = c / 2 + threadId / (F / 2);
          f_gl = f / 2 + threadId % (F / 2);
          if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
              r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                       ratio_f_sm[f_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            // printf("!wf.isNull() (%d %d %d): %f<-(%f %f %f)\n", r_gl, c_gl,
            // f_gl, res,
            //   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
            //                v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
            //                ratio_f_sm[f_sm - 1]);
            *wf(r_gl, c_gl, f_gl) = res;
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
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                       ratio_c_sm[c_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wc(r_gl, c_gl, f_gl) = res;
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
            T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            res = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wcf(r_gl, c_gl, f_gl) = res;
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
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                       ratio_f_sm[f_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wf(r_gl, c_gl, f_gl) = res;
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
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                       ratio_r_sm[r_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wr(r_gl, c_gl, f_gl) = res;
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
            T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            res = lerp(f1, f2, ratio_r_sm[r_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wrf(r_gl, c_gl, f_gl) = res;
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
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                       ratio_c_sm[c_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wc(r_gl, c_gl, f_gl) = res;
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
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                       ratio_r_sm[r_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wr(r_gl, c_gl, f_gl) = res;
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
            T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wrc(r_gl, c_gl, f_gl) = res;
          }
        }
      }
    }

    if (!wr.isNull() && c + (C / 2) * 2 == nc_p - 1 &&
        f + (F / 2) * 2 == nf_p - 1) {
      if (threadId >= (R / 2) * (C / 2) * (F / 2) * 3 &&
          threadId < (R / 2) * (C / 2) * (F / 2) * 3 + (R / 2)) {
        r_sm = (threadId - (R / 2) * (C / 2) * (F / 2) * 3) * 2 + 1;
        c_sm = (C / 2) * 2;
        f_sm = (F / 2) * 2;
        r_gl = r / 2 + threadId - (R / 2) * (C / 2) * (F / 2) * 3;
        c_gl = c / 2 + (C / 2);
        f_gl = f / 2 + (F / 2);
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
          res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                     ratio_r_sm[r_sm - 1]);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
          *wr(r_gl, c_gl, f_gl) = res;
        }
      }
    }

    if (!wc.isNull() && r + (R / 2) * 2 == nr_p - 1 &&
        f + (F / 2) * 2 == nf_p - 1) {
      if (threadId >= (R / 2) * (C / 2) * (F / 2) * 4 &&
          threadId < (R / 2) * (C / 2) * (F / 2) * 4 + (C / 2)) {
        r_sm = (R / 2) * 2;
        c_sm = (threadId - (R / 2) * (C / 2) * (F / 2) * 4) * 2 + 1;
        f_sm = (F / 2) * 2;
        r_gl = r / 2 + (R / 2);
        c_gl = c / 2 + threadId - (R / 2) * (C / 2) * (F / 2) * 4;
        f_gl = f / 2 + (F / 2);
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
          res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                     ratio_c_sm[c_sm - 1]);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
          *wc(r_gl, c_gl, f_gl) = res;
        }
      }
    }

    if (!wf.isNull() && r + (R / 2) * 2 == nr_p - 1 &&
        c + (C / 2) * 2 == nc_p - 1) {
      if (threadId >= (R / 2) * (C / 2) * (F / 2) * 5 &&
          threadId < (R / 2) * (C / 2) * (F / 2) * 5 + (F / 2)) {
        r_sm = (R / 2) * 2;
        c_sm = (C / 2) * 2;
        f_sm = (threadId - (R / 2) * (C / 2) * (F / 2) * 5) * 2 + 1;
        r_gl = r / 2 + (R / 2);
        c_gl = c / 2 + (C / 2);
        f_gl = f / 2 + threadId - (R / 2) * (C / 2) * (F / 2) * 5;
        if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
            r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
          res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                     v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                     ratio_f_sm[f_sm - 1]);
          res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
          *wf(r_gl, c_gl, f_gl) = res;
          // printf("!wf.isNull()(%d %d %d): %f\n", r_gl, c_gl, f_gl,
          // !wf.isNull()[get_idx(lddwf1, lddwf2, r_gl, c_gl, f_gl)]);
        }
      }
    }

    // if (r == 0 && c == 0 && f == 0 && threadId == 0) {
    //   printf("out config: %d %d %d (%d %d %d)\n", (R/2), (C/2), (F/2),
    //   r,c,f); for (int i = 0; i < (R/2) * 2 + 1; i++) {
    //     for (int j = 0; j < (C/2) * 2 + 1; j++) {
    //       for (int k = 0; k < (F/2) * 2 + 1; k++) {
    //         printf("%2.2f ", v_sm[get_idx(ldsm1, ldsm2, i, j, k)]);
    //       }
    //       printf("\n");
    //     }
    //     printf("\n");
    //   }
    // }
  }

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

private:
  // functor parameters
  SIZE nr, nc, nf, nr_c, nc_c, nf_c;
  SubArray<1, T, DeviceType> ratio_r, ratio_c, ratio_f;
  SubArray<D, T, DeviceType> v, w, wf, wc, wr, wcf, wrf, wrc, wrcf;

  // thread local variables
  SIZE r, c, f;
  SIZE rest_r, rest_c, rest_f;
  SIZE nr_p, nc_p, nf_p;
  SIZE rest_r_p, rest_c_p, rest_f_p;
  SIZE r_sm, c_sm, f_sm;
  SIZE r_sm_ex, c_sm_ex, f_sm_ex;
  SIZE r_gl, c_gl, f_gl;
  SIZE r_gl_ex, c_gl_ex, f_gl_ex;
  SIZE threadId;
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
class GpkReo3DKernel : public Kernel {
public:
  constexpr static DIM NumDim = D;
  using DataType = T;
  constexpr static std::string_view Name = "gpk_reo_3d";

  MGARDX_CONT
  GpkReo3DKernel(SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
                 SubArray<1, T, DeviceType> ratio_r,
                 SubArray<1, T, DeviceType> ratio_c,
                 SubArray<1, T, DeviceType> ratio_f,
                 SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> w,
                 SubArray<D, T, DeviceType> wf, SubArray<D, T, DeviceType> wc,
                 SubArray<D, T, DeviceType> wr, SubArray<D, T, DeviceType> wcf,
                 SubArray<D, T, DeviceType> wrf, SubArray<D, T, DeviceType> wrc,
                 SubArray<D, T, DeviceType> wrcf)
      : nr(nr), nc(nc), nf(nf), nr_c(nr_c), nc_c(nc_c), nf_c(nf_c),
        ratio_r(ratio_r), ratio_c(ratio_c), ratio_f(ratio_f), v(v), w(w),
        wf(wf), wc(wc), wr(wr), wcf(wcf), wrf(wrf), wrc(wrc), wrcf(wrcf) {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<GpkReo3DFunctor<D, T, R, C, F, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = GpkReo3DFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(nr, nc, nf, nr_c, nc_c, nf_c, ratio_r, ratio_c, ratio_f,
                        v, w, wf, wc, wr, wcf, wrf, wrc, wrcf);

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
                std::string(Name));
  }

private:
  SIZE nr, nc, nf, nr_c, nc_c, nf_c;
  SubArray<1, T, DeviceType> ratio_r, ratio_c, ratio_f;
  SubArray<D, T, DeviceType> v, w, wf, wc, wr, wcf, wrf, wrc, wrcf;
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class GpkRev3DFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT GpkRev3DFunctor() {}
  MGARDX_CONT GpkRev3DFunctor(
      SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
      SubArray<1, T, DeviceType> ratio_r, SubArray<1, T, DeviceType> ratio_c,
      SubArray<1, T, DeviceType> ratio_f, SubArray<D, T, DeviceType> v,
      SubArray<D, T, DeviceType> w, SubArray<D, T, DeviceType> wf,
      SubArray<D, T, DeviceType> wc, SubArray<D, T, DeviceType> wr,
      SubArray<D, T, DeviceType> wcf, SubArray<D, T, DeviceType> wrf,
      SubArray<D, T, DeviceType> wrc, SubArray<D, T, DeviceType> wrcf, SIZE svr,
      SIZE svc, SIZE svf, SIZE nvr, SIZE nvc, SIZE nvf)
      : nr(nr), nc(nc), nf(nf), nr_c(nr_c), nc_c(nc_c), nf_c(nf_c),
        ratio_r(ratio_r), ratio_c(ratio_c), ratio_f(ratio_f), v(v), w(w),
        wf(wf), wc(wc), wr(wr), wrcf(wrcf), wcf(wcf), wrf(wrf), wrc(wrc),
        svr(svr), svc(svc), svf(svf), nvr(nvr), nvc(nvc), nvf(nvf) {
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

    // load dist
    if (c_sm == 0 && f_sm == 0 && r_sm < rest_r - 2) {
      ratio_r_sm[r_sm] = *ratio_r(r + r_sm);
      if (nr % 2 == 0 && (R / 2) * 2 + 1 >= rest_r_p && r_sm == 0) {
        ratio_r_sm[rest_r_p - 3] = 0.5;
      }
    }
    if (r_sm == 0 && f_sm == 0 && c_sm < rest_c - 2) {
      ratio_c_sm[c_sm] = *ratio_c(c + c_sm);
      if (nc % 2 == 0 && (C / 2) * 2 + 1 >= rest_c_p && c_sm == 0) {
        ratio_c_sm[rest_c_p - 3] = 0.5;
      }
    }
    if (c_sm == 0 && r_sm == 0 && f_sm < rest_f - 2) {
      ratio_f_sm[f_sm] = *ratio_f(f + f_sm);
      if (nf % 2 == 0 && (F / 2) * 2 + 1 >= rest_f_p && f_sm == 0) {
        ratio_f_sm[rest_f_p - 3] = 0.5;
      }
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
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = *w(r_gl, c_gl, f_gl);
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
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = *w(r_gl, c_gl, f_gl);
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
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = *w(r_gl, c_gl, f_gl);
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
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = *w(r_gl, c_gl, f_gl);
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
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = *w(r_gl, c_gl, f_gl);
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
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = *w(r_gl, c_gl, f_gl);
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
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = *w(r_gl, c_gl, f_gl);
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
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = *w(r_gl, c_gl, f_gl);
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
        res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                    v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                    ratio_f_sm[f_sm - 1]);
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
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
        res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                    v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                    ratio_c_sm[c_sm - 1]);
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
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
            res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
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
            res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
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
            res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
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
            res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
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
            res += lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] = res;
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
    r_sm = FunctorBase<DeviceType>::GetThreadIdZ();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();

    r_sm_ex = FunctorBase<DeviceType>::GetBlockDimZ();
    c_sm_ex = FunctorBase<DeviceType>::GetBlockDimY();
    f_sm_ex = FunctorBase<DeviceType>::GetBlockDimX();

    r_gl = r + r_sm;
    c_gl = c + c_sm;
    f_gl = f + f_sm;

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
        if (nr % 2 != 0 && (R / 2) * 2 + 1 == rest_r) {
          *v(r_gl_ex, c_gl, f_gl) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm)];
        }
        if (nr % 2 == 0 && (R / 2) * 2 + 1 >= rest_r_p) {
          v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, f_sm)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm)];
          // if ( v_sm[get_idx(ldsm1, ldsm2, rest_r_p-1, c_sm, f_sm)] ==
          // 71177117) printf("un-padding0 error block: (%d %d %d) thread: (%d
          // %d %d) un-padding (%d %d %d) %f (%d %d %d)\n", blockIdx.z,
          // blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
          //   rest_r-1, c_sm, f_sm,
          //     v_sm[get_idx(ldsm1, ldsm2, rest_r_p-1, c_sm, f_sm)],
          //     rest_r_p-1, c_sm, f_sm);
        }
      }

      if (D >= 2 && c_sm == 0) {
        if (nc % 2 != 0 && (C / 2) * 2 + 1 == rest_c) {
          *v(r_gl, c_gl_ex, f_gl) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)];
        }
        if (nc % 2 == 0 && (C / 2) * 2 + 1 >= rest_c_p) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, f_sm)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm)];
          // if (v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm)] ==
          // 71177117)
          //   printf("un-padding1 error block: (%d %d %d) thread: (%d %d %d) "
          //          "un-padding (%d %d %d) %f (%d %d %d)\n",
          //          blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
          //          threadIdx.y, threadIdx.x, r_sm, rest_c - 1, f_sm,
          //          v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm)],
          //          r_sm, rest_c_p - 1, f_sm);
        }
      }

      if (D >= 1 && f_sm == 0) {
        if (nf % 2 != 0 && (F / 2) * 2 + 1 == rest_f) {
          *v(r_gl, c_gl, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm_ex)];
        }
        if (nf % 2 == 0 && (F / 2) * 2 + 1 >= rest_f_p) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f - 1)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f_p - 1)];
          // if ( v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f_p-1)] ==
          // 71177117) printf("un-padding2 error block: (%d %d %d) thread: (%d
          // %d %d) un-padding (%d %d %d) %f (%d %d %d)\n", blockIdx.z,
          // blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
          //   r_sm, c_sm, rest_f-1,
          //     v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, rest_f_p-1)], r_sm,
          //     c_sm, rest_f_p-1);
        }
      }

      // load extra edges
      if (D >= 2 && c_sm == 0 && f_sm == 0) {
        if (nc % 2 != 0 && (C / 2) * 2 + 1 == rest_c && nf % 2 != 0 &&
            (F / 2) * 2 + 1 == rest_f) {
          *v(r_gl, c_gl_ex, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm_ex)];
        }
        if (nc % 2 == 0 && nf % 2 == 0 && (C / 2) * 2 + 1 >= rest_c_p &&
            (F / 2) * 2 + 1 >= rest_f_p) {
          v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c - 1, rest_f - 1)] =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, rest_f_p - 1)];
          // printf("block: (%d %d %d) thread: (%d %d %d) un-padding (%d %d %d)
          // %f
          // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
          // threadIdx.y, threadIdx.x, r_sm, rest_c-1, rest_f-1,
          //     v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c-1, rest_f-1)], r_sm,
          //     rest_c_p-1, rest_f_p-1);
        }
        if (nc % 2 == 0 && nf % 2 != 0 && (C / 2) * 2 + 1 >= rest_c_p &&
            (F / 2) * 2 + 1 == rest_f) {
          *v(r_gl, c_gl_ex, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c_p - 1, f_sm_ex)];
        }
        if (nc % 2 != 0 && nf % 2 == 0 && (C / 2) * 2 + 1 == rest_c &&
            (F / 2) * 2 + 1 >= rest_f_p) {
          *v(r_gl, c_gl_ex, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, rest_f_p - 1)];
          // printf("(%d %d %d): %f <- (%d %d %d)\n",
          //         r_gl, c_gl_ex, f_gl_ex,
          //         dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl_ex)],
          //         r_sm, c_sm_ex, f_gl_ex);
        }
      }

      if (D >= 3 && r_sm == 0 && f_sm == 0) {
        if (nr % 2 != 0 && (R / 2) * 2 + 1 == rest_r && nf % 2 != 0 &&
            (F / 2) * 2 + 1 == rest_f) {
          *v(r_gl_ex, c_gl, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, f_sm_ex)];
        }
        if (nr % 2 == 0 && nf % 2 == 0 && (R / 2) * 2 + 1 >= rest_r_p &&
            (F / 2) * 2 + 1 >= rest_f_p) {
          v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, c_sm, rest_f - 1)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, rest_f_p - 1)];
          // if ( v_sm[get_idx(ldsm1, ldsm2, rest_r_p-1, c_sm, rest_f_p-1)] ==
          // 71177117) printf("un-padding3 error block: (%d %d %d) thread: (%d
          // %d %d) un-padding (%d %d %d) %f (%d %d %d)\n", blockIdx.z,
          // blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
          //   rest_r-1, c_sm, rest_f-1,
          //     v_sm[get_idx(ldsm1, ldsm2, rest_r_p-1, c_sm, rest_f_p-1)],
          //     rest_r_p-1, c_sm, rest_f_p-1);
        }
        if (nr % 2 == 0 && nf % 2 != 0 && (R / 2) * 2 + 1 >= rest_r_p &&
            (F / 2) * 2 + 1 == rest_f) {
          *v(r_gl_ex, c_gl, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm, f_sm_ex)];
        }
        if (nr % 2 != 0 && nf % 2 == 0 && (R / 2) * 2 + 1 == rest_r &&
            (F / 2) * 2 + 1 >= rest_f_p) {
          *v(r_gl_ex, c_gl, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm, rest_f_p - 1)];
          // printf("(%d %d %d): %f <- (%d %d %d)\n",
          //         r_gl_ex, c_gl, rest_f-1,
          //         dv[get_idx(lddv1, lddv2, r_gl_ex-1, c_gl, f_gl_ex)],
          //         r_sm_ex, c_sm, rest_f_p-1);
        }
      }

      if (D >= 3 && r_sm == 0 && c_sm == 0) {
        if (nr % 2 != 0 && (R / 2) * 2 + 1 == rest_r && nc % 2 != 0 &&
            (C / 2) * 2 + 1 == rest_c) {
          *v(r_gl_ex, c_gl_ex, f_gl) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm)];
        }
        if (nr % 2 == 0 && nc % 2 == 0 && (R / 2) * 2 + 1 >= rest_r_p &&
            (C / 2) * 2 + 1 >= rest_c_p) {
          v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, f_sm)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1, f_sm)];
          // if ( v_sm[get_idx(ldsm1, ldsm2, rest_r_p-1, rest_c_p-1, f_sm)] ==
          // 71177117) printf("un-padding4 error block: (%d %d %d) thread: (%d
          // %d %d) un-padding (%d %d %d) %f (%d %d %d)\n", blockIdx.z,
          // blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x,
          //   rest_r-1, rest_c-1, f_sm,
          //     v_sm[get_idx(ldsm1, ldsm2, rest_r_p-1, rest_c_p-1, f_sm)],
          //     rest_r_p-1, rest_c_p-1, f_sm);
        }
        if (nr % 2 == 0 && nc % 2 != 0 && (R / 2) * 2 + 1 >= rest_r_p &&
            (C / 2) * 2 + 1 == rest_c) {
          *v(r_gl_ex, c_gl_ex, f_gl) =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm)];
        }
        if (nr % 2 != 0 && nc % 2 == 0 && (R / 2) * 2 + 1 == rest_r &&
            (C / 2) * 2 + 1 >= rest_c_p) {
          *v(r_gl_ex, c_gl_ex, f_gl) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm)];
        }
      }
      // load extra vertex

      if (D >= 3 && r_sm == 0 && c_sm == 0 && f_sm == 0) {
        if (nr % 2 != 0 && (R / 2) * 2 + 1 == rest_r && nc % 2 != 0 &&
            (C / 2) * 2 + 1 == rest_c && nf % 2 != 0 &&
            (F / 2) * 2 + 1 == rest_f) {
          *v(r_gl_ex, c_gl_ex, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, f_sm_ex)];
        }

        if (nr % 2 == 0 && nc % 2 == 0 && nf % 2 == 0 &&
            (R / 2) * 2 + 1 >= rest_r_p && (C / 2) * 2 + 1 >= rest_c_p &&
            (F / 2) * 2 + 1 >= rest_f_p) {
          v_sm[get_idx(ldsm1, ldsm2, rest_r - 1, rest_c - 1, rest_f - 1)] =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1,
                           rest_f_p - 1)];

          // printf("block: (%d %d %d) thread: (%d %d %d) un-padding (%d %d %d)
          // %f
          // (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
          // threadIdx.y, threadIdx.x, rest_r-1, rest_c-1, rest_f-1,
          //     v_sm[get_idx(ldsm1, ldsm2, r_sm, rest_c-1, rest_f-1)],
          //     rest_r_p-1, rest_c_p-1, rest_f_p-1);
        }
        if (nr % 2 == 0 && nc % 2 == 0 && nf % 2 != 0 &&
            (R / 2) * 2 + 1 >= rest_r_p && (C / 2) * 2 + 1 >= rest_c_p &&
            (F / 2) * 2 + 1 == rest_f) {
          *v(r_gl_ex, c_gl_ex, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, rest_c_p - 1, f_sm_ex)];
        }
        if (nr % 2 == 0 && nc % 2 != 0 && nf % 2 == 0 &&
            (R / 2) * 2 + 1 >= rest_r_p && (C / 2) * 2 + 1 == rest_c &&
            (F / 2) * 2 + 1 >= rest_f_p) {
          *v(r_gl_ex, c_gl_ex, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, rest_f_p - 1)];
        }
        if (nr % 2 != 0 && nc % 2 == 0 && nf % 2 == 0 &&
            (R / 2) * 2 + 1 == rest_r && (C / 2) * 2 + 1 >= rest_c_p &&
            (F / 2) * 2 + 1 >= rest_f_p) {
          *v(r_gl_ex, c_gl_ex, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, rest_f_p - 1)];
        }
        if (nr % 2 == 0 && nc % 2 != 0 && nf % 2 != 0 &&
            (R / 2) * 2 + 1 >= rest_r_p && (C / 2) * 2 + 1 == rest_c &&
            (F / 2) * 2 + 1 == rest_f) {
          *v(r_gl_ex, c_gl_ex, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, rest_r_p - 1, c_sm_ex, f_sm_ex)];
        }
        if (nr % 2 != 0 && nc % 2 == 0 && nf % 2 != 0 &&
            (R / 2) * 2 + 1 == rest_r && (C / 2) * 2 + 1 >= rest_c_p &&
            (F / 2) * 2 + 1 == rest_f) {
          *v(r_gl_ex, c_gl_ex, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, rest_c_p - 1, f_sm_ex)];
        }
        if (nr % 2 != 0 && nc % 2 != 0 && nf % 2 == 0 &&
            (R / 2) * 2 + 1 == rest_r && (C / 2) * 2 + 1 == rest_c &&
            (F / 2) * 2 + 1 >= rest_f_p) {
          *v(r_gl_ex, c_gl_ex, f_gl_ex) =
              v_sm[get_idx(ldsm1, ldsm2, r_sm_ex, c_sm_ex, rest_f_p - 1)];
        }
      }
    }
  }

  MGARDX_EXEC void Operation5() {
    if (r_sm < rest_r && c_sm < rest_c && f_sm < rest_f) {
      if (r_gl >= svr && r_gl < svr + nvr && c_gl >= svc && c_gl < svc + nvc &&
          f_gl >= svf && f_gl < svf + nvf) {
        *v(r_gl, c_gl, f_gl) = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];

        // if (c_gl == nc - 1 && f_gl == nf - 1) {
        //   printf("block: (%d %d %d) thread: (%d %d %d) store (%d %d %d) %f
        //   (%d %d %d)\n", blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z,
        //   threadIdx.y, threadIdx.x, r_gl, c_gl, f_gl,
        //     v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)], r_sm, c_sm, f_sm);
        // }
      }
    }
  }

private:
  // functor parameters
  SIZE nr, nc, nf, nr_c, nc_c, nf_c;
  SubArray<1, T, DeviceType> ratio_r, ratio_c, ratio_f;
  SubArray<D, T, DeviceType> v, w, wf, wc, wr, wcf, wrf, wrc, wrcf;
  SIZE svr, svc, svf, nvr, nvc, nvf;

  // thread local variables
  SIZE r, c, f;
  SIZE rest_r, rest_c, rest_f;
  SIZE nr_p, nc_p, nf_p;
  SIZE rest_r_p, rest_c_p, rest_f_p;
  SIZE r_sm, c_sm, f_sm;
  SIZE r_sm_ex, c_sm_ex, f_sm_ex;
  SIZE r_gl, c_gl, f_gl;
  SIZE r_gl_ex, c_gl_ex, f_gl_ex;
  SIZE threadId;
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
class GpkRev3DKernel : public Kernel {
public:
  constexpr static DIM NumDim = D;
  using DataType = T;
  constexpr static std::string_view Name = "gpk_rev_3d";
  MGARDX_CONT
  GpkRev3DKernel(SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
                 SubArray<1, T, DeviceType> ratio_r,
                 SubArray<1, T, DeviceType> ratio_c,
                 SubArray<1, T, DeviceType> ratio_f,
                 SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> w,
                 SubArray<D, T, DeviceType> wf, SubArray<D, T, DeviceType> wc,
                 SubArray<D, T, DeviceType> wr, SubArray<D, T, DeviceType> wcf,
                 SubArray<D, T, DeviceType> wrf, SubArray<D, T, DeviceType> wrc,
                 SubArray<D, T, DeviceType> wrcf, SIZE svr, SIZE svc, SIZE svf,
                 SIZE nvr, SIZE nvc, SIZE nvf)
      : nr(nr), nc(nc), nf(nf), nr_c(nr_c), nc_c(nc_c), nf_c(nf_c),
        ratio_r(ratio_r), ratio_c(ratio_c), ratio_f(ratio_f), v(v), w(w),
        wf(wf), wc(wc), wr(wr), wrcf(wrcf), wcf(wcf), wrf(wrf), wrc(wrc),
        svr(svr), svc(svc), svf(svf), nvr(nvr), nvc(nvc), nvf(nvf) {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<GpkRev3DFunctor<D, T, R, C, F, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = GpkRev3DFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(nr, nc, nf, nr_c, nc_c, nf_c, ratio_r, ratio_c, ratio_f,
                        v, w, wf, wc, wr, wcf, wrf, wrc, wrcf, svr, svc, svf,
                        nvr, nvc, nvf);

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
                std::string(Name));
  }

private:
  // functor parameters
  SIZE nr, nc, nf, nr_c, nc_c, nf_c;
  SubArray<1, T, DeviceType> ratio_r, ratio_c, ratio_f;
  SubArray<D, T, DeviceType> v, w, wf, wc, wr, wcf, wrf, wrc, wrcf;
  SIZE svr, svc, svf, nvr, nvc, nvf;
};

} // namespace multi_dimension

} // namespace data_refactoring

} // namespace mgard_x

#endif