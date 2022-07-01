/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_GRID_PROCESSING_KERNEL_3D_WITH_ERROR_COLLECTION_TEMPLATE
#define MGARD_X_GRID_PROCESSING_KERNEL_3D_WITH_ERROR_COLLECTION_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"
#include "GPKFunctor.h"
// #include "GridProcessingKernel3D.h"

// #include "../../Functor.h"
// #include "../../AutoTuners/AutoTuner.h"
// #include "../../Task.h"
// #include "../../DeviceAdapters/DeviceAdapter.h"

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class GpkReo3DWithErrorCollectionFunctor : public Functor<DeviceType> {
public:
  using AtomicOp = Atomic<T, AtomicGlobalMemory, AtomicDeviceScope, DeviceType>;
  MGARDX_CONT GpkReo3DWithErrorCollectionFunctor() {}
  MGARDX_CONT GpkReo3DWithErrorCollectionFunctor(
      SIZE level, SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
      SubArray<1, T, DeviceType> ratio_r, SubArray<1, T, DeviceType> ratio_c,
      SubArray<1, T, DeviceType> ratio_f, SubArray<D, T, DeviceType> v,
      SubArray<D, T, DeviceType> w, SubArray<D, T, DeviceType> wf,
      SubArray<D, T, DeviceType> wc, SubArray<D, T, DeviceType> wr,
      SubArray<D, T, DeviceType> wcf, SubArray<D, T, DeviceType> wrf,
      SubArray<D, T, DeviceType> wrc, SubArray<D, T, DeviceType> wrcf,
      SubArray<D+1, T, DeviceType> max_abs_coefficient,
      SubArray<D+1, T, DeviceType> max_abs_coefficient_finer)
      : level(level), nr(nr), nc(nc), nf(nf), nr_c(nr_c), nc_c(nc_c), nf_c(nf_c),
        ratio_r(ratio_r), ratio_c(ratio_c), ratio_f(ratio_f), v(v), w(w),
        wf(wf), wc(wc), wr(wr), wcf(wcf), wrf(wrf), wrc(wrc), wrcf(wrcf),
        max_abs_coefficient(max_abs_coefficient),
        max_abs_coefficient_finer(max_abs_coefficient_finer) {
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
      res = 0;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];;
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
      res = 0;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];;
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
      res = 0;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];;
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
      res = 0;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];;
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
      res = 0;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];;
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
      res = 0;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];;
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
      res = 0;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];;
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
      res = 0;
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
        *w(r_gl, c_gl, f_gl) = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];;
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
      r_sm = ((threadId - (R / 2) * (C / 2) * (F / 2)) / ((C / 2) * (F / 2))) * 2;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2)) % ((C / 2) * (F / 2))) / (F / 2)) * 2;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2)) % ((C / 2) * (F / 2))) % (F / 2)) * 2 + 1;
      r_gl = r / 2 + (threadId - (R / 2) * (C / 2) * (F / 2)) / ((C / 2) * (F / 2));
      c_gl = c / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2)) % ((C / 2) * (F / 2))) / (F / 2);
      f_gl = f / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2)) % ((C / 2) * (F / 2))) % (F / 2);

      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc_c && f_gl < nf - nf_c) {
        res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                   ratio_f_sm[f_sm - 1]);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
        *wf(r_gl, c_gl, f_gl) = res;
        T abs_coefficient = fabs(res);
        if (f_gl < nf_c-1) {
          if (r_gl < nr_c-1 && c_gl < nc_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
          if (r_gl < nr_c-1 && c_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl), abs_coefficient);
          if (r_gl >= 1     && c_gl < nc_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl), abs_coefficient);
          if (r_gl >= 1     && c_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl-1, f_gl), abs_coefficient);
        }
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
      r_sm = ((threadId - (R / 2) * (C / 2) * (F / 2) * 2) / ((C / 2) * (F / 2))) * 2;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 2) %
               ((C / 2) * (F / 2))) / (F / 2)) * 2 + 1;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 2) %
               ((C / 2) * (F / 2))) % (F / 2)) * 2;
      r_gl = r / 2 + (threadId - (R / 2) * (C / 2) * (F / 2) * 2) / ((C / 2) * (F / 2));
      c_gl = c / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 2) % ((C / 2) * (F / 2))) / (F / 2);
      f_gl = f / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 2) % ((C / 2) * (F / 2))) % (F / 2);
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c && c_gl < nc - nc_c && f_gl < nf_c) {
        res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                   v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                   ratio_c_sm[c_sm - 1]);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
        *wc(r_gl, c_gl, f_gl) = res;
        T abs_coefficient = fabs(res);
        if (c_gl < nc_c-1) {
          if (r_gl < nr_c-1 && f_gl < nf_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
          if (r_gl < nr_c-1 && f_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl-1), abs_coefficient);
          if (r_gl >= 1     && f_gl < nf_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl), abs_coefficient);
          if (r_gl >= 1     && f_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl-1), abs_coefficient);
        }
      }
    }

    // asm volatile("membar.cta;");
    // start = clock64() - start;
    // printf("[(C/2)-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
    // blockIdx.y, blockIdx.x, start); start = clock64();

    // if (r_sm % 2 != 0 && c_sm % 2 == 0 && f_sm % 2 == 0) {
    if (!wr.isNull() && threadId >= (R / 2) * (C / 2) * (F / 2) * 3 &&
        threadId < (R / 2) * (C / 2) * (F / 2) * 4) {
      r_sm = ((threadId - (R / 2) * (C / 2) * (F / 2) * 3) / ((C / 2) * (F / 2))) * 2 + 1;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 3) % ((C / 2) * (F / 2))) / (F / 2)) * 2;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 3) % ((C / 2) * (F / 2))) % (F / 2)) * 2;
      r_gl = r / 2 + (threadId - (R / 2) * (C / 2) * (F / 2) * 3) / ((C / 2) * (F / 2));
      c_gl = c / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 3) % ((C / 2) * (F / 2))) / (F / 2);
      f_gl = f / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 3) % ((C / 2) * (F / 2))) % (F / 2);
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr - nr_c && c_gl < nc_c && f_gl < nf_c) {
        res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                   v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                   ratio_r_sm[r_sm - 1]);
        res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
        *wr(r_gl, c_gl, f_gl) = res;
        // printf("rcf: %u %u %u wr-res: %f\n", r_gl, c_gl, f_gl, res);
        T abs_coefficient = fabs(res);
        if (r_gl < nr_c-1) {
          if (c_gl < nc_c-1 && f_gl < nf_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
          if (c_gl < nc_c-1 && f_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl-1), abs_coefficient);
          if (c_gl >= 1     && f_gl < nf_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl), abs_coefficient);
          if (c_gl >= 1     && f_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl-1), abs_coefficient);
        }
      }
    }

    // asm volatile("membar.cta;");
    // start = clock64() - start;
    // printf("[(R/2)-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
    // blockIdx.y, blockIdx.x, start); start = clock64();
    // __syncthreads();
    if (!wcf.isNull() && threadId >= (R / 2) * (C / 2) * (F / 2) * 4 &&
        threadId < (R / 2) * (C / 2) * (F / 2) * 5) {
      r_sm = ((threadId - (R / 2) * (C / 2) * (F / 2) * 4) / ((C / 2) * (F / 2))) * 2;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 4) % ((C / 2) * (F / 2))) / (F / 2)) * 2 + 1;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 4) % ((C / 2) * (F / 2))) % (F / 2)) * 2 + 1;
      r_gl = r / 2 + (threadId - (R / 2) * (C / 2) * (F / 2) * 4) / ((C / 2) * (F / 2));
      c_gl = c / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 4) % ((C / 2) * (F / 2))) / (F / 2);
      f_gl = f / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 4) % ((C / 2) * (F / 2))) % (F / 2);
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
        T abs_coefficient = fabs(res);
        if (c_gl < nc_c-1 && f_gl < nf_c-1) {
          if (r_gl < nr_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
          if (r_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl), abs_coefficient);
        }
      }
    }

    // asm volatile("membar.cta;");
    // start = clock64() - start;
    // printf("[CF-store] block id %d,%d,%d elapsed %lu\n", blockIdx.z,
    // blockIdx.y, blockIdx.x, start); start = clock64();

    if (!wrf.isNull() && threadId >= (R / 2) * (C / 2) * (F / 2) * 5 &&
        threadId < (R / 2) * (C / 2) * (F / 2) * 6) {
      r_sm = ((threadId - (R / 2) * (C / 2) * (F / 2) * 5) / ((C / 2) * (F / 2))) * 2 + 1;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 5) % ((C / 2) * (F / 2))) / (F / 2)) * 2;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 5) % ((C / 2) * (F / 2))) % (F / 2)) * 2 + 1;
      r_gl = r / 2 + (threadId - (R / 2) * (C / 2) * (F / 2) * 5) / ((C / 2) * (F / 2));
      c_gl = c / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 5) % ((C / 2) * (F / 2))) / (F / 2);
      f_gl = f / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 5) % ((C / 2) * (F / 2))) % (F / 2);
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
        T abs_coefficient = fabs(res);
        if (r_gl < nr_c-1 && f_gl < nf_c-1) {
          if (c_gl < nc_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
          if (c_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl), abs_coefficient);
        }
      }
    }

    if (!wrc.isNull() && threadId >= (R / 2) * (C / 2) * (F / 2) * 6 &&
        threadId < (R / 2) * (C / 2) * (F / 2) * 7) {
      r_sm = ((threadId - (R / 2) * (C / 2) * (F / 2) * 6) / ((C / 2) * (F / 2))) * 2 + 1;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 6) % ((C / 2) * (F / 2))) / (F / 2)) * 2 + 1;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 6) % ((C / 2) * (F / 2))) % (F / 2)) * 2;
      r_gl = r / 2 + (threadId - (R / 2) * (C / 2) * (F / 2) * 6) / ((C / 2) * (F / 2));
      c_gl = c / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 6) % ((C / 2) * (F / 2))) / (F / 2);
      f_gl = f / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 6) % ((C / 2) * (F / 2))) % (F / 2);
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
        T abs_coefficient = fabs(res);
        if (r_gl < nr_c-1 && c_gl < nc_c-1) {
          if (f_gl < nf_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
          if (f_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl-1), abs_coefficient);
        }
      }
    }

    if (!wrcf.isNull() && threadId >= (R / 2) * (C / 2) * (F / 2) * 7 &&
        threadId < (R / 2) * (C / 2) * (F / 2) * 8) {
      r_sm = ((threadId - (R / 2) * (C / 2) * (F / 2) * 7) / ((C / 2) * (F / 2))) * 2 + 1;
      c_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 7) % ((C / 2) * (F / 2))) / (F / 2)) * 2 + 1;
      f_sm = (((threadId - (R / 2) * (C / 2) * (F / 2) * 7) % ((C / 2) * (F / 2))) % (F / 2)) * 2 + 1;
      r_gl = r / 2 + (threadId - (R / 2) * (C / 2) * (F / 2) * 7) / ((C / 2) * (F / 2));
      c_gl = c / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 7) % ((C / 2) * (F / 2))) / (F / 2);
      f_gl = f / 2 + ((threadId - (R / 2) * (C / 2) * (F / 2) * 7) % ((C / 2) * (F / 2))) % (F / 2);
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
        T abs_coefficient = fabs(res);
        if (r_gl < nr_c-1 && c_gl < nc_c-1 && f_gl < nf_c-1) {
          AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
        }
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
            T abs_coefficient = fabs(res);
            if (f_gl < nf_c-1) {
              if (r_gl < nr_c-1 && c_gl < nc_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
              if (r_gl < nr_c-1 && c_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl), abs_coefficient);
              if (r_gl >= 1 && c_gl < nc_c-1)     AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl), abs_coefficient);
              if (r_gl >= 1 && c_gl >= 1)         AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl-1, f_gl), abs_coefficient);
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
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                       ratio_c_sm[c_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wc(r_gl, c_gl, f_gl) = res;
            T abs_coefficient = fabs(res);
            if (c_gl < nc_c-1) {
              if (r_gl < nr_c-1 && f_gl < nf_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
              if (r_gl < nr_c-1 && f_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl-1), abs_coefficient);
              if (r_gl >= 1 && f_gl < nf_c-1)     AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl), abs_coefficient);
              if (r_gl >= 1 && f_gl >= 1)         AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl-1), abs_coefficient);
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
            T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            res = lerp(f1, f2, ratio_c_sm[c_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wcf(r_gl, c_gl, f_gl) = res;
            T abs_coefficient = fabs(res);
            if (c_gl < nc_c-1 && f_gl < nf_c-1) {
              if (r_gl < nr_c-1 ) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
              if (r_gl >= 1)      AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl), abs_coefficient);
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
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm - 1)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm + 1)],
                       ratio_f_sm[f_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wf(r_gl, c_gl, f_gl) = res;
            T abs_coefficient = fabs(res);
            if (f_gl < nf_c-1) {
              if (r_gl < nr_c-1 && c_gl < nc_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
              if (r_gl < nr_c-1 && c_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl), abs_coefficient);
              if (r_gl >= 1 && c_gl < nc_c-1)     AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl), abs_coefficient);
              if (r_gl >= 1 && c_gl >= 1)         AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl-1, f_gl), abs_coefficient);
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
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                       ratio_r_sm[r_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wr(r_gl, c_gl, f_gl) = res;
            // printf("rcf: %u %u %u wr-res: %f\n", r_gl, c_gl, f_gl, res);
            T abs_coefficient = fabs(res);
            if (r_gl < nr_c-1) {
              if (c_gl < nc_c-1 && f_gl < nf_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
              if (c_gl < nc_c-1 && f_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl-1), abs_coefficient);
              if (c_gl >= 1 && f_gl < nf_c-1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl), abs_coefficient);
              if (c_gl >= 1 && f_gl >= 1)         AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl-1), abs_coefficient);
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
            T f1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            T f2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm - 1)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm + 1)],
                        ratio_f_sm[f_sm - 1]);
            res = lerp(f1, f2, ratio_r_sm[r_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wrf(r_gl, c_gl, f_gl) = res;
            T abs_coefficient = fabs(res);
            if (r_gl < nr_c-1 && f_gl < nf_c-1) {
              if (c_gl < nc_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
              if (c_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl), abs_coefficient);
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
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm - 1, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm + 1, f_sm)],
                       ratio_c_sm[c_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wc(r_gl, c_gl, f_gl) = res;
            T abs_coefficient = fabs(res);
            if (c_gl < nc_c-1) {
              if (r_gl < nr_c-1 && f_gl < nf_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
              if (r_gl < nr_c-1 && f_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl-1), abs_coefficient);
              if (r_gl >= 1 && f_gl < nf_c-1)     AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl), abs_coefficient);
              if (r_gl >= 1 && f_gl >= 1)         AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl-1), abs_coefficient);
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
            res = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm, f_sm)],
                       v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm, f_sm)],
                       ratio_r_sm[r_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wr(r_gl, c_gl, f_gl) = res;
            T abs_coefficient = fabs(res);
            if (r_gl < nr_c-1) {
              if (c_gl < nc_c-1 && f_gl < nf_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
              if (c_gl < nc_c-1 && f_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl-1), abs_coefficient);
              if (c_gl >= 1 && f_gl < nf_c-1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl), abs_coefficient);
              if (c_gl >= 1 && f_gl >= 1)         AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl-1), abs_coefficient);
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
            T c1 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm - 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            T c2 = lerp(v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm - 1, f_sm)],
                        v_sm[get_idx(ldsm1, ldsm2, r_sm + 1, c_sm + 1, f_sm)],
                        ratio_c_sm[c_sm - 1]);
            res = lerp(c1, c2, ratio_r_sm[r_sm - 1]);
            res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)] - res;
            *wrc(r_gl, c_gl, f_gl) = res;
            T abs_coefficient = fabs(res);
            if (r_gl < nr_c-1 && c_gl < nc_c-1) {
              if (f_gl < nf_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
              if (f_gl >= 1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl-1), abs_coefficient);
            }
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
          T abs_coefficient = fabs(res);
          if (r_gl < nr_c-1) {
            if (c_gl < nc_c-1 && f_gl < nf_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
            if (c_gl < nc_c-1 && f_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl-1), abs_coefficient);
            if (c_gl >= 1 && f_gl < nf_c-1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl), abs_coefficient);
            if (c_gl >= 1 && f_gl >= 1)         AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl-1), abs_coefficient);
          }
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
          T abs_coefficient = fabs(res);
          if (c_gl < nc_c-1) {
            if (r_gl < nr_c-1 && f_gl < nf_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
            if (r_gl < nr_c-1 && f_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl-1), abs_coefficient);
            if (r_gl >= 1 && f_gl < nf_c-1)     AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl), abs_coefficient);
            if (r_gl >= 1 && f_gl >= 1)         AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl-1), abs_coefficient);
          }
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
          T abs_coefficient = fabs(res);
          if (f_gl < nf_c-1) {
            if (r_gl < nr_c-1 && c_gl < nc_c-1) AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl, f_gl), abs_coefficient);
            if (r_gl < nr_c-1 && c_gl >= 1)     AtomicOp::Max(max_abs_coefficient(level, r_gl, c_gl-1, f_gl), abs_coefficient);
            if (r_gl >= 1 && c_gl < nc_c-1)     AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl, f_gl), abs_coefficient);
            if (r_gl >= 1 && c_gl >= 1)         AtomicOp::Max(max_abs_coefficient(level, r_gl-1, c_gl-1, f_gl), abs_coefficient);
          }
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

  // Collect max abs coefficients from the next finer level
  MGARDX_EXEC void Operation4() {
    // Skip finest level 
    if (level == 0) return;

    if (!max_abs_coefficient.isNull() && threadId < (R / 2) * (C / 2) * (F / 2)) {
      r_sm = (threadId / ((C / 2) * (F / 2))) * 2;
      c_sm = ((threadId % ((C / 2) * (F / 2))) / (F / 2)) * 2;
      f_sm = ((threadId % ((C / 2) * (F / 2))) % (F / 2)) * 2;
      r_gl = r / 2 + threadId / ((C / 2) * (F / 2));
      c_gl = c / 2 + threadId % ((C / 2) * (F / 2)) / (F / 2);
      f_gl = f / 2 + threadId % ((C / 2) * (F / 2)) % (F / 2);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c-1 && c_gl < nc_c-1 && f_gl < nf_c-1) {
        for (int l = 0; l < level; l++) {
          T abs_coefficient = *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2);
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2));
          }
          if (r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2+1));
          }
          *max_abs_coefficient(l, r_gl, c_gl, f_gl) = abs_coefficient;
        }
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

    if (!max_abs_coefficient.isNull() && r + (R / 2) * 2 == nr_p - 1 && threadId >= base &&
        threadId < base + (C / 2) * (F / 2)) {
      r_sm = (R / 2) * 2;
      c_sm = ((threadId - base) / (F / 2)) * 2;
      f_sm = ((threadId - base) % (F / 2)) * 2;
      r_gl = r / 2 + (R / 2);
      c_gl = c / 2 + (threadId - base) / (F / 2);
      f_gl = f / 2 + (threadId - base) % (F / 2);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c-1 && c_gl < nc_c-1 && f_gl < nf_c-1) {
        for (int l = 0; l < level; l++) {
          T abs_coefficient = *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2);
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2));
          }
          if (r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2+1));
          }
          *max_abs_coefficient(l, r_gl, c_gl, f_gl) = abs_coefficient;
        }
      }
    }

    base += (C / 2) * (F / 2); // ROUND_UP_WARP((C/2) * (F/2)) * WARP_SIZE;
    if (!max_abs_coefficient.isNull() && c + (C / 2) * 2 == nc_p - 1 && threadId >= base &&
        threadId < base + (R / 2) * (F / 2)) {
      r_sm = ((threadId - base) / (F / 2)) * 2;
      c_sm = (C / 2) * 2;
      f_sm = ((threadId - base) % (F / 2)) * 2;
      r_gl = r / 2 + (threadId - base) / (F / 2);
      c_gl = c / 2 + (C / 2);
      f_gl = f / 2 + (threadId - base) % (F / 2);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c-1 && c_gl < nc_c-1 && f_gl < nf_c-1) {
        for (int l = 0; l < level; l++) {
          T abs_coefficient = *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2);
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2));
          }
          if (r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2+1));
          }
          *max_abs_coefficient(l, r_gl, c_gl, f_gl) = abs_coefficient;
        }
      }
      // printf("(%d %d %d) (%d %d %d) %f\n",
      //         r_sm, c_sm, f_sm, r_gl, c_gl, f_gl, dwork[get_idx(lddv1, lddv2,
      //         r_gl, c_gl, f_gl)]);
    }

    base += (R / 2) * (F / 2); // ROUND_UP_WARP((R/2) * (F/2)) * WARP_SIZE;
    // printf("%d %d\n", base,  threadId);
    if (!max_abs_coefficient.isNull() && f + (F / 2) * 2 == nf_p - 1 && threadId >= base &&
        threadId < base + (R / 2) * (C / 2)) {
      r_sm = ((threadId - base) / (C / 2)) * 2;
      c_sm = ((threadId - base) % (C / 2)) * 2;
      f_sm = (F / 2) * 2;
      r_gl = r / 2 + (threadId - base) / (C / 2);
      c_gl = c / 2 + (threadId - base) % (C / 2);
      f_gl = f / 2 + (F / 2);
      res = v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm)];
      if (r_sm < rest_r_p && c_sm < rest_c_p && f_sm < rest_f_p &&
          r_gl < nr_c-1 && c_gl < nc_c-1 && f_gl < nf_c-1) {
        for (int l = 0; l < level; l++) {
          T abs_coefficient = *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2);
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2));
          }
          if (r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2+1));
          }
          *max_abs_coefficient(l, r_gl, c_gl, f_gl) = abs_coefficient;
        }
      }
    }

    base += (R / 2) * (C / 2); // ROUND_UP_WARP((R/2) * (C/2)) * WARP_SIZE;
    // load extra edges
    if (!max_abs_coefficient.isNull() && c + (C / 2) * 2 == nc_p - 1 &&
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
          r_gl < nr_c-1 && c_gl < nc_c-1 && f_gl < nf_c-1) {
        for (int l = 0; l < level; l++) {
          T abs_coefficient = *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2);
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2));
          }
          if (r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2+1));
          }
          *max_abs_coefficient(l, r_gl, c_gl, f_gl) = abs_coefficient;
        }
      }
    }

    base += (R / 2); // ROUND_UP_WARP((R/2)) * WARP_SIZE;
    // if (TYPE == 2) printf("%d %d, %d, %llu, %d\n",!w.isNull() == NULL, f +
    // (F/2) * 2, nf_p
    // - 1, threadId, (C/2));
    if (!max_abs_coefficient.isNull() && r + (R / 2) * 2 == nr_p - 1 &&
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
          r_gl < nr_c-1 && c_gl < nc_c-1 && f_gl < nf_c-1) {
        for (int l = 0; l < level; l++) {
          T abs_coefficient = *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2);
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2));
          }
          if (r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2+1));
          }
          *max_abs_coefficient(l, r_gl, c_gl, f_gl) = abs_coefficient;
        }
      }
      // printf("store[%d %d %d]: %f\n", r_sm, c_sm, f_sm, v_sm[get_idx(ldsm1,
      // ldsm2, r_sm, c_sm, f_sm)]);
    }

    base += (C / 2); // ROUND_UP_WARP((C/2)) * WARP_SIZE;
    if (!max_abs_coefficient.isNull() && r + (R / 2) * 2 == nr_p - 1 &&
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
          r_gl < nr_c-1 && c_gl < nc_c-1 && f_gl < nf_c-1) {
        for (int l = 0; l < level; l++) {
          T abs_coefficient = *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2);
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2));
          }
          if (r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2+1));
          }
          *max_abs_coefficient(l, r_gl, c_gl, f_gl) = abs_coefficient;
        }
      }
    }
    base += (F / 2); // ROUND_UP_WARP((F/2)) * WARP_SIZE;
    // // load extra vertex
    if (!max_abs_coefficient.isNull() && r + (R / 2) * 2 == nr_p - 1 &&
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
          r_gl < nr_c-1 && c_gl < nc_c-1 && f_gl < nf_c-1) {
        for (int l = 0; l < level; l++) {
          T abs_coefficient = *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2);
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2));
          }
          if (r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2, c_gl*2+1, f_gl*2+1));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2, f_gl*2+1));
          }
          if (c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2));
          }
          if (f_gl*2+1 < max_abs_coefficient_finer.getShape(0) &&
              c_gl*2+1 < max_abs_coefficient_finer.getShape(1) &&
              r_gl*2+1 < max_abs_coefficient_finer.getShape(2)) {
            abs_coefficient = Math<DeviceType>::Max(abs_coefficient, *max_abs_coefficient_finer(l, r_gl*2+1, c_gl*2+1, f_gl*2+1));
          }
          *max_abs_coefficient(l, r_gl, c_gl, f_gl) = abs_coefficient;
        }
      }
    }
  }

  MGARDX_EXEC void Operation5() {}

private:
  // functor parameters
  SIZE level, nr, nc, nf, nr_c, nc_c, nf_c;
  SubArray<1, T, DeviceType> ratio_r, ratio_c, ratio_f;
  SubArray<D, T, DeviceType> v, w, wf, wc, wr, wcf, wrf, wrc, wrcf;
  SubArray<D+1, T, DeviceType> max_abs_coefficient;
  SubArray<D+1, T, DeviceType> max_abs_coefficient_finer;
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
  SIZE parent_r, parent_c, parent_f;
};

template <DIM D, typename T, typename DeviceType>
class GpkReo3DWithErrorCollection : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  GpkReo3DWithErrorCollection() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<GpkReo3DWithErrorCollectionFunctor<D, T, R, C, F, DeviceType>>
  GenTask(SIZE level, SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
          SubArray<1, T, DeviceType> ratio_r,
          SubArray<1, T, DeviceType> ratio_c,
          SubArray<1, T, DeviceType> ratio_f, SubArray<D, T, DeviceType> v,
          SubArray<D, T, DeviceType> w, SubArray<D, T, DeviceType> wf,
          SubArray<D, T, DeviceType> wc, SubArray<D, T, DeviceType> wr,
          SubArray<D, T, DeviceType> wcf, SubArray<D, T, DeviceType> wrf,
          SubArray<D, T, DeviceType> wrc, SubArray<D, T, DeviceType> wrcf,
          SubArray<D+1, T, DeviceType> max_abs_coefficient,
          SubArray<D+1, T, DeviceType> max_abs_coefficient_finer,
          int queue_idx) {
    using FunctorType = GpkReo3DWithErrorCollectionFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(level, nr, nc, nf, nr_c, nc_c, nf_c, ratio_r, ratio_c, ratio_f,
                        v, w, wf, wc, wr, wcf, wrf, wrc, wrcf, max_abs_coefficient,
                        max_abs_coefficient_finer);

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
                "GpkReo3DWithErrorCollection");
  }

  MGARDX_CONT
  void Execute(SIZE level, SIZE nr, SIZE nc, SIZE nf, SIZE nr_c, SIZE nc_c, SIZE nf_c,
               SubArray<1, T, DeviceType> ratio_r,
               SubArray<1, T, DeviceType> ratio_c,
               SubArray<1, T, DeviceType> ratio_f, SubArray<D, T, DeviceType> v,
               SubArray<D, T, DeviceType> w, SubArray<D, T, DeviceType> wf,
               SubArray<D, T, DeviceType> wc, SubArray<D, T, DeviceType> wr,
               SubArray<D, T, DeviceType> wcf, SubArray<D, T, DeviceType> wrf,
               SubArray<D, T, DeviceType> wrc, SubArray<D, T, DeviceType> wrcf,
               SubArray<D+1, T, DeviceType> max_abs_coefficient,
               SubArray<D+1, T, DeviceType> max_abs_coefficient_finer,
               int queue_idx) {
    int range_l = std::min(6, (int)std::log2(nf) - 1);
    int prec = TypeToIdx<T>();
    int config =
        AutoTuner<DeviceType>::autoTuningTable.gpk_reo_3d[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define GPK(CONFIG)                                                            \
    if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
      const int R = GPK_CONFIG[D - 1][CONFIG][0];                                \
      const int C = GPK_CONFIG[D - 1][CONFIG][1];                                \
      const int F = GPK_CONFIG[D - 1][CONFIG][2];                                \
      using FunctorType = GpkReo3DWithErrorCollectionFunctor<D, T, R, C, F, DeviceType>;\
      using TaskType = Task<FunctorType>;                                        \
      TaskType task = GenTask<R, C, F>(level, nr, nc, nf, nr_c, nc_c, nf_c, ratio_r,    \
                                       ratio_c, ratio_f, v, w, wf, wc, wr, wcf,  \
                                       wrf, wrc, wrcf, max_abs_coefficient, max_abs_coefficient_finer, queue_idx); \
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
      std::cout << log::log_err << "no suitable config for GpkReo3DWithErrorCollection.\n";
      exit(-1);
    }
#undef GPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("gpk_reo_3d", prec, range_l, min_config);
    }
  }
};



} // namespace mgard_x

#endif