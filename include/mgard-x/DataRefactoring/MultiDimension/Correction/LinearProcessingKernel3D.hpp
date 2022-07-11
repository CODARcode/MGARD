/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_LINEAR_PROCESSSING_KERNEL_3D_TEMPLATE
#define MGARD_X_LINEAR_PROCESSSING_KERNEL_3D_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"
#include "LPKFunctor.h"
// #include "LinearProcessingKernel.h"

// #include "../../Functor.h"
// #include "../../AutoTuners/AutoTuner.h"
// #include "../../Task.h"
// #include "../../DeviceAdapters/DeviceAdapter.h"

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class Lpk1Reo3DFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT Lpk1Reo3DFunctor() {}
  MGARDX_CONT Lpk1Reo3DFunctor(SIZE nr, SIZE nc, SIZE nf, SIZE nf_c,
                               SIZE zero_r, SIZE zero_c, SIZE zero_f,
                               SubArray<1, T, DeviceType> ddist_f,
                               SubArray<1, T, DeviceType> dratio_f,
                               SubArray<D, T, DeviceType> dv1,
                               SubArray<D, T, DeviceType> dv2,
                               SubArray<D, T, DeviceType> dw)
      : nr(nr), nc(nc), nf(nf), nf_c(nf_c), zero_r(zero_r), zero_c(zero_c),
        zero_f(zero_f), ddist_f(ddist_f), dratio_f(dratio_f), dv1(dv1),
        dv2(dv2), dw(dw) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    // bool debug = false;
    // if (blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 1 &&
    // threadIdx.y == 0 && threadIdx.z == 0 ) debug = false;

    // bool debug2 = false;
    // if (blockIdx.z == gridDim.z-1 && blockIdx.y == 1 && blockIdx.x == 16)
    // debug2 = false;

    PADDING = (nf % 2 == 0);

    T *sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F * 2 + 3;
    ldsm2 = C;
    v_sm = sm;
    dist_f_sm = sm + ldsm1 * ldsm2 * R;
    ratio_f_sm = dist_f_sm + ldsm1;

    // if (FunctorBase<DeviceType>::GetBlockIdZ() == 0 &&
    // FunctorBase<DeviceType>::GetBlockIdY() == 0 &&
    // FunctorBase<DeviceType>::GetBlockIdX() == 0 &&
    // FunctorBase<DeviceType>::GetThreadIdZ() == 1 &&
    // FunctorBase<DeviceType>::GetThreadIdY() == 0 ) debug = true;

    r_gl = FunctorBase<DeviceType>::GetBlockIdZ() *
               FunctorBase<DeviceType>::GetBlockDimZ() +
           FunctorBase<DeviceType>::GetThreadIdZ();
    c_gl = FunctorBase<DeviceType>::GetBlockIdY() *
               FunctorBase<DeviceType>::GetBlockDimY() +
           FunctorBase<DeviceType>::GetThreadIdY();
    f_gl = FunctorBase<DeviceType>::GetBlockIdX() *
               FunctorBase<DeviceType>::GetBlockDimX() +
           FunctorBase<DeviceType>::GetThreadIdX();

    blockId = FunctorBase<DeviceType>::GetBlockIdX();

    r_sm = FunctorBase<DeviceType>::GetThreadIdZ();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();

    actual_F = F;
    if (nf_c - blockId * FunctorBase<DeviceType>::GetBlockDimX() < F) {
      actual_F = nf_c - blockId * FunctorBase<DeviceType>::GetBlockDimX();
    }

    // if (nf_c % 2 == 1){
    //   if(nf_c-1 - blockId * blockDim.x < F) { actual_F = nf_c - 1 - blockId *
    //   blockDim.x; }
    // } else {
    //   if(nf_c - blockId * blockDim.x < F) { actual_F = nf_c - blockId *
    //   blockDim.x; }
    // }

    // if (debug) printf("actual_F %d\n", actual_F);

    if (r_gl < nr && c_gl < nc && f_gl < nf_c) {
      if (r_gl < zero_r && c_gl < zero_c && f_gl < zero_f) {
        // if (debug) printf("load left vsm[%d]: 0.0\n", f_sm * 2 + 2);
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 2)] = 0.0;
      } else {
        // if (debug) printf("load left vsm[%d]<-dv1[%d, %d, %d]: %f\n", f_sm *
        // 2
        // + 2, r_gl, c_gl, f_gl, dv1[get_idx(lddv11, lddv12, r_gl, c_gl,
        // f_gl)]);
        v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 2)] =
            *dv1(r_gl, c_gl, f_gl);
      }

      if (f_sm == actual_F - 1) {
        if (r_gl < zero_r && c_gl < zero_c && f_gl < zero_f) {
          // if (debug) printf("load left+1 vsm[%d]: 0.0\n", actual_F * 2 + 2);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 2)] = 0.0;
        } else {
          if (f_gl + 1 < nf_c) {
            // if (debug) printf("load left+1 vsm[%d]: %f\n", actual_F * 2 + 2,
            // dv1[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl + 1)]);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 2)] =
                *dv1(r_gl, c_gl, f_gl + 1);
          } else {
            // if (debug) printf("load left+1 vsm[%d]: 0.0\n", actual_F * 2 +
            // 2);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 2)] = 0.0;
          }
        }
      }

      if (f_sm == 0) {
        // left
        if (r_gl < zero_r && c_gl < zero_c && f_gl < zero_f) {
          // coarse (-1)
          // if (debug) printf("load left-1 vsm[0]: 0.0\n");
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 0.0;
        } else {
          if (f_gl >= 1) {
            // other (-1)
            // if (debug) printf("load left-1 vsm[0]: %f\n", dv1[get_idx(lddv11,
            // lddv12, r_gl, c_gl, f_gl-1)]);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
                *dv1(r_gl, c_gl, f_gl - 1);
          } else {
            // other (-1)
            // if (debug) printf("load left-1 vsm[0]: 0.0\n");
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] = 0.0;
          }
        }
      }

      // right
      if (!PADDING) {
        if (nf_c % 2 != 0) {
          if (f_gl >= 1 && f_gl < nf_c) {
            // if (debug) printf("load right vsm[%d]: %f <- %d %d %d\n", f_sm *
            // 2
            // + 1, dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl - 1)], r_gl,
            // c_gl, f_gl - 1);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)] =
                *dv2(r_gl, c_gl, f_gl - 1);
          } else {
            // if (debug) printf("load right vsm[%d]: 0\n", f_sm * 2 + 1);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)] = 0.0;
          }
        } else { // nf_c % 2 == 0
          if (f_gl < nf_c - 1) {
            // if (debug) printf("load right vsm[%d]: %f <- %d %d %d\n", f_sm *
            // 2
            // + 3, dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl)], r_gl, c_gl,
            // f_gl);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 3)] =
                *dv2(r_gl, c_gl, f_gl);
          } else {
            // if (debug) printf("load right vsm[%d]: 0\n", f_sm * 2 + 3);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 3)] = 0.0;
          }
        }
      } else { // PADDING
        if (nf_c % 2 != 0) {
          if (f_gl >= 1 && f_gl < nf_c - 1) {
            // if (debug) printf("load right vsm[%d]: %f <- %d %d %d\n", f_sm *
            // 2
            // + 1, dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl - 1)], r_gl,
            // c_gl, f_gl - 1);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)] =
                *dv2(r_gl, c_gl, f_gl - 1);
          } else {
            // if (debug) printf("load right vsm[%d]: 0\n", f_sm * 2 + 1);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 1)] = 0.0;
          }
        } else { // nf_c % 2 == 0
          if (f_gl < nf_c - 2) {
            // if (debug) printf("load right vsm[%d]: %f <- %d %d %d\n", f_sm *
            // 2
            // + 3, dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl)], r_gl, c_gl,
            // f_gl);
            v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, f_sm * 2 + 3)] =
                *dv2(r_gl, c_gl, f_gl);
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
              // actual_F * 2 + 1, dv2[get_idx(lddv21, lddv22, r_gl, c_gl,
              // f_gl)], r_gl, c_gl, f_gl);
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 1)] =
                  *dv2(r_gl, c_gl, f_gl);
            } else {
              // if (debug) printf("load right+1 vsm[%d]: 0.0\n", actual_F * 2 +
              // 1);
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 1)] = 0.0;
            }
          } else { // nf_c % 2 == 0
            if (f_gl >= actual_F) {
              // if (debug) printf("load right-1 vsm[1]: %f <- %d %d %d\n",
              // dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl - actual_F)],
              // r_gl, c_gl, f_gl - actual_F);
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 1)] =
                  *dv2(r_gl, c_gl, f_gl - actual_F);
            } else {
              // if (debug) printf("load right-1 vsm[1]: 0.0\n");
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 1)] = 0.0;
            }
          }
        } else {
          if (nf_c % 2 != 0) {
            if (f_gl < nf_c - 2) {
              // if (debug) printf("actual_F(%d), load right+1 vsm[%d]: %f <- %d
              // %d %d\n", actual_F, actual_F * 2 + 1, dv2[get_idx(lddv21,
              // lddv22, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 1)] =
                  *dv2(r_gl, c_gl, f_gl);
            } else {
              // if (debug) printf("load right+1 vsm[%d]: 0.0\n", actual_F * 2 +
              // 1);
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, actual_F * 2 + 1)] = 0.0;
            }
          } else { // nf_c % 2 == 0
            if (f_gl >= actual_F && f_gl - actual_F < nf_c - 2) {
              // if (debug) printf("load right-1 vsm[1]: %f <- %d %d %d\n",
              // dv2[get_idx(lddv21, lddv22, r_gl, c_gl, f_gl - actual_F)],
              // r_gl, c_gl, f_gl - actual_F);
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 1)] =
                  *dv2(r_gl, c_gl, f_gl - actual_F);
            } else {
              // if (debug) printf("load right-1 vsm[1]: 0.0\n");
              v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 1)] = 0.0;
            }
          }
        }
      }
    }

    // debug = false;
    // if (r_gl == 0 && c_gl == 0) debug = true;

    if (r_sm == 0 && c_sm == 0 && f_sm < actual_F) {
      if (blockId * F * 2 + f_sm < nf) {
        dist_f_sm[2 + f_sm] = *ddist_f(blockId * F * 2 + f_sm);
        ratio_f_sm[2 + f_sm] = *dratio_f(blockId * F * 2 + f_sm);
        // if (debug) printf("load dist[%d] -> sm[%d]: %f\n", blockId * F * 2 +
        // f_sm, 2 + f_sm, *ddist_f(blockId * F * 2 + f_sm));
      } else {
        dist_f_sm[2 + f_sm] = 0.0;
        ratio_f_sm[2 + f_sm] = 0.0;
      }

      if (blockId * F * 2 + actual_F + f_sm < nf) {
        dist_f_sm[2 + actual_F + f_sm] =
            *ddist_f(blockId * F * 2 + actual_F + f_sm);
        ratio_f_sm[2 + actual_F + f_sm] =
            *dratio_f(blockId * F * 2 + actual_F + f_sm);
        // if (debug) printf("load dist[%d] -> sm[%d]: %f\n", blockId * F * 2 +
        // actual_F + f_sm, 2 + actual_F + f_sm, *ddist_f(blockId * F * 2 +
        // actual_F + f_sm));
      } else {
        dist_f_sm[2 + actual_F + f_sm] = 0.0;
        ratio_f_sm[2 + actual_F + f_sm] = 0.0;
      }
      // dist_f_sm[2 + f_sm] = ddist_f[f_gl];
      // dist_f_sm[2 + actual_F + f_sm] = ddist_f[actual_F + f_gl];
      // ratio_f_sm[2 + f_sm] = dratio_f[f_gl];
      // ratio_f_sm[2 + actual_F + f_sm] = dratio_f[actual_F + f_gl];
    }

    if (blockId > 0) {
      if (f_sm < 2) {
        // dist_f_sm[f_sm] = ddist_f[f_gl - 2];
        // ratio_f_sm[f_sm] = dratio_f[f_gl - 2];
        dist_f_sm[f_sm] = *ddist_f(blockId * F * 2 + f_sm - 2);
        ratio_f_sm[f_sm] = *dratio_f(blockId * F * 2 + f_sm - 2);
      }
    } else {
      if (f_sm < 2) {
        dist_f_sm[f_sm] = 0.0;
        ratio_f_sm[f_sm] = 0.0;
      }
    }
  }

  MGARDX_EXEC void Operation2() {
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

      // if (f_gl == 0 && c_gl == 2 && r_gl == 0) {
      //   printf("f_sm(%d) %f %f %f %f %f\n",f_sm, a,b,c,d,e);
      //   printf("f_sm_h(%d) %f %f %f %f\n",f_sm, h1,h2,h3,h4);
      //   printf("f_sm_r(%d) %f %f %f %f\n",f_sm, r1,r2,r3,r4);
      // }

      // T tb = a * h1/6 + b * (h1+h2)/3 + c * h2/6;
      // T tc = b * h2/6 + c * (h2+h3)/3 + d * h3/6;
      // T td = c * h3/6 + d * (h3+h4)/3 + e * h4/6;

      // if (f_gl == 0 && c_gl == 2 && r_gl == 0) {
      //   printf("tb tc td %f %f %f\n", tb, tc, td);
      // }

      // if (debug) printf("f_sm(%d) tb tc td tc: %f %f %f %f\n", f_sm, tb, tc,
      // td, tc+tb * r1 + td * r4);

      // tc += tb * r1 + td * r4;

      // if (f_gl == 0 && c_gl == 2 && r_gl == 0) {
      //   printf("tc %f\n", tc, td);
      // }

      *dw(r_gl, c_gl, f_gl) =
          mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4);

      // if (f_gl == 0 && c_gl == 2 && r_gl == 0) {
      //   // if (debug)
      //     printf("store[%d %d %d] %f \n", r_gl, c_gl, f_gl,
      //           mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));
      // }

      // printf("test block %d F %d nf %d\n", blockId, F, nf);
      // if (f_gl+1 == nf_c-1) {

      //     // T te = h4 * d + 2 * h4 * e;
      //     //printf("f_sm(%d) mm-e: %f\n", f_sm, te);
      //     // te += td * r3;
      //     dw[get_idx(lddw1, lddw2, r_gl, c_gl, f_gl+1)] =
      //       mass_trans(c, d, e, (T)0.0, (T)0.0, h1, h2, (T)0.0, (T)0.0, r1,
      //       r2, (T)0.0, (T)0.0);
      // }
    }
  }

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size = (R * C * (F * 2 + 3) + (F * 2 + 3) * 2) * sizeof(T);
    return size;
  }

private:
  // functor parameters
  SIZE nr, nc, nf, nf_c, zero_r, zero_c, zero_f;
  SubArray<1, T, DeviceType> ddist_f;
  SubArray<1, T, DeviceType> dratio_f;
  SubArray<D, T, DeviceType> dv1, dv2, dw;

  // thread local variables
  bool PADDING;
  SIZE ldsm1;
  SIZE ldsm2;
  T *v_sm;
  T *dist_f_sm;
  T *ratio_f_sm;
  SIZE r_gl, c_gl, f_gl;
  SIZE blockId;
  SIZE r_sm, c_sm, f_sm;
  SIZE actual_F;
  bool debug = false;
};

template <DIM D, typename T, typename DeviceType>
class Lpk1Reo3D : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  Lpk1Reo3D() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Lpk1Reo3DFunctor<D, T, R, C, F, DeviceType>>
  GenTask(SIZE nr, SIZE nc, SIZE nf, SIZE nf_c, SIZE zero_r, SIZE zero_c,
          SIZE zero_f, SubArray<1, T, DeviceType> ddist_f,
          SubArray<1, T, DeviceType> dratio_f, SubArray<D, T, DeviceType> dv1,
          SubArray<D, T, DeviceType> dv2, SubArray<D, T, DeviceType> dw,
          int queue_idx) {
    using FunctorType = Lpk1Reo3DFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(nr, nc, nf, nf_c, zero_r, zero_c, zero_f, ddist_f,
                        dratio_f, dv1, dv2, dw);

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
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "Lpk1Reo3D");
  }

  MGARDX_CONT
  void Execute(SIZE nr, SIZE nc, SIZE nf, SIZE nf_c, SIZE zero_r, SIZE zero_c,
               SIZE zero_f, SubArray<1, T, DeviceType> ddist_f,
               SubArray<1, T, DeviceType> dratio_f,
               SubArray<D, T, DeviceType> dv1, SubArray<D, T, DeviceType> dv2,
               SubArray<D, T, DeviceType> dw, int queue_idx) {
    int range_l = std::min(6, (int)std::log2(nf) - 1);
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.lpk1_3d[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define LPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = LPK_CONFIG[D - 1][CONFIG][0];                                \
    const int C = LPK_CONFIG[D - 1][CONFIG][1];                                \
    const int F = LPK_CONFIG[D - 1][CONFIG][2];                                \
    using FunctorType = Lpk1Reo3DFunctor<D, T, R, C, F, DeviceType>;           \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task =                                                            \
        GenTask<R, C, F>(nr, nc, nf, nf_c, zero_r, zero_c, zero_f, ddist_f,    \
                         dratio_f, dv1, dv2, dw, queue_idx);                   \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    LPK(6) if (!ret.success) config--;
    LPK(5) if (!ret.success) config--;
    LPK(4) if (!ret.success) config--;
    LPK(3) if (!ret.success) config--;
    LPK(2) if (!ret.success) config--;
    LPK(1) if (!ret.success) config--;
    LPK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err << "no suitable config for Lpk1Reo3D.\n";
      exit(-1);
    }
#undef LPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("lpk1_3d", prec, range_l, min_config);
    }
  }
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class Lpk2Reo3DFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT Lpk2Reo3DFunctor() {}
  MGARDX_CONT Lpk2Reo3DFunctor(SIZE nr, SIZE nc, SIZE nf_c, SIZE nc_c,
                               SubArray<1, T, DeviceType> ddist_c,
                               SubArray<1, T, DeviceType> dratio_c,
                               SubArray<D, T, DeviceType> dv1,
                               SubArray<D, T, DeviceType> dv2,
                               SubArray<D, T, DeviceType> dw)
      : nr(nr), nc(nc), nf_c(nf_c), nc_c(nc_c), ddist_c(ddist_c),
        dratio_c(dratio_c), dv1(dv1), dv2(dv2), dw(dw) {
    Functor<DeviceType>();
  }
  MGARDX_EXEC void Operation1() {
    // bool debug = false;
    // if (blockIdx.y == gridDim.y-1 && blockIdx.x == 0 &&
    // threadIdx.x == 0 ) debug = false;

    // bool debug2 = false;
    // if (blockIdx.z == gridDim.z-1 && blockIdx.y == 1 && blockIdx.x == 16)
    // debug2 = false;

    PADDING = (nc % 2 == 0);

    T *sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F;
    ldsm2 = C * 2 + 3;
    v_sm = sm;
    dist_c_sm = sm + ldsm1 * ldsm2 * R;
    ratio_c_sm = dist_c_sm + ldsm2;

    // bool debug = false;
    // if (blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 &&
    // threadIdx.z == 0 && threadIdx.x == 0 ) debug = false;

    r_gl = FunctorBase<DeviceType>::GetBlockIdZ() *
               FunctorBase<DeviceType>::GetBlockDimZ() +
           FunctorBase<DeviceType>::GetThreadIdZ();
    c_gl = FunctorBase<DeviceType>::GetBlockIdY() *
               FunctorBase<DeviceType>::GetBlockDimY() +
           FunctorBase<DeviceType>::GetThreadIdY();
    f_gl = FunctorBase<DeviceType>::GetBlockIdX() *
               FunctorBase<DeviceType>::GetBlockDimX() +
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
    //   if(nc_c-1 - blockIdx.y * blockDim.y < C) { actual_C = nc_c - 1 -
    //   blockIdx.y * blockDim.y; }
    // } else {
    //   if(nc_c - blockIdx.y * blockDim.y < C) { actual_C = nc_c - blockIdx.y *
    //   blockDim.y; }
    // }

    // if (debug) printf("actual_C %d\n", actual_C);

    if (r_gl < nr && c_gl < nc_c && f_gl < nf_c) {
      // if (debug) printf("load up vsm[%d]: %f <- %d %d %d\n", c_sm * 2 + 2,
      // dv1[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
      v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 2, f_sm)] =
          *dv1(r_gl, c_gl, f_gl);

      if (c_sm == actual_C - 1) {
        if (c_gl + 1 < nc_c) {
          // if (debug) printf("load up+1 vsm[%d]: %f <- %d %d %d\n", actual_C *
          // 2
          // + 2, dv1[get_idx(lddv11, lddv12, r_gl, blockId * C + actual_C,
          // f_gl)], r_gl, blockId * C + actual_C, f_gl);
          // c_gl+1 == blockId * C + C
          v_sm[get_idx(ldsm1, ldsm2, r_sm, actual_C * 2 + 2, f_sm)] =
              *dv1(r_gl, c_gl + 1, f_gl);
        } else {
          // if (debug) printf("load up+1 vsm[%d]: 0.0\n", actual_C * 2 + 2);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, actual_C * 2 + 2, f_sm)] = 0.0;
        }
      }

      if (c_sm == 0) {
        if (c_gl >= 1) {
          // if (debug) printf("load up-1 vsm[0]: %f <- %d %d %d\n",
          // dv1[get_idx(lddv11, lddv12, r_gl, c_gl-1, f_gl)], r_gl, c_gl-1,
          // f_gl);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
              *dv1(r_gl, c_gl - 1, f_gl);
        } else {
          // if (debug) printf("load up-1 vsm[0]: 0.0\n");
          v_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] = 0.0;
        }
      }

      if (!PADDING) {
        if (c_gl < nc_c - 1) {
          // if (debug) printf("load down vsm[%d]: %f <- %d %d %d\n", c_sm * 2 +
          // 3, dv2[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)], r_gl, c_gl,
          // f_gl);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)] =
              *dv2(r_gl, c_gl, f_gl);
        } else {
          // if (debug) printf("load down vsm[%d]: 0.0\n", c_sm * 2 + 3);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)] = 0.0;
        }
      } else {
        if (c_gl < nc_c - 2) {
          // if (debug) printf("load down vsm[%d]: %f <- %d %d %d\n", c_sm * 2 +
          // 3, dv2[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)], r_gl, c_gl,
          // f_gl);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)] =
              *dv2(r_gl, c_gl, f_gl);
        } else {
          // if (debug) printf("load down vsm[%d]: 0.0\n", c_sm * 2 + 3);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm * 2 + 3, f_sm)] = 0.0;
        }
      }

      if (c_gl >= 1 &&
          (PADDING && c_gl - 1 < nc_c - 2 || !PADDING && c_gl - 1 < nc_c - 1)) {
        if (c_sm == 0) {
          // if (debug) printf("PADDING: %d, c_gl-1: %d nc_c-2: %d\n", PADDING,
          // c_gl-1, nc_c - 2); if (debug) printf("load down-1 vsm[1]: %f <- %d
          // %d %d\n", dv2[get_idx(lddv11, lddv12, r_gl, c_gl-1, f_gl)], r_gl,
          // c_gl-1, f_gl);
          v_sm[get_idx(ldsm1, ldsm2, r_sm, 1, f_sm)] =
              *dv2(r_gl, c_gl - 1, f_gl);
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
        dist_c_sm[2 + f_sm] = *ddist_c(blockId * C * 2 + f_sm);
        ratio_c_sm[2 + f_sm] = *dratio_c(blockId * C * 2 + f_sm);
      } else {
        dist_c_sm[2 + f_sm] = 0.0;
        ratio_c_sm[2 + f_sm] = 0.0;
      }

      if (blockId * C * 2 + actual_C + f_sm < nc) {
        dist_c_sm[2 + actual_C + f_sm] =
            *ddist_c(blockId * C * 2 + actual_C + f_sm);
        ratio_c_sm[2 + actual_C + f_sm] =
            *dratio_c(blockId * C * 2 + actual_C + f_sm);
      } else {
        dist_c_sm[2 + actual_C + f_sm] = 0.0;
        ratio_c_sm[2 + actual_C + f_sm] = 0.0;
      }
    }

    if (blockId > 0) {
      if (f_sm < 2) {
        dist_c_sm[f_sm] = *ddist_c(blockId * C * 2 - 2 + f_sm);
        ratio_c_sm[f_sm] = *dratio_c(blockId * C * 2 - 2 + f_sm);
      }
    } else {
      if (f_sm < 2) {
        dist_c_sm[f_sm] = 0.0;
        ratio_c_sm[f_sm] = 0.0;
      }
    }
  }

  MGARDX_EXEC void Operation2() {
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

      *dw(r_gl, c_gl, f_gl) =
          mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4);

      // if (debug) printf("store[%d %d %d] %f \n", r_gl, c_gl, f_gl,
      //           mass_trans(a, b, c, d, e, h1, h2, h3, h4, r1, r2, r3, r4));

      // printf("%d %d %d\n", r_gl, c_gl, f_gl);
      // if (blockId * C + C == nc-1) {
      // if (c_gl + 1 == nc_c - 1) {
      //   // T te = h4 * d + 2 * h4 * e;
      //   // te += td * r3;
      //   dw[get_idx(lddw1, lddw2, r_gl, blockId * C + actual_C, f_gl)] =
      //     mass_trans(c, d, e, (T)0.0, (T)0.0,
      //       h1, h2, (T)0.0, (T)0.0, r1, r2, (T)0.0, (T)0.0);
      // }
      // }
    }
  }

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size = (R * (C * 2 + 3) * F + (C * 2 + 3) * 2) * sizeof(T);
    return size;
  }

private:
  // functor parameters
  SIZE nr, nc, nf_c, nc_c;
  SubArray<1, T, DeviceType> ddist_c;
  SubArray<1, T, DeviceType> dratio_c;
  SubArray<D, T, DeviceType> dv1, dv2, dw;

  // thread local variables
  bool PADDING;
  SIZE ldsm1;
  SIZE ldsm2;
  T *v_sm;
  T *dist_c_sm;
  T *ratio_c_sm;
  SIZE r_gl, c_gl, f_gl;
  SIZE blockId;
  SIZE r_sm, c_sm, f_sm;
  SIZE actual_C;
  bool debug;
};

template <DIM D, typename T, typename DeviceType>
class Lpk2Reo3D : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  Lpk2Reo3D() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Lpk2Reo3DFunctor<D, T, R, C, F, DeviceType>>
  GenTask(SIZE nr, SIZE nc, SIZE nf_c, SIZE nc_c,
          SubArray<1, T, DeviceType> ddist_c,
          SubArray<1, T, DeviceType> dratio_c, SubArray<D, T, DeviceType> dv1,
          SubArray<D, T, DeviceType> dv2, SubArray<D, T, DeviceType> dw,
          int queue_idx) {
    using FunctorType = Lpk2Reo3DFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(nr, nc, nf_c, nc_c, ddist_c, dratio_c, dv1, dv2, dw);

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
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "Lpk2Reo3D");
  }

  MGARDX_CONT
  void Execute(SIZE nr, SIZE nc, SIZE nf_c, SIZE nc_c,
               SubArray<1, T, DeviceType> ddist_c,
               SubArray<1, T, DeviceType> dratio_c,
               SubArray<D, T, DeviceType> dv1, SubArray<D, T, DeviceType> dv2,
               SubArray<D, T, DeviceType> dw, int queue_idx) {
    int range_l = std::min(6, (int)std::log2(nf_c) - 1);
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.lpk2_3d[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define LPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = LPK_CONFIG[D - 1][CONFIG][0];                                \
    const int C = LPK_CONFIG[D - 1][CONFIG][1];                                \
    const int F = LPK_CONFIG[D - 1][CONFIG][2];                                \
    using FunctorType = Lpk2Reo3DFunctor<D, T, R, C, F, DeviceType>;           \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<R, C, F>(nr, nc, nf_c, nc_c, ddist_c, dratio_c,    \
                                     dv1, dv2, dw, queue_idx);                 \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    LPK(6) if (!ret.success) config--;
    LPK(5) if (!ret.success) config--;
    LPK(4) if (!ret.success) config--;
    LPK(3) if (!ret.success) config--;
    LPK(2) if (!ret.success) config--;
    LPK(1) if (!ret.success) config--;
    LPK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err << "no suitable config for Lpk2Reo3D.\n";
      exit(-1);
    }
#undef LPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("lpk2_3d", prec, range_l, min_config);
    }
  }
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, typename DeviceType>
class Lpk3Reo3DFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT Lpk3Reo3DFunctor() {}
  MGARDX_CONT Lpk3Reo3DFunctor(SIZE nr, SIZE nc_c, SIZE nf_c, SIZE nr_c,
                               SubArray<1, T, DeviceType> ddist_r,
                               SubArray<1, T, DeviceType> dratio_r,
                               SubArray<D, T, DeviceType> dv1,
                               SubArray<D, T, DeviceType> dv2,
                               SubArray<D, T, DeviceType> dw)
      : nr(nr), nc_c(nc_c), nf_c(nf_c), nr_c(nr_c), ddist_r(ddist_r),
        dratio_r(dratio_r), dv1(dv1), dv2(dv2), dw(dw) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    // bool debug = false;
    // if (blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 &&
    // threadIdx.y == 0 && threadIdx.x == 0 ) debug = true;

    // bool debug2 = false;
    // if (blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0)
    // debug2 = true;

    PADDING = (nr % 2 == 0);
    T *sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F;
    ldsm2 = C;
    v_sm = sm;
    dist_r_sm = sm + ldsm1 * ldsm2 * (R * 2 + 3);
    ratio_r_sm = dist_r_sm + (R * 2 + 3);

    r_gl = FunctorBase<DeviceType>::GetBlockIdZ() *
               FunctorBase<DeviceType>::GetBlockDimZ() +
           FunctorBase<DeviceType>::GetThreadIdZ();
    c_gl = FunctorBase<DeviceType>::GetBlockIdY() *
               FunctorBase<DeviceType>::GetBlockDimY() +
           FunctorBase<DeviceType>::GetThreadIdY();
    f_gl = FunctorBase<DeviceType>::GetBlockIdX() *
               FunctorBase<DeviceType>::GetBlockDimX() +
           FunctorBase<DeviceType>::GetThreadIdX();

    // if (debug) printf("debugging gl: %d %d %d\n", r_gl, c_gl, f_gl);

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
    //   if(nr_c-1 - blockIdx.z * blockDim.z < R) { actual_R = nr_c - 1 -
    //   blockIdx.z * blockDim.z; }
    // } else {
    //   if(nr_c - blockIdx.z * blockDim.z < R) { actual_R = nr_c - blockIdx.z *
    //   blockDim.z; }
    // }

    // if (debug) printf("actual_R %d\n", actual_R);

    // if (debug) printf("RCF: %d %d %d\n", R, C, F);
    if (r_gl < nr_c && c_gl < nc_c && f_gl < nf_c) {
      // if (debug) printf("load front vsm[%d]: %f <- %d %d %d\n", r_sm * 2 + 2,
      // dv1[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)], r_gl, c_gl, f_gl);
      v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 2, c_sm, f_sm)] =
          *dv1(r_gl, c_gl, f_gl);

      if (r_sm == actual_R - 1) {
        if (r_gl + 1 < nr_c) {
          // if (debug) printf("load front+1 vsm[%d]: %f <- %d %d %d\n",
          // actual_R
          // * 2 + 2, dv1[get_idx(lddv11, lddv12, blockId * R + actual_R, c_gl,
          // f_gl)], blockId * R + actual_R, c_gl, f_gl);
          v_sm[get_idx(ldsm1, ldsm2, actual_R * 2 + 2, c_sm, f_sm)] =
              *dv1(r_gl + 1, c_gl, f_gl);
        } else {
          // if (debug) printf("load front+1 vsm[%d]: 0.0\n", actual_R * 2 + 2);
          v_sm[get_idx(ldsm1, ldsm2, actual_R * 2 + 2, c_sm, f_sm)] = 0.0;
        }
      }

      if (r_sm == 0) {
        if (r_gl >= 1) {
          // if (debug) printf("load front-1 vsm[0]: %f <- %d %d %d\n",
          // dv1[get_idx(lddv11, lddv12, r_gl-1, c_gl, f_gl)], r_gl-1, c_gl,
          // f_gl);
          v_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
              *dv1(r_gl - 1, c_gl, f_gl);
        } else {
          // if (debug) printf("load front-1 vsm[0]: 0.0\n");
          v_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] = 0.0;
        }
      }

      if (!PADDING) {
        if (r_gl < nr_c - 1) {
          // if (debug) printf("load back vsm[%d]: %f <- %d %d %d\n", r_sm * 2 +
          // 3, dv2[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)], r_gl, c_gl,
          // f_gl);
          v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)] =
              *dv2(r_gl, c_gl, f_gl);
        } else {
          // if (debug) printf("load back vsm[%d]: 0.0\n", r_sm * 2 + 3);
          v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)] = 0.0;
        }
      } else {
        if (r_gl < nr_c - 2) {
          // if (debug) printf("load back vsm[%d]: %f <- %d %d %d\n", r_sm * 2 +
          // 3, dv2[get_idx(lddv11, lddv12, r_gl, c_gl, f_gl)], r_gl, c_gl,
          // f_gl);
          v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)] =
              *dv2(r_gl, c_gl, f_gl);
        } else {
          // if (debug) printf("load back vsm[%d]: 0.0\n", r_sm * 2 + 3);
          v_sm[get_idx(ldsm1, ldsm2, r_sm * 2 + 3, c_sm, f_sm)] = 0.0;
        }
      }

      if (r_gl >= 1 &&
          (PADDING && r_gl - 1 < nr_c - 2 || !PADDING && r_gl < nr_c)) {
        // if (blockId > 0) {
        if (r_sm == 0) {
          // if (debug) printf("load back-1 vsm[1]: %f <- %d %d %d\n",
          // dv2[get_idx(lddv11, lddv12, r_gl-1, c_gl, f_gl)], r_gl-1, c_gl,
          // f_gl);
          v_sm[get_idx(ldsm1, ldsm2, 1, c_sm, f_sm)] =
              *dv2(r_gl - 1, c_gl, f_gl);
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
    // if (debug2) printf("actual_R: %u\n", actual_R);
    if (r_sm == 0 && c_sm == 0 && f_sm < actual_R) {
      // if (debug2) printf(" RCF (%u %u %u)blockid(%u) fsm(%u) nr(%u)\n", R, C,
      // F, blockId, blockId * R * 2 + f_sm, nr);
      if (blockId * R * 2 + f_sm < nr) {

        dist_r_sm[2 + f_sm] = *ddist_r(blockId * R * 2 + f_sm);
        // if (debug2 ) printf("load dist 1 [%d]: %f [%d]\n", 2 + f_sm,
        // dist_r_sm[2 + f_sm], blockId * R * 2 + f_sm);
        ratio_r_sm[2 + f_sm] = *dratio_r(blockId * R * 2 + f_sm);
        // if (debug2 )printf("load ratio 1 [%d]: %f [%d]\n", 2 + f_sm,
        // ratio_r_sm[2 + f_sm], blockId * R * 2 + f_sm);
      } else {
        dist_r_sm[2 + f_sm] = 0.0;
        ratio_r_sm[2 + f_sm] = 0.0;
      }
      if (blockId * R * 2 + actual_R + f_sm < nr) {
        dist_r_sm[2 + actual_R + f_sm] =
            *ddist_r(blockId * R * 2 + actual_R + f_sm);
        // if (debug2 )printf("load dist 2 [%d]: %f [%d]\n", 2 + actual_R +
        // f_sm, dist_r_sm[2 + actual_R + f_sm], blockId * R * 2 + actual_R +
        // f_sm);
        ratio_r_sm[2 + actual_R + f_sm] =
            *dratio_r(blockId * R * 2 + actual_R + f_sm);
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
        dist_r_sm[f_sm] = *ddist_r(blockId * R * 2 - 2 + f_sm);
        // if (debug2 )printf("load dist -1 [%d]: %f [%d]\n", f_sm,
        // dist_r_sm[f_sm], blockId * R * 2 - 2 + f_sm);
        ratio_r_sm[f_sm] = *dratio_r(blockId * R * 2 - 2 + f_sm);
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

  MGARDX_EXEC void Operation2() {
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

      // T tb = a * h1/6 + b * 2 * (h1+h2)/6 + c * h2/6;
      // T tc = b * h2/6 + c * 2 * (h2+h3)/6 + d * h3/6;
      // T td = c * h3/6 + d * 2 * (h3+h4)/6 + e * h4/6;

      // if (debug) printf("f_sm(%d) tb tc td tc: %f %f %f %f\n", f_sm, tb, tc,
      // td, tc+tb * r1 + td * r4);

      // tc += tb * r1 + td * r4;

      *dw(r_gl, c_gl, f_gl) =
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
      //   dw[get_idx(lddw1, lddw2, blockId * R + actual_R, c_gl, f_gl)] =
      //     mass_trans(c, d, e, (T)0.0, (T)0.0,
      //       h1, h2, (T)0.0, (T)0.0, r1, r2, (T)0.0, (T)0.0);

      //   if (debug) printf("store-last[%d %d %d] %f\n", blockId * R +
      //   actual_R, c_gl, f_gl,
      //             mass_trans(c, d, e, (T)0.0, (T)0.0,
      //       h1, h2, (T)0.0, (T)0.0, r1, r2, (T)0.0, (T)0.0));
      // }
      //}
    }
  }

  MGARDX_EXEC void Operation3() {}

  MGARDX_EXEC void Operation4() {}

  MGARDX_EXEC void Operation5() {}

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size = ((R * 2 + 3) * C * F + (R * 2 + 3) * 2) * sizeof(T);
    return size;
  }

private:
  // functor parameters
  SIZE nr, nc_c, nf_c, nr_c;
  SubArray<1, T, DeviceType> ddist_r;
  SubArray<1, T, DeviceType> dratio_r;
  SubArray<D, T, DeviceType> dv1, dv2, dw;

  // thread local variables
  bool PADDING;
  SIZE ldsm1;
  SIZE ldsm2;
  T *v_sm;
  T *dist_r_sm;
  T *ratio_r_sm;
  SIZE r_gl, c_gl, f_gl;
  SIZE blockId;
  SIZE r_sm, c_sm, f_sm;
  SIZE actual_R;
  bool debug;
};

template <DIM D, typename T, typename DeviceType>
class Lpk3Reo3D : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  Lpk3Reo3D() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Lpk3Reo3DFunctor<D, T, R, C, F, DeviceType>>
  GenTask(SIZE nr, SIZE nc_c, SIZE nf_c, SIZE nr_c,
          SubArray<1, T, DeviceType> ddist_r,
          SubArray<1, T, DeviceType> dratio_r, SubArray<D, T, DeviceType> dv1,
          SubArray<D, T, DeviceType> dv2, SubArray<D, T, DeviceType> dw,
          int queue_idx) {
    using FunctorType = Lpk3Reo3DFunctor<D, T, R, C, F, DeviceType>;
    FunctorType functor(nr, nc_c, nf_c, nr_c, ddist_r, dratio_r, dv1, dv2, dw);

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
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "Lpk3Reo3D");
  }

  MGARDX_CONT
  void Execute(SIZE nr, SIZE nc_c, SIZE nf_c, SIZE nr_c,
               SubArray<1, T, DeviceType> ddist_r,
               SubArray<1, T, DeviceType> dratio_r,
               SubArray<D, T, DeviceType> dv1, SubArray<D, T, DeviceType> dv2,
               SubArray<D, T, DeviceType> dw, int queue_idx) {
    int range_l = std::min(6, (int)std::log2(nf_c) - 1);
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.lpk3_3d[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define LPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = LPK_CONFIG[D - 1][CONFIG][0];                                \
    const int C = LPK_CONFIG[D - 1][CONFIG][1];                                \
    const int F = LPK_CONFIG[D - 1][CONFIG][2];                                \
    using FunctorType = Lpk3Reo3DFunctor<D, T, R, C, F, DeviceType>;           \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task = GenTask<R, C, F>(nr, nc_c, nf_c, nr_c, ddist_r, dratio_r,  \
                                     dv1, dv2, dw, queue_idx);                 \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    LPK(6) if (!ret.success) config--;
    LPK(5) if (!ret.success) config--;
    LPK(4) if (!ret.success) config--;
    LPK(3) if (!ret.success) config--;
    LPK(2) if (!ret.success) config--;
    LPK(1) if (!ret.success) config--;
    LPK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err << "no suitable config for Lpk3Reo3D.\n";
      exit(-1);
    }
#undef LPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("lpk3_3d", prec, range_l, min_config);
    }
  }
};

} // namespace mgard_x

#endif