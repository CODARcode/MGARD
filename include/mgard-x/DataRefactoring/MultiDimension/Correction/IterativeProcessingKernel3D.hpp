/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_ITERATIVE_PROCESSING_KERNEL_3D_TEMPLATE
#define MGARD_X_ITERATIVE_PROCESSING_KERNEL_3D_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"
#include "IPKFunctor.h"
// #include "IterativeProcessingKernel3D.h"

// #include "../../Functor.h"
// #include "../../AutoTuners/AutoTuner.h"
// #include "../../Task.h"
// #include "../../DeviceAdapters/DeviceAdapter.h"

namespace mgard_x {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, SIZE G,
          typename DeviceType>
class Ipk1Reo3DFunctor : public IterFunctor<DeviceType> {
public:
  MGARDX_CONT Ipk1Reo3DFunctor() {}
  MGARDX_CONT Ipk1Reo3DFunctor(SIZE nr, SIZE nc, SIZE nf,
                               SubArray<1, T, DeviceType> am,
                               SubArray<1, T, DeviceType> bm,
                               SubArray<1, T, DeviceType> dist_f,
                               SubArray<D, T, DeviceType> v)
      : nr(nr), nc(nc), nf(nf), am(am), bm(bm), dist_f(dist_f), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    c_gl = FunctorBase<DeviceType>::GetBlockIdX() * C;
    r_gl = FunctorBase<DeviceType>::GetBlockIdY() * R;
    f_gl = FunctorBase<DeviceType>::GetThreadIdX();

    c_sm = FunctorBase<DeviceType>::GetThreadIdX();
    r_sm = FunctorBase<DeviceType>::GetThreadIdY();
    f_sm = FunctorBase<DeviceType>::GetThreadIdX();

    v.offset(r_gl, c_gl, 0);
    T *sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F + G;
    ldsm2 = C;
    vec_sm = sm;
    am_sm = sm + R * ldsm1 * ldsm2;
    bm_sm = am_sm + ldsm1;

    prev_vec_sm = 0.0;

    c_rest = Math<DeviceType>::Min(
        C, nc - FunctorBase<DeviceType>::GetBlockIdX() * C);
    r_rest = Math<DeviceType>::Min(
        R, nr - FunctorBase<DeviceType>::GetBlockIdY() * R);

    // printf("r_rest: %u, c_rest: %u\n", r_rest, c_rest);
    // printf("RCF: %u %u %u\n", R,C,F);
    // printf("n: %u %u %u\n", nr, nc, nf_c);
    f_rest = nf;
    f_ghost = Math<DeviceType>::Min(nf, G);
    // printf("G%u, f_ghost:%u\n ", G, f_ghost);
    f_main = F;

    /* Load first ghost */
    if (r_sm < r_rest && f_sm < f_ghost) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = *v(r_sm, i, f_gl);
      }
      if (r_sm == 0) {
        am_sm[f_sm] = *am(f_gl);
        bm_sm[f_sm] = *bm(f_gl);
        // printf("am[%u]: %f, bm[%u]: %f\n", f_sm, f_sm, am_sm[f_sm],
        // bm_sm[f_sm]);
      }
    }

    f_rest -= f_ghost;
  }

  MGARDX_EXEC void Operation2() {}

  MGARDX_EXEC bool LoopCondition1() { return f_rest > F - f_ghost; }

  MGARDX_EXEC void Operation3() {
    f_main = Math<DeviceType>::Min(F, f_rest);
    if (r_sm < r_rest && f_sm < f_main) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + f_ghost)] =
            *v(r_sm, i, f_gl + f_ghost);
      }
      if (r_sm == 0) {
        am_sm[f_sm + f_ghost] = *am(f_gl + f_ghost);
        bm_sm[f_sm + f_ghost] = *bm(f_gl + f_ghost);
        // printf("am[%u]: %f, bm[%u]: %f\n", f_sm + f_ghost, f_sm + f_ghost,
        // am_sm[f_sm + f_ghost], bm_sm[f_sm + f_ghost]);
      }
    }
  }

  MGARDX_EXEC void Operation4() {
    /* Computation of v in parallel*/
    if (r_sm < r_rest && c_sm < c_rest) {
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
          tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                           vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);

      //#pragma unroll 32
      for (SIZE i = 1; i < F; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = tridiag_forward2(
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i],
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, F - 1)];
    }
  }

  MGARDX_EXEC void Operation5() {
    /* flush results to v */
    if (r_sm < r_rest && f_sm < F) {
      for (SIZE i = 0; i < c_rest; i++) {
        *v(r_sm, i, f_gl) = vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
  }

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
      if (r_sm == 0) {
        am_sm[f_sm] = am_sm[f_sm + FunctorBase<DeviceType>::GetBlockDimX()];
        bm_sm[f_sm] = bm_sm[f_sm + FunctorBase<DeviceType>::GetBlockDimX()];
      }
    }
  }

  MGARDX_EXEC void Operation7() {
    /* Load all rest col */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + f_ghost)] =
            *v(i, f_gl + f_ghost);
      }
      if (r_sm == 0) {
        am_sm[f_sm + f_ghost] = *am(f_gl + f_ghost);
        bm_sm[f_sm + f_ghost] = *bm(f_gl + f_ghost);
        // printf("am-ghost[%u]: %f, bm-ghost[%u]: %f\n", f_sm + f_ghost, f_sm +
        // f_ghost, am_sm[f_sm + f_ghost], bm_sm[f_sm + f_ghost]);
      }
    }
  }

  MGARDX_EXEC void Operation8() {
    /* Only 1 col remain */
    if (f_ghost + f_rest == 1) {
      if (r_sm < r_rest && c_sm < c_rest) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
            tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
      }
      //__syncthreads();

    } else {
      if (r_sm < r_rest && c_sm < c_rest) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
            tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
        // printf("am: %f, bm: %f\n", am_sm[0], bm_sm[0]);
        for (SIZE i = 1; i < f_ghost + f_rest; i++) {
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = tridiag_forward2(
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i],
              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
          // printf("am: %f, bm: %f\n", am_sm[i], bm_sm[i]);
        }
      }
    }
  }

  MGARDX_EXEC void Operation9() {
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_ghost + f_rest) {
      for (SIZE i = 0; i < c_rest; i++) {
        *v(r_sm, i, f_gl) = vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
  }

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
      }
    }
    if (r_sm == 0 && f_gl <= nf) {
      am_sm[f_sm] = *am(nf - f_gl);
      bm_sm[f_sm] = *bm(nf - f_gl); // * -1;
    }
    f_rest -= f_ghost;
  }

  MGARDX_EXEC bool LoopCondition2() { return f_rest > F - f_ghost; }

  MGARDX_EXEC void Operation11() {
    f_main = Math<DeviceType>::Min(F, f_rest);
    if (r_sm < r_rest && f_sm < f_main) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm + f_ghost)] =
            *v(r_sm, i, (nf - 1) - f_gl - f_ghost);
      }
    }
    if (r_sm == 0 && f_gl + f_ghost <= nf) {
      am_sm[f_sm + f_ghost] = *am(nf - f_gl - f_ghost);
      bm_sm[f_sm + f_ghost] = *bm(nf - f_gl - f_ghost); // * -1;
    }
  }

  MGARDX_EXEC void Operation12() {
    /* Computation of v in parallel*/
    if (r_sm < r_rest && c_sm < c_rest) {
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
          tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                            vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
      //#pragma unroll 32
      for (SIZE i = 1; i < F; i++) {
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

  MGARDX_EXEC void Operation13() {
    /* flush results to v */
    if (r_sm < r_rest && f_sm < F) {
      for (SIZE i = 0; i < c_rest; i++) {
        *v(r_sm, i, (nf - 1) - f_gl) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
  }

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
      // printf("%u %u\n", f_gl, f_ghost);
    }
  }

  MGARDX_EXEC void Operation16() {
    /* Only 1 col remain */
    if (f_ghost + f_rest == 1) {
      if (r_sm < r_rest && c_sm < c_rest) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
            tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                              vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
      }
      //__syncthreads();

    } else {
      if (r_sm < r_rest && c_sm < c_rest) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)] =
            tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                              vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, 0)]);
        for (SIZE i = 1; i < f_ghost + f_rest; i++) {
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)] = tridiag_backward2(
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i - 1)], am_sm[i],
              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm, i)]);
        }
      }
    }
  }

  MGARDX_EXEC void Operation17() {
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_ghost + f_rest) {
      for (SIZE i = 0; i < c_rest; i++) {
        *v(r_sm, i, (nf - 1) - f_gl) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }

    v.reset_offset();
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size = (R * C + 2) * (F + G) * sizeof(T);
    return size;
  }

private:
  // functor parameters
  SIZE nr, nc, nf;
  SubArray<1, T, DeviceType> am, bm;
  SubArray<1, T, DeviceType> dist_f;
  SubArray<D, T, DeviceType> v;

  // thread local variables
  SIZE c_gl, r_gl, f_gl;
  SIZE c_sm, r_sm, f_sm;
  SIZE ldsm1, ldsm2;
  T *vec_sm;
  T *am_sm;
  T *bm_sm;
  T prev_vec_sm;
  SIZE c_rest, r_rest;
  SIZE f_rest, f_ghost, f_main;
};

template <DIM D, typename T, typename DeviceType>
class Ipk1Reo3D : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  Ipk1Reo3D() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F, SIZE G>
  MGARDX_CONT Task<Ipk1Reo3DFunctor<D, T, R, C, F, G, DeviceType>>
  GenTask(SIZE nr, SIZE nc, SIZE nf, SubArray<1, T, DeviceType> am,
          SubArray<1, T, DeviceType> bm, SubArray<1, T, DeviceType> dist_f,
          SubArray<D, T, DeviceType> v, int queue_idx) {
    using FunctorType = Ipk1Reo3DFunctor<D, T, R, C, F, G, DeviceType>;
    FunctorType functor(nr, nc, nf, am, bm, dist_f, v);

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
    tbx = F;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "Ipk1Reo3D");
  }

  MGARDX_CONT
  void Execute(SIZE nr, SIZE nc, SIZE nf, SubArray<1, T, DeviceType> am,
               SubArray<1, T, DeviceType> bm, SubArray<1, T, DeviceType> dist_f,
               SubArray<D, T, DeviceType> v, int queue_idx) {
    int range_l = std::min(6, (int)std::log2(nf) - 1);
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.ipk1_3d[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define IPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = IPK_CONFIG[D - 1][CONFIG][0];                                \
    const int C = IPK_CONFIG[D - 1][CONFIG][1];                                \
    const int F = IPK_CONFIG[D - 1][CONFIG][2];                                \
    const int G = IPK_CONFIG[D - 1][CONFIG][3];                                \
    using FunctorType = Ipk1Reo3DFunctor<D, T, R, C, F, G, DeviceType>;        \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task =                                                            \
        GenTask<R, C, F, G>(nr, nc, nf, am, bm, dist_f, v, queue_idx);         \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    IPK(6) if (!ret.success) config--;
    IPK(5) if (!ret.success) config--;
    IPK(4) if (!ret.success) config--;
    IPK(3) if (!ret.success) config--;
    IPK(2) if (!ret.success) config--;
    IPK(1) if (!ret.success) config--;
    IPK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err << "no suitable config for Ipk1Reo3D.\n";
      exit(-1);
    }
#undef IPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("ipk1_3d", prec, range_l, min_config);
    }
  }
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, SIZE G,
          typename DeviceType>
class Ipk2Reo3DFunctor : public IterFunctor<DeviceType> {
public:
  MGARDX_CONT Ipk2Reo3DFunctor() {}
  MGARDX_CONT Ipk2Reo3DFunctor(SIZE nr, SIZE nc, SIZE nf,
                               SubArray<1, T, DeviceType> am,
                               SubArray<1, T, DeviceType> bm,
                               SubArray<1, T, DeviceType> dist_c,
                               SubArray<D, T, DeviceType> v)
      : nr(nr), nc(nc), nf(nf), am(am), bm(bm), dist_c(dist_c), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    f_gl = FunctorBase<DeviceType>::GetBlockIdX() * F;
    r_gl = FunctorBase<DeviceType>::GetBlockIdY() * R;
    c_gl = 0;

    f_sm = FunctorBase<DeviceType>::GetThreadIdX();
    r_sm = FunctorBase<DeviceType>::GetThreadIdY();
    c_sm = FunctorBase<DeviceType>::GetThreadIdX();

    v.offset(r_gl, 0, f_gl);
    T *sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F;
    ldsm2 = C + G;
    vec_sm = sm;
    am_sm = sm + R * ldsm1 * ldsm2;
    bm_sm = am_sm + ldsm2;

    prev_vec_sm = 0.0;

    f_rest = Math<DeviceType>::Min(
        F, nf - FunctorBase<DeviceType>::GetBlockIdX() * F);
    r_rest = Math<DeviceType>::Min(
        R, nr - FunctorBase<DeviceType>::GetBlockIdY() * R);

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

  MGARDX_EXEC void Operation2() {}

  MGARDX_EXEC bool LoopCondition1() { return c_rest > C - c_ghost; }

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

  MGARDX_EXEC void Operation4() {
    /* Computation of v in parallel*/
    if (r_sm < r_rest && f_sm < f_rest) {
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
          tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                           vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);

      for (SIZE i = 1; i < C; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = tridiag_forward2(
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], am_sm[i],
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
      }
      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, C - 1, f_sm)];
    }
  }

  MGARDX_EXEC void Operation5() {
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < C; i++) {
        *v(r_sm, c_gl + i, f_sm) = vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
  }

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

  MGARDX_EXEC void Operation8() {
    /* Only 1 col remain */
    if (c_ghost + c_rest == 1) {
      if (r_sm < r_rest && f_sm < f_rest) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
            tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
      }
      //__syncthreads();

    } else {
      if (r_sm < r_rest && f_sm < f_rest) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)] =
            tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, f_sm)]);
        for (SIZE i = 1; i < c_ghost + c_rest; i++) {
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = tridiag_forward2(
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], am_sm[i],
              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
        }
      }
    }
  }

  MGARDX_EXEC void Operation9() {
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_ghost + c_rest; i++) {
        *v(r_sm, c_gl + i, f_sm) = vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
  }

  MGARDX_EXEC void Operation10() {
    /* backward */
    c_rest = nc;
    c_ghost = Math<DeviceType>::Min(nc, G);
    c_main = C;
    c_gl = 0;
    prev_vec_sm = 0.0;

    /* Load first ghost */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] =
            *v(r_sm, (nc - 1) - (c_gl + i), f_sm);
      }
    }
    if (r_sm == 0 && c_sm < c_ghost) {
      am_sm[c_sm] = *am(nc - (c_gl + c_sm));
      bm_sm[c_sm] = *bm(nc - (c_gl + c_sm));
    }
    c_rest -= c_ghost;
  }

  MGARDX_EXEC bool LoopCondition2() { return c_rest > C - c_ghost; }

  MGARDX_EXEC void Operation11() {
    c_main = Math<DeviceType>::Min(C, c_rest);
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_main; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)] =
            *v(r_sm, (nc - 1) - (c_gl + i + c_ghost), f_sm);
      }
    }
    if (r_sm == 0 && c_sm < c_main) {
      am_sm[c_sm + c_ghost] = *am(nc - (c_gl + c_sm + c_ghost));
      bm_sm[c_sm + c_ghost] = *bm(nc - (c_gl + c_sm + c_ghost));
    }
  }

  MGARDX_EXEC void Operation12() {
    /* Computation of v in parallel*/
    if (r_sm < r_rest && f_sm < f_rest) {
      vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)] =
          tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                            vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)]);

      for (SIZE i = 1; i < C; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = tridiag_backward2(
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], am_sm[i],
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, r_sm, C - 1, f_sm)];
    }
  }

  MGARDX_EXEC void Operation13() {
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < C; i++) {
        *v(r_sm, (nc - 1) - (c_gl + i), f_sm) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
  }

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

  MGARDX_EXEC void Operation15() {
    // Load all rest col
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, i + c_ghost, f_sm)] =
            *v(r_sm, (nc - 1) - (c_gl + i + c_ghost), f_sm);
      }
    }
    if (r_sm == 0 && c_sm < c_rest) {
      am_sm[c_sm + c_ghost] = *am(nc - (c_gl + c_sm + c_ghost));
      bm_sm[c_sm + c_ghost] = *bm(nc - (c_gl + c_sm + c_ghost));
    }
  }

  MGARDX_EXEC void Operation16() {
    /* Only 1 col remain */
    if (c_ghost + c_rest == 1) {
      if (r_sm < r_rest && f_sm < f_rest) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)] =
            tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                              vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)]);
      }
      //__syncthreads();

    } else {
      if (r_sm < r_rest && f_sm < f_rest) {
        vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)] =
            tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                              vec_sm[get_idx(ldsm1, ldsm2, r_sm, 0, c_sm)]);
        for (SIZE i = 1; i < c_ghost + c_rest; i++) {
          vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)] = tridiag_backward2(
              vec_sm[get_idx(ldsm1, ldsm2, r_sm, i - 1, f_sm)], am_sm[i],
              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)]);
        }
      }
    }
  }

  MGARDX_EXEC void Operation17() {
    /* flush results to v */
    if (r_sm < r_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < c_ghost + c_rest; i++) {
        *v(r_sm, (nc - 1) - (c_gl + i), f_sm) =
            vec_sm[get_idx(ldsm1, ldsm2, r_sm, i, f_sm)];
      }
    }
    v.reset_offset();
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size = (R * F + 2) * (C + G) * sizeof(T);
    return size;
  }

private:
  // functor parameters
  SIZE nr, nc, nf;
  SubArray<1, T, DeviceType> am, bm;
  SubArray<1, T, DeviceType> dist_c;
  SubArray<D, T, DeviceType> v;

  // thread local variables
  SIZE c_gl, r_gl, f_gl;
  SIZE c_sm, r_sm, f_sm;
  SIZE ldsm1, ldsm2;
  T *vec_sm;
  T *am_sm;
  T *bm_sm;
  T prev_vec_sm;
  SIZE f_rest, r_rest;
  SIZE c_rest, c_ghost, c_main;
};

template <DIM D, typename T, typename DeviceType>
class Ipk2Reo3D : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  Ipk2Reo3D() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F, SIZE G>
  MGARDX_CONT Task<Ipk2Reo3DFunctor<D, T, R, C, F, G, DeviceType>>
  GenTask(SIZE nr, SIZE nc, SIZE nf, SubArray<1, T, DeviceType> am,
          SubArray<1, T, DeviceType> bm, SubArray<1, T, DeviceType> dist_c,
          SubArray<D, T, DeviceType> v, int queue_idx) {
    using FunctorType = Ipk2Reo3DFunctor<D, T, R, C, F, G, DeviceType>;
    FunctorType functor(nr, nc, nf, am, bm, dist_c, v);

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
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "Ipk2Reo3D");
  }

  MGARDX_CONT
  void Execute(SIZE nr, SIZE nc, SIZE nf, SubArray<1, T, DeviceType> am,
               SubArray<1, T, DeviceType> bm, SubArray<1, T, DeviceType> dist_c,
               SubArray<D, T, DeviceType> v, int queue_idx) {
    int range_l = std::min(6, (int)std::log2(nf) - 1);
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.ipk2_3d[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define IPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = IPK_CONFIG[D - 1][CONFIG][0];                                \
    const int C = IPK_CONFIG[D - 1][CONFIG][1];                                \
    const int F = IPK_CONFIG[D - 1][CONFIG][2];                                \
    const int G = IPK_CONFIG[D - 1][CONFIG][3];                                \
    using FunctorType = Ipk2Reo3DFunctor<D, T, R, C, F, G, DeviceType>;        \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task =                                                            \
        GenTask<R, C, F, G>(nr, nc, nf, am, bm, dist_c, v, queue_idx);         \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    IPK(6) if (!ret.success) config--;
    IPK(5) if (!ret.success) config--;
    IPK(4) if (!ret.success) config--;
    IPK(3) if (!ret.success) config--;
    IPK(2) if (!ret.success) config--;
    IPK(1) if (!ret.success) config--;
    IPK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err << "no suitable config for Ipk2Reo3D.\n";
      exit(-1);
    }
#undef IPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("ipk2_3d", prec, range_l, min_config);
    }
  }
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, SIZE G,
          typename DeviceType>
class Ipk3Reo3DFunctor : public IterFunctor<DeviceType> {
public:
  MGARDX_CONT Ipk3Reo3DFunctor() {}
  MGARDX_CONT Ipk3Reo3DFunctor(SIZE nr, SIZE nc, SIZE nf,
                               SubArray<1, T, DeviceType> am,
                               SubArray<1, T, DeviceType> bm,
                               SubArray<1, T, DeviceType> dist_r,
                               SubArray<D, T, DeviceType> v)
      : nr(nr), nc(nc), nf(nf), am(am), bm(bm), dist_r(dist_r), v(v) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    f_gl = FunctorBase<DeviceType>::GetBlockIdX() * F;
    c_gl = FunctorBase<DeviceType>::GetBlockIdY() * C;
    r_gl = 0;

    f_sm = FunctorBase<DeviceType>::GetThreadIdX();
    c_sm = FunctorBase<DeviceType>::GetThreadIdY();
    r_sm = FunctorBase<DeviceType>::GetThreadIdX();

    v.offset(0, c_gl, f_gl);
    T *sm = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    ldsm1 = F;
    ldsm2 = C;
    vec_sm = sm;
    am_sm = sm + (R + G) * ldsm1 * ldsm2;
    bm_sm = am_sm + (R + G);

    prev_vec_sm = 0.0;

    f_rest = Math<DeviceType>::Min(
        F, nf - FunctorBase<DeviceType>::GetBlockIdX() * F);
    c_rest = Math<DeviceType>::Min(
        C, nc - FunctorBase<DeviceType>::GetBlockIdY() * C);

    r_rest = nr;
    r_ghost = Math<DeviceType>::Min(nr, G);
    r_main = R;

    /* Load first ghost */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_ghost; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = *v(r_gl + i, c_sm, f_sm);
      }
    }

    if (c_sm == 0 && r_sm < r_ghost) {
      am_sm[r_sm] = *am(r_gl + r_sm);
      bm_sm[r_sm] = *bm(r_gl + r_sm);
    }
    r_rest -= r_ghost;
  }

  MGARDX_EXEC void Operation2() {}

  MGARDX_EXEC bool LoopCondition1() { return r_rest > R - r_ghost; }

  MGARDX_EXEC void Operation3() {
    r_main = Math<DeviceType>::Min(R, r_rest);
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_main; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)] =
            *v(r_gl + i + r_ghost, c_sm, f_sm);
      }
    }
    if (c_sm == 0 && r_sm < r_main) {
      am_sm[r_sm + r_ghost] = *am(r_gl + r_sm + r_ghost);
      bm_sm[r_sm + r_ghost] = *bm(r_gl + r_sm + r_ghost);
    }
  }

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

  MGARDX_EXEC void Operation5() {
    /* flush results to v */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < R; i++) {
        *v(r_gl + i, c_sm, f_sm) = vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];
      }
    }
  }

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
      }
    }
    if (c_sm == 0 && r_sm < r_ghost) {
      am_sm[r_sm] = am_sm[r_sm + R];
      bm_sm[r_sm] = bm_sm[r_sm + R];
    }
  }

  MGARDX_EXEC void Operation7() {
    /* Load all rest col */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)] =
            *v(r_gl + i + r_ghost, c_sm, f_sm);
      }
    }

    if (c_sm == 0 && r_sm < r_rest) {
      am_sm[r_sm + r_ghost] = *am(r_gl + r_sm + r_ghost);
      bm_sm[r_sm + r_ghost] = *bm(r_gl + r_sm + r_ghost);
    }
  }

  MGARDX_EXEC void Operation8() {
    /* Only 1 col remain */
    if (r_ghost + r_rest == 1) {
      if (c_sm < c_rest && f_sm < f_rest) {
        vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
            tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
      }
      //__syncthreads();

    } else {
      if (c_sm < c_rest && f_sm < f_rest) {
        vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
            tridiag_forward2(prev_vec_sm, am_sm[0], bm_sm[0],
                             vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
        for (SIZE i = 1; i < r_ghost + r_rest; i++) {
          vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = tridiag_forward2(
              vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], am_sm[i],
              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
        }
      }
    }
  }

  MGARDX_EXEC void Operation9() {
    /* flush results to v */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_ghost + r_rest; i++) {
        *v(r_gl + i, c_sm, f_sm) = vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];
      }
    }
  }

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

  MGARDX_EXEC bool LoopCondition2() { return r_rest > R - r_ghost; }

  MGARDX_EXEC void Operation11() {
    r_main = Math<DeviceType>::Min(R, r_rest);
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_main; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)] =
            *v((nr - 1) - (r_gl + i + r_ghost), c_sm, f_sm);
      }
    }
    if (c_sm == 0 && r_sm < r_main) {
      am_sm[r_sm + r_ghost] = *am(nr - (r_gl + r_sm + r_ghost));
      bm_sm[r_sm + r_ghost] = *bm(nr - (r_gl + r_sm + r_ghost));
    }
  }

  MGARDX_EXEC void Operation12() {
    /* Computation of v in parallel*/
    if (c_sm < c_rest && f_sm < f_rest) {
      vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
          tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                            vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
      for (SIZE i = 1; i < R; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = tridiag_backward2(
            vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], am_sm[i],
            bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
      }

      /* Store last v */
      prev_vec_sm = vec_sm[get_idx(ldsm1, ldsm2, R - 1, c_sm, f_sm)];
    }
  }

  MGARDX_EXEC void Operation13() {
    /* flush results to v */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < R; i++) {
        *v((nr - 1) - (r_gl + i), c_sm, f_sm) =
            vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];
      }
    }
  }

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
      }
    }
    if (c_sm == 0 && r_sm < r_ghost) {
      am_sm[r_sm] = am_sm[r_sm + R];
      bm_sm[r_sm] = bm_sm[r_sm + R];
    }
  }

  MGARDX_EXEC void Operation15() {
    /* Load all rest col */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_rest; i++) {
        vec_sm[get_idx(ldsm1, ldsm2, i + r_ghost, c_sm, f_sm)] =
            *v((nr - 1) - (r_gl + i + r_ghost), c_sm, f_sm);
      }
    }
    if (c_sm == 0 && r_sm < r_rest) {
      am_sm[r_sm + r_ghost] = *am(nr - (r_gl + r_sm + r_ghost));
      bm_sm[r_sm + r_ghost] = *bm(nr - (r_gl + r_sm + r_ghost));
    }
  }

  MGARDX_EXEC void Operation16() {
    /* Only 1 col remain */
    if (r_ghost + r_rest == 1) {
      if (c_sm < c_rest && f_sm < f_rest) {
        vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
            tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                              vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
      }
      //__syncthreads();

    } else {
      if (c_sm < c_rest && f_sm < f_rest) {
        vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)] =
            tridiag_backward2(prev_vec_sm, am_sm[0], bm_sm[0],
                              vec_sm[get_idx(ldsm1, ldsm2, 0, c_sm, f_sm)]);
        for (SIZE i = 1; i < r_ghost + r_rest; i++) {
          vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)] = tridiag_backward2(
              vec_sm[get_idx(ldsm1, ldsm2, i - 1, c_sm, f_sm)], am_sm[i],
              bm_sm[i], vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)]);
        }
      }
    }
  }

  MGARDX_EXEC void Operation17() {
    /* flush results to v */
    if (c_sm < c_rest && f_sm < f_rest) {
      for (SIZE i = 0; i < r_ghost + r_rest; i++) {
        *v((nr - 1) - (r_gl + i), c_sm, f_sm) =
            vec_sm[get_idx(ldsm1, ldsm2, i, c_sm, f_sm)];
      }
    }
    v.reset_offset();
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    size = (C * F + 2) * (R + G) * sizeof(T);
    return size;
  }

private:
  // functor parameters
  SIZE nr, nc, nf;
  SubArray<1, T, DeviceType> am, bm;
  SubArray<1, T, DeviceType> dist_r;
  SubArray<D, T, DeviceType> v;

  // thread local variables
  SIZE c_gl, r_gl, f_gl;
  SIZE c_sm, r_sm, f_sm;
  SIZE ldsm1, ldsm2;
  T *vec_sm;
  T *am_sm;
  T *bm_sm;
  T prev_vec_sm;
  SIZE f_rest, c_rest;
  SIZE r_rest, r_ghost, r_main;
};

template <DIM D, typename T, typename DeviceType>
class Ipk3Reo3D : public AutoTuner<DeviceType> {
public:
  MGARDX_CONT
  Ipk3Reo3D() : AutoTuner<DeviceType>() {}

  template <SIZE R, SIZE C, SIZE F, SIZE G>
  MGARDX_CONT Task<Ipk3Reo3DFunctor<D, T, R, C, F, G, DeviceType>>
  GenTask(SIZE nr, SIZE nc, SIZE nf, SubArray<1, T, DeviceType> am,
          SubArray<1, T, DeviceType> bm, SubArray<1, T, DeviceType> dist_r,
          SubArray<D, T, DeviceType> v, int queue_idx) {
    using FunctorType = Ipk3Reo3DFunctor<D, T, R, C, F, G, DeviceType>;
    FunctorType functor(nr, nc, nf, am, bm, dist_r, v);

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
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                "Ipk3Reo3D");
  }

  MGARDX_CONT
  void Execute(SIZE nr, SIZE nc, SIZE nf, SubArray<1, T, DeviceType> am,
               SubArray<1, T, DeviceType> bm, SubArray<1, T, DeviceType> dist_r,
               SubArray<D, T, DeviceType> v, int queue_idx) {
    int range_l = std::min(6, (int)std::log2(nf) - 1);
    int prec = TypeToIdx<T>();
    int config = AutoTuner<DeviceType>::autoTuningTable.ipk3_3d[prec][range_l];
    double min_time = std::numeric_limits<double>::max();
    int min_config = 0;
    ExecutionReturn ret;

#define IPK(CONFIG)                                                            \
  if (config == CONFIG || AutoTuner<DeviceType>::ProfileKernels) {             \
    const int R = IPK_CONFIG[D - 1][CONFIG][0];                                \
    const int C = IPK_CONFIG[D - 1][CONFIG][1];                                \
    const int F = IPK_CONFIG[D - 1][CONFIG][2];                                \
    const int G = IPK_CONFIG[D - 1][CONFIG][3];                                \
    using FunctorType = Ipk3Reo3DFunctor<D, T, R, C, F, G, DeviceType>;        \
    using TaskType = Task<FunctorType>;                                        \
    TaskType task =                                                            \
        GenTask<R, C, F, G>(nr, nc, nf, am, bm, dist_r, v, queue_idx);         \
    DeviceAdapter<TaskType, DeviceType> adapter;                               \
    ret = adapter.Execute(task);                                               \
    if (AutoTuner<DeviceType>::ProfileKernels) {                               \
      if (ret.success && min_time > ret.execution_time) {                      \
        min_time = ret.execution_time;                                         \
        min_config = CONFIG;                                                   \
      }                                                                        \
    }                                                                          \
  }

    IPK(6) if (!ret.success) config--;
    IPK(5) if (!ret.success) config--;
    IPK(4) if (!ret.success) config--;
    IPK(3) if (!ret.success) config--;
    IPK(2) if (!ret.success) config--;
    IPK(1) if (!ret.success) config--;
    IPK(0) if (!ret.success) config--;
    if (config < 0 && !ret.success) {
      std::cout << log::log_err << "no suitable config for Ipk3Reo3D.\n";
      exit(-1);
    }
#undef IPK

    if (AutoTuner<DeviceType>::ProfileKernels) {
      FillAutoTunerTable<DeviceType>("ipk3_3d", prec, range_l, min_config);
    }
  }
};

} // namespace mgard_x

#endif