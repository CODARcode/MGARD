/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_LEVELWISE_PROCESSING_KERNEL_TEMPLATE
#define MGARD_X_LEVELWISE_PROCESSING_KERNEL_TEMPLATE

#include "../../../RuntimeX/RuntimeX.h"

namespace mgard_x {

namespace data_refactoring {

namespace multi_dimension {

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, OPTION OP,
          typename DeviceType>
class LwpkFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT LwpkFunctor() {}
  MGARDX_CONT LwpkFunctor(SubArray<D, T, DeviceType> v,
                          SubArray<D, T, DeviceType> work)
      : v(v), work(work) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    threadId = (FunctorBase<DeviceType>::GetThreadIdZ() *
                (FunctorBase<DeviceType>::GetBlockDimX() *
                 FunctorBase<DeviceType>::GetBlockDimY())) +
               (FunctorBase<DeviceType>::GetThreadIdY() *
                FunctorBase<DeviceType>::GetBlockDimX()) +
               FunctorBase<DeviceType>::GetThreadIdX();

    SIZE idx[D];
    SIZE firstD = div_roundup(v.shape(D - 1), F);

    SIZE bidx = FunctorBase<DeviceType>::GetBlockIdX();
    idx[D - 1] = (bidx % firstD) * F + FunctorBase<DeviceType>::GetThreadIdX();

    bidx /= firstD;
    if (D >= 2)
      idx[D - 2] = FunctorBase<DeviceType>::GetBlockIdY() *
                       FunctorBase<DeviceType>::GetBlockDimY() +
                   FunctorBase<DeviceType>::GetThreadIdY();
    if (D >= 3)
      idx[D - 3] = FunctorBase<DeviceType>::GetBlockIdZ() *
                       FunctorBase<DeviceType>::GetBlockDimZ() +
                   FunctorBase<DeviceType>::GetThreadIdZ();

    for (int d = D - 4; d >= 0; d--) {
      idx[d] = bidx % v.shape(d);
      bidx /= v.shape(d);
    }
    bool in_range = true;
    for (DIM d = 0; d < D; d++) {
      if (idx[d] >= v.shape(d))
        in_range = false;
    }
    if (in_range) {
      // printf("%d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
      if (OP == COPY) {
        // *work(idx) = *v(idx);
        work[idx] = v[idx];
      }
      if (OP == ADD) {
        work[idx] += v[idx];
      }
      if (OP == SUBTRACT) {
        work[idx] -= v[idx];
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SubArray<D, T, DeviceType> v, work;
  IDX threadId;
};

template <DIM D, typename T, OPTION OP, typename DeviceType>
class LwpkReoKernel : public Kernel {
public:
  constexpr static DIM NumDim = D;
  using DataType = T;
  constexpr static std::string_view Name = "lwpk";
  MGARDX_CONT
  LwpkReoKernel(SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> work)
      : v(v), work(work) {}
  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<LwpkFunctor<D, T, R, C, F, OP, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = LwpkFunctor<D, T, R, C, F, OP, DeviceType>;
    FunctorType functor(v, work);

    SIZE total_thread_z = 1;
    SIZE total_thread_y = 1;
    SIZE total_thread_x = 1;
    if (D >= 3)
      total_thread_z = v.shape(D - 3);
    if (D >= 2)
      total_thread_y = v.shape(D - 2);
    total_thread_x = v.shape(D - 1);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    for (int d = D - 4; d >= 0; d--) {
      gridx *= v.shape(d);
    }
    // printf("%u %u %u\n", shape.dataHost()[2], shape.dataHost()[1],
    // shape.dataHost()[0]); PrintSubarray("shape", shape);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<D, T, DeviceType> v, work;
};

template <DIM D, typename T, SIZE R, SIZE C, SIZE F, OPTION OP, bool PADDING,
          typename DeviceType>
class Lwpk3DFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT Lwpk3DFunctor() {}
  MGARDX_CONT Lwpk3DFunctor(SIZE nr, SIZE nc, SIZE nf,
                            SubArray<D, T, DeviceType> v,
                            SubArray<D, T, DeviceType> work)
      : nr(nr), nc(nc), nf(nf), v(v), work(work) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    f_gl = FunctorBase<DeviceType>::GetBlockIdX() *
               FunctorBase<DeviceType>::GetBlockDimX() +
           FunctorBase<DeviceType>::GetThreadIdX();
    if constexpr (D >= 2)
      c_gl = FunctorBase<DeviceType>::GetBlockIdY() *
                 FunctorBase<DeviceType>::GetBlockDimY() +
             FunctorBase<DeviceType>::GetThreadIdY();
    if constexpr (D >= 3)
      r_gl = FunctorBase<DeviceType>::GetBlockIdZ() *
                 FunctorBase<DeviceType>::GetBlockDimZ() +
             FunctorBase<DeviceType>::GetThreadIdZ();

    if (r_gl < nr && c_gl < nc && f_gl < nf) {
      // printf("%d %d %d %d\n", idx[3], idx[2], idx[1], idx[0]);
      if (OP == COPY) {
        // *work(idx) = *v(idx);
        *work(r_gl, c_gl, f_gl) = *v(r_gl, c_gl, f_gl);

        /*
        if (nr % 2 == 0 && r_gl == 0) {
          *work(r_gl+1, c_gl, f_gl) = *v(r_gl, c_gl, f_gl);
        }

        if (c_sm == 0) {
          if (rest_c > (C / 2) * 2) {
            // v_sm[get_idx(ldsm1, ldsm2, r_sm, c_sm_ex, f_sm)] =
            //     dv[get_idx(lddv1, lddv2, r_gl, c_gl_ex, f_gl)];
            // printf("load-c[%d %d %d]:%f --> [%d %d %d]\n", r_gl, c_gl_ex,
        f_gl,
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
            // printf("load-f[%d %d %d]:%f --> [%d %d %d]\n", r_gl, c_gl,
        f_gl_ex,
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
            // printf("load-rcf[%d %d %d]:%f --> [%d %d %d]\n", r_gl_ex,
        c_gl_ex,
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

        */
      }
      if (OP == ADD) {
        *work(r_gl, c_gl, f_gl) += *v(r_gl, c_gl, f_gl);
      }
      if (OP == SUBTRACT) {
        *work(r_gl, c_gl, f_gl) -= *v(r_gl, c_gl, f_gl);
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    size_t size = 0;
    return size;
  }

private:
  SIZE nr, nc, nf;
  SubArray<D, T, DeviceType> v, work;
  SIZE r_gl, c_gl, f_gl;
};

template <DIM D, typename T, OPTION OP, bool PADDING, typename DeviceType>
class Lwpk3DKernel : public Kernel {
public:
  constexpr static DIM NumDim = D;
  using DataType = T;
  constexpr static std::string_view Name = "lwpk";
  MGARDX_CONT
  Lwpk3DKernel(SIZE nr, SIZE nc, SIZE nf, SubArray<D, T, DeviceType> v,
               SubArray<D, T, DeviceType> work)
      : nr(nr), nc(nc), nf(nf), v(v), work(work) {}
  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<Lwpk3DFunctor<D, T, R, C, F, OP, PADDING, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = Lwpk3DFunctor<D, T, R, C, F, OP, PADDING, DeviceType>;
    FunctorType functor(nr, nc, nf, v, work);
    SIZE total_thread_z = nr;
    SIZE total_thread_y = nc;
    SIZE total_thread_x = nf;

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = ceil((float)total_thread_z / tbz);
    gridy = ceil((float)total_thread_y / tby);
    gridx = ceil((float)total_thread_x / tbx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SIZE nr, nc, nf;
  SubArray<D, T, DeviceType> v, work;
};

} // namespace multi_dimension

} // namespace data_refactoring

} // namespace mgard_x

#endif