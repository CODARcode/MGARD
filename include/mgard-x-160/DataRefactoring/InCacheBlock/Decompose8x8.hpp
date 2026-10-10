/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_DECOMPOSE_8x8_KERNEL_TEMPLATE
#define MGARD_X_160_DECOMPOSE_8x8_KERNEL_TEMPLATE

#include "../../RuntimeX/RuntimeX.h"

#include "IndexTableLowDim.hpp"

namespace mgard_x_160 {

namespace data_refactoring {

namespace in_cache_block {

/*
2D counterpart of Decompose8x8x8Functor. One thread block owns one 8x8 tile
and 64 threads; the stages are the same as the 3D block's, minus the z pass:

v         x        y        c              total
8*8(64)   5*8(40)  5*5(25)  0              129
                            8*8-5*5(39)

The 39 coefficients per tile are written to coeff[bid * 39 ...] in the
row-major non-coarse order of Coeff_Offset_8x8, and the 25 coarse values to
coarse[y_tb * 5 + jy][x_tb * 5 + jx].
*/
template <DIM D, typename T, SIZE Y, SIZE X, typename DeviceType>
class Decompose8x8Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Decompose8x8Functor() {}
  MGARDX_CONT Decompose8x8Functor(SubArray<D, T, DeviceType> v,
                                  SubArray<D, T, DeviceType> coarse,
                                  SubArray<1, T, DeviceType> coeff,
                                  bool orthogonal_projection = true)
      : v(v), coarse(coarse), coeff(coeff),
        orthogonal_projection(orthogonal_projection) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void initialize_sm_8x8() {
    sm_v = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_x = sm_v + SMV_SIZE_8x8; // sm_v is padded for bank-conflict avoidance
    sm_y = sm_x + SMX_SIZE_8x8;
  }

  MGARDX_EXEC void init_indices() {
    x = FunctorBase<DeviceType>::GetThreadIdX();
    y = FunctorBase<DeviceType>::GetThreadIdY();
    x_tb = FunctorBase<DeviceType>::GetBlockIdX();
    y_tb = FunctorBase<DeviceType>::GetBlockIdY();
    x_gl = X * x_tb + x;
    y_gl = Y * y_tb + y;
    tid = y * X + x;
    bid = y_tb * FunctorBase<DeviceType>::GetGridDimX() + x_tb;
    if (tid == 0)
      sm_v[ZERO_V_8x8] = (T)0;
    offset = offset8x8(y, x);
  }

  // Load
  MGARDX_EXEC void Operation1() {
    initialize_sm_8x8();
    init_indices();
    sm_v[offset] = *v(y_gl, x_gl);
  }

  // Interpolation. The 1D and 2D coefficient updates are independent: a 1D
  // update writes a node with exactly one odd coordinate, while the 2D update
  // reads only nodes with two even coordinates.
  MGARDX_EXEC void Operation2() {
    if (tid < NumCoeff1D_8x8) {
      int const *index = Coeff1D_Offset_8x8(tid);
      T middle = sm_v[index[0]];
      T left = sm_v[index[1]];
      T right = sm_v[index[2]];
      sm_v[index[0]] = middle - (left + right) * (T)0.5;
    } else if (tid >= 32 && tid < 32 + NumCoeff2D_8x8) {
      int const *index = Coeff2D_Offset_8x8(tid - 32);
      T c11 = sm_v[index[0]];
      T c00 = sm_v[index[1]];
      T c02 = sm_v[index[2]];
      T c20 = sm_v[index[3]];
      T c22 = sm_v[index[4]];
      sm_v[index[0]] = c11 - (c00 + c02 + c20 + c22) / 4;
    }
  }

  // MassTransX
  MGARDX_EXEC void Operation3() {
    if (!orthogonal_projection)
      return;
    if (tid < NumMassTransX_8x8) {
      int const *index = MassTrans_X_Offset_8x8(tid);
      T a = sm_v[index[0]];
      T b = sm_v[index[1]];
      T c = sm_v[index[2]];
      T d = sm_v[index[3]];
      T e = sm_v[index[4]];
      T const *dist = MassTrans_Weights_8x8x8<T>(index[6]);
      sm_x[index[5]] =
          a * dist[0] + b * dist[1] + c * dist[2] + d * dist[3] + e * dist[4];
    }
  }

  // MassTransY
  MGARDX_EXEC void Operation4() {
    if (!orthogonal_projection)
      return;
    if (tid < NumMassTransY_8x8) {
      int const *index = MassTrans_Y_Offset_8x8(tid);
      T a = sm_x[index[0]];
      T b = sm_x[index[1]];
      T c = sm_x[index[2]];
      T d = sm_x[index[3]];
      T e = sm_x[index[4]];
      T const *dist = MassTrans_Weights_8x8x8<T>(index[6]);
      sm_y[index[5]] =
          a * dist[0] + b * dist[1] + c * dist[2] + d * dist[3] + e * dist[4];
    }
  }

  MGARDX_EXEC void solve_tridiag(int const *index) {
    T a = sm_y[index[0]];
    T b = sm_y[index[1]];
    T c = sm_y[index[2]];
    T d = sm_y[index[3]];
    T e = sm_y[index[4]];

    a += 0 * amxbm_8x8x8<T>(0);
    b += a * amxbm_8x8x8<T>(1);
    c += b * amxbm_8x8x8<T>(2);
    d += c * amxbm_8x8x8<T>(3);
    e += d * amxbm_8x8x8<T>(4);

    e = (e + am_8x8x8<T>(5) * 0) * bm_8x8x8<T>(5);
    d = (d + am_8x8x8<T>(4) * e) * bm_8x8x8<T>(4);
    c = (c + am_8x8x8<T>(3) * d) * bm_8x8x8<T>(3);
    b = (b + am_8x8x8<T>(2) * c) * bm_8x8x8<T>(2);
    a = (a + am_8x8x8<T>(1) * b) * bm_8x8x8<T>(1);

    sm_y[index[0]] = a;
    sm_y[index[1]] = b;
    sm_y[index[2]] = c;
    sm_y[index[3]] = d;
    sm_y[index[4]] = e;
  }

  // TridiagX
  MGARDX_EXEC void Operation5() {
    if (!orthogonal_projection)
      return;
    if (tid < LowDim_Coarse) {
      solve_tridiag(TriDiag_X_Offset_8x8(tid));
    }
  }

  // TridiagY
  MGARDX_EXEC void Operation6() {
    if (!orthogonal_projection)
      return;
    if (tid < LowDim_Coarse) {
      solve_tridiag(TriDiag_Y_Offset_8x8(tid));
    }
  }

  // Apply correction and write out
  MGARDX_EXEC void Operation7() {
    if (tid < NumCoarse_8x8) {
      if (orthogonal_projection)
        sm_v[Coarse_Offset_8x8(tid)] += sm_y[tid];
      *coarse(y_tb * LowDim_Coarse + tid / LowDim_Coarse,
              x_tb * LowDim_Coarse + tid % LowDim_Coarse) =
          sm_v[Coarse_Offset_8x8(tid)];
    } else {
      int op_tid = tid - NumCoarse_8x8;
      *coeff(bid * NumCoeff_8x8 + op_tid) = sm_v[Coeff_Offset_8x8(op_tid)];
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    return (size_t)SM_SIZE_8x8 * sizeof(T);
  }

protected:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, T, DeviceType> coeff;
  bool orthogonal_projection;
  T *sm_v, *sm_x, *sm_y;
  int y, x, y_tb, x_tb, y_gl, x_gl;
  int tid, bid;
  int offset;
};

// Fused decompose+quantize variant: identical transform pipeline, but
// (1) the input is read with bounds checks so unpadded inputs can be consumed
//     directly (out-of-range positions contribute zeros, matching the
//     memset+copy padding of the unfused path bit for bit), and
// (2) the 39 per-tile coefficients are quantized on write-out while still in
//     shared memory, emitting Q symbols (with the Huffman dictionary shift
//     folded in, mirroring QuantizeLocalLevelFunctor) instead of T values.
// The 5x5 coarse output stays in T: it is the next level's input.
template <DIM D, typename T, typename Q, SIZE Y, SIZE X, typename DeviceType>
class DecomposeQuantize8x8Functor
    : public Decompose8x8Functor<D, T, Y, X, DeviceType> {
  using Base = Decompose8x8Functor<D, T, Y, X, DeviceType>;

public:
  MGARDX_CONT DecomposeQuantize8x8Functor() {}
  MGARDX_CONT DecomposeQuantize8x8Functor(
      SubArray<D, T, DeviceType> v, SubArray<D, T, DeviceType> coarse,
      SubArray<1, Q, DeviceType> quantized_coeff, T quantizer,
      SubArray<1, T, DeviceType> block_quantizers, bool use_block_quantizers,
      bool prep_huffman, SIZE dict_size, bool orthogonal_projection = true)
      : Base(v, coarse, SubArray<1, T, DeviceType>(), orthogonal_projection),
        quantized_coeff(quantized_coeff), quantizer(quantizer),
        block_quantizers(block_quantizers),
        use_block_quantizers(use_block_quantizers), prep_huffman(prep_huffman),
        dict_size(dict_size) {}

  MGARDX_EXEC void Operation1() {
    this->initialize_sm_8x8();
    this->init_indices();
    this->sm_v[this->offset] = (T)0;
    // Unlike the unfused functor, keep the bounds check: the input here is
    // not pre-padded to a multiple of 8, so edge tiles must zero-fill.
    if (this->y_gl < (int)this->v.shape(D - 2) &&
        this->x_gl < (int)this->v.shape(D - 1)) {
      this->sm_v[this->offset] = *this->v(this->y_gl, this->x_gl);
    }
  }

  MGARDX_EXEC void Operation7() {
    if (this->tid < NumCoarse_8x8) {
      if (this->orthogonal_projection)
        this->sm_v[Coarse_Offset_8x8(this->tid)] += this->sm_y[this->tid];
      *this->coarse(this->y_tb * LowDim_Coarse + this->tid / LowDim_Coarse,
                    this->x_tb * LowDim_Coarse + this->tid % LowDim_Coarse) =
          this->sm_v[Coarse_Offset_8x8(this->tid)];
    } else {
      int op_tid = this->tid - NumCoarse_8x8;
      T t = this->sm_v[Coeff_Offset_8x8(op_tid)];
      T q = use_block_quantizers ? *block_quantizers(this->bid) : quantizer;
      // Must stay bit-identical to QuantizeLocalLevelFunctor (volume == 1).
      Q quantized_data;
      if constexpr (sizeof(T) == sizeof(double)) {
        quantized_data = copysign((T)0.5 + fabs(t * q), t);
      } else if constexpr (sizeof(T) == sizeof(float)) {
        quantized_data = copysign((T)0.5 + fabsf(t * q), t);
      }
      if (prep_huffman) {
        quantized_data += dict_size / 2;
      }
      *quantized_coeff(this->bid * NumCoeff_8x8 + op_tid) = quantized_data;
    }
  }

protected:
  SubArray<1, Q, DeviceType> quantized_coeff;
  T quantizer;
  SubArray<1, T, DeviceType> block_quantizers;
  bool use_block_quantizers;
  bool prep_huffman;
  SIZE dict_size;
};

template <DIM D, typename T, typename DeviceType>
class Decompose8x8Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lwpk_2d";
  MGARDX_CONT
  Decompose8x8Kernel(SubArray<D, T, DeviceType> v,
                     SubArray<D, T, DeviceType> coarse,
                     SubArray<1, T, DeviceType> coeff,
                     bool orthogonal_projection = true)
      : v(v), coarse(coarse), coeff(coeff),
        orthogonal_projection(orthogonal_projection) {}

  MGARDX_CONT Task<Decompose8x8Functor<D, T, 8, 8, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = Decompose8x8Functor<D, T, 8, 8, DeviceType>;
    FunctorType functor(v, coarse, coeff, orthogonal_projection);

    SIZE total_thread_y = v.shape(D - 2);
    SIZE total_thread_x = v.shape(D - 1);

    size_t sm_size = functor.shared_memory_size();
    SIZE tbz = 1, tby = 8, tbx = 8;
    SIZE gridz = 1;
    SIZE gridy = ceil((double)total_thread_y / tby);
    SIZE gridx = ceil((double)total_thread_x / tbx);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, T, DeviceType> coeff;
  bool orthogonal_projection;
};

template <DIM D, typename T, typename Q, typename DeviceType>
class DecomposeQuantize8x8Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lwpk_2d_fq";
  MGARDX_CONT
  DecomposeQuantize8x8Kernel(SubArray<D, T, DeviceType> v,
                             SubArray<D, T, DeviceType> coarse,
                             SubArray<1, Q, DeviceType> quantized_coeff,
                             T quantizer,
                             SubArray<1, T, DeviceType> block_quantizers,
                             bool use_block_quantizers, bool prep_huffman,
                             SIZE dict_size, bool orthogonal_projection = true)
      : v(v), coarse(coarse), quantized_coeff(quantized_coeff),
        quantizer(quantizer), block_quantizers(block_quantizers),
        use_block_quantizers(use_block_quantizers), prep_huffman(prep_huffman),
        dict_size(dict_size), orthogonal_projection(orthogonal_projection) {}

  MGARDX_CONT Task<DecomposeQuantize8x8Functor<D, T, Q, 8, 8, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = DecomposeQuantize8x8Functor<D, T, Q, 8, 8, DeviceType>;
    FunctorType functor(v, coarse, quantized_coeff, quantizer, block_quantizers,
                        use_block_quantizers, prep_huffman, dict_size,
                        orthogonal_projection);

    // Same launch geometry as Decompose8x8Kernel; v may be unpadded here but
    // ceil(shape / 8) matches the padded grid exactly.
    SIZE total_thread_y = v.shape(D - 2);
    SIZE total_thread_x = v.shape(D - 1);

    size_t sm_size = functor.shared_memory_size();
    SIZE tbz = 1, tby = 8, tbx = 8;
    SIZE gridz = 1;
    SIZE gridy = ceil((double)total_thread_y / tby);
    SIZE gridx = ceil((double)total_thread_x / tbx);

    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, Q, DeviceType> quantized_coeff;
  T quantizer;
  SubArray<1, T, DeviceType> block_quantizers;
  bool use_block_quantizers;
  bool prep_huffman;
  SIZE dict_size;
  bool orthogonal_projection;
};

} // namespace in_cache_block

} // namespace data_refactoring

} // namespace mgard_x_160

#endif
