/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_RECOMPOSE_8_KERNEL_TEMPLATE
#define MGARD_X_160_RECOMPOSE_8_KERNEL_TEMPLATE

#include "../../RuntimeX/RuntimeX.h"

#include "IndexTableLowDim.hpp"

namespace mgard_x_160 {

namespace data_refactoring {

namespace in_cache_block {

// Exact inverse of Decompose8Functor: same stages in reverse, with the
// correction subtracted instead of added and the interpolation added back.
template <DIM D, typename T, SIZE TILES, SIZE X, typename DeviceType>
class Recompose8Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Recompose8Functor() {}
  MGARDX_CONT Recompose8Functor(SubArray<D, T, DeviceType> v,
                                SubArray<D, T, DeviceType> coarse,
                                SubArray<1, T, DeviceType> coeff,
                                bool orthogonal_projection = true)
      : v(v), coarse(coarse), coeff(coeff),
        orthogonal_projection(orthogonal_projection) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void initialize_sm_8() {
    sm_v = (T *)FunctorBase<DeviceType>::GetSharedMemory();
    sm_x = sm_v + TILES * SMV_STRIDE_8;
  }

  MGARDX_EXEC void init_indices() {
    item = FunctorBase<DeviceType>::GetThreadIdX();
    tile = FunctorBase<DeviceType>::GetThreadIdY();
    bid = FunctorBase<DeviceType>::GetBlockIdX() * TILES + tile;
    num_tiles = (v.shape(D - 1) + X - 1) / X;
    active = bid < (int)num_tiles;
    base_v = tile * SMV_STRIDE_8;
    base_x = tile * SMX_STRIDE_8;
    x_gl = bid * X + item;
    if (item == 0)
      sm_v[base_v + SMV_ZERO_8] = (T)0;
  }

  // Load
  MGARDX_EXEC void Operation1() {
    initialize_sm_8();
    init_indices();
    if (!active)
      return;
    if (item < NumCoarse_8) {
      sm_v[base_v + Coarse_Offset_8(item)] =
          *coarse(bid * LowDim_Coarse + item);
    } else {
      int op_tid = item - NumCoarse_8;
      sm_v[base_v + Coeff_Offset_8(op_tid)] = *coeff(bid * NumCoeff_8 + op_tid);
    }
  }

  // MassTransX
  MGARDX_EXEC void Operation2() {
    if (!orthogonal_projection)
      return;
    if (active && item < NumMassTransX_8) {
      int const *index = MassTrans_X_Offset_8(item);
      T a = sm_v[base_v + index[0]];
      T b = sm_v[base_v + index[1]];
      T c = sm_v[base_v + index[2]];
      T d = sm_v[base_v + index[3]];
      T e = sm_v[base_v + index[4]];
      T const *dist = MassTrans_Weights_8x8x8<T>(index[6]);
      sm_x[base_x + index[5]] =
          a * dist[0] + b * dist[1] + c * dist[2] + d * dist[3] + e * dist[4];
    }
  }

  MGARDX_EXEC void solve_tridiag() {
    T a = sm_x[base_x + 0];
    T b = sm_x[base_x + 1];
    T c = sm_x[base_x + 2];
    T d = sm_x[base_x + 3];
    T e = sm_x[base_x + 4];

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

    sm_x[base_x + 0] = a;
    sm_x[base_x + 1] = b;
    sm_x[base_x + 2] = c;
    sm_x[base_x + 3] = d;
    sm_x[base_x + 4] = e;
  }

  // TridiagX
  MGARDX_EXEC void Operation3() {
    if (!orthogonal_projection)
      return;
    if (active && item == 0) {
      solve_tridiag();
    }
  }

  // Subtract correction
  MGARDX_EXEC void Operation4() {
    if (orthogonal_projection && active && item < NumCoarse_8) {
      sm_v[base_v + Coarse_Offset_8(item)] -= sm_x[base_x + item];
    }
  }

  // Undo interpolation
  MGARDX_EXEC void Operation5() {
    if (active && item < NumCoeff_8) {
      int const *index = Coeff1D_Offset_8(item);
      T middle = sm_v[base_v + index[0]];
      T left = sm_v[base_v + index[1]];
      T right = sm_v[base_v + index[2]];
      sm_v[base_v + index[0]] = middle + (left + right) * (T)0.5;
    }
  }

  // Store
  MGARDX_EXEC void Operation6() {
    if (active) {
      *v(x_gl) = sm_v[base_v + item];
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    return (size_t)TILES * (SMV_STRIDE_8 + SMX_STRIDE_8) * sizeof(T);
  }

protected:
  SubArray<D, T, DeviceType> v;
  SubArray<D, T, DeviceType> coarse;
  SubArray<1, T, DeviceType> coeff;
  bool orthogonal_projection;
  T *sm_v, *sm_x;
  int item, tile, bid, x_gl;
  int base_v, base_x;
  SIZE num_tiles;
  bool active;
};

// Fused dequantize+recompose variant. See RecomposeDequantize8x8x8Functor:
// the 3 per-tile coefficients are read as Q symbols and dequantized while
// being staged into shared memory, and the store is bounds checked so an
// unpadded destination can be filled directly.
template <DIM D, typename T, typename Q, SIZE TILES, SIZE X,
          typename DeviceType>
class RecomposeDequantize8Functor
    : public Recompose8Functor<D, T, TILES, X, DeviceType> {
  using Base = Recompose8Functor<D, T, TILES, X, DeviceType>;

public:
  MGARDX_CONT RecomposeDequantize8Functor() {}
  MGARDX_CONT RecomposeDequantize8Functor(
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
    this->initialize_sm_8();
    this->init_indices();
    if (!this->active)
      return;
    if (this->item < NumCoarse_8) {
      this->sm_v[this->base_v + Coarse_Offset_8(this->item)] =
          *this->coarse(this->bid * LowDim_Coarse + this->item);
    } else {
      int op_tid = this->item - NumCoarse_8;
      Q quantized_data = *quantized_coeff(this->bid * NumCoeff_8 + op_tid);
      if (prep_huffman) {
        quantized_data -= dict_size / 2;
      }
      T q = use_block_quantizers ? *block_quantizers(this->bid) : quantizer;
      // Must stay bit-identical to QuantizeLocalLevelFunctor (volume == 1,
      // non-reciprocal quantizer).
      this->sm_v[this->base_v + Coeff_Offset_8(op_tid)] = q * (T)quantized_data;
    }
  }

  MGARDX_EXEC void Operation6() {
    if (this->active && this->x_gl < (int)this->v.shape(D - 1)) {
      *this->v(this->x_gl) = this->sm_v[this->base_v + this->item];
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
class Recompose8Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lwpk_1d";
  MGARDX_CONT
  Recompose8Kernel(SubArray<D, T, DeviceType> v,
                   SubArray<D, T, DeviceType> coarse,
                   SubArray<1, T, DeviceType> coeff,
                   bool orthogonal_projection = true)
      : v(v), coarse(coarse), coeff(coeff),
        orthogonal_projection(orthogonal_projection) {}

  MGARDX_CONT Task<Recompose8Functor<D, T, LowDim_Tiles_1D, 8, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = Recompose8Functor<D, T, LowDim_Tiles_1D, 8, DeviceType>;
    FunctorType functor(v, coarse, coeff, orthogonal_projection);

    SIZE num_tiles = (v.shape(D - 1) + 7) / 8;
    size_t sm_size = functor.shared_memory_size();
    SIZE tbz = 1, tby = LowDim_Tiles_1D, tbx = 8;
    SIZE gridz = 1, gridy = 1;
    SIZE gridx = ceil((double)num_tiles / LowDim_Tiles_1D);

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
class RecomposeDequantize8Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lwpk_1d_fq";
  MGARDX_CONT
  RecomposeDequantize8Kernel(SubArray<D, T, DeviceType> v,
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

  MGARDX_CONT
  Task<RecomposeDequantize8Functor<D, T, Q, LowDim_Tiles_1D, 8, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType =
        RecomposeDequantize8Functor<D, T, Q, LowDim_Tiles_1D, 8, DeviceType>;
    FunctorType functor(v, coarse, quantized_coeff, quantizer, block_quantizers,
                        use_block_quantizers, prep_huffman, dict_size,
                        orthogonal_projection);

    SIZE num_tiles = (v.shape(D - 1) + 7) / 8;
    size_t sm_size = functor.shared_memory_size();
    SIZE tbz = 1, tby = LowDim_Tiles_1D, tbx = 8;
    SIZE gridz = 1, gridy = 1;
    SIZE gridx = ceil((double)num_tiles / LowDim_Tiles_1D);

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
