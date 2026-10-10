/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_DECOMPOSE_8_KERNEL_TEMPLATE
#define MGARD_X_160_DECOMPOSE_8_KERNEL_TEMPLATE

#include "../../RuntimeX/RuntimeX.h"

#include "IndexTableLowDim.hpp"

namespace mgard_x_160 {

namespace data_refactoring {

namespace in_cache_block {

/*
1D counterpart of Decompose8x8x8Functor:

v       x       c            total
8       5(cor)  8-5(3)       13 per tile

A single 8-element tile is far too little work for a thread block, so each
block owns TILES tiles side by side: threadIdx.y selects the tile and
threadIdx.x the element within it. Every shared-memory table entry is a
within-tile offset, to which the kernel adds the tile's base.

The 3 coefficients per tile are written to coeff[tile * 3 ...] and the 5
coarse values to coarse[tile * 5 ...], where `tile` is the tile's index in
row-major order over the whole array -- the same convention the 3D block uses
for its linearized thread-block id.
*/
template <DIM D, typename T, SIZE TILES, SIZE X, typename DeviceType>
class Decompose8Functor : public Functor<DeviceType> {
public:
  MGARDX_CONT Decompose8Functor() {}
  MGARDX_CONT Decompose8Functor(SubArray<D, T, DeviceType> v,
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
    if (active) {
      sm_v[base_v + item] = *v(x_gl);
    }
  }

  // Interpolation
  MGARDX_EXEC void Operation2() {
    if (active && item < NumCoeff_8) {
      int const *index = Coeff1D_Offset_8(item);
      T middle = sm_v[base_v + index[0]];
      T left = sm_v[base_v + index[1]];
      T right = sm_v[base_v + index[2]];
      sm_v[base_v + index[0]] = middle - (left + right) * (T)0.5;
    }
  }

  // MassTransX
  MGARDX_EXEC void Operation3() {
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
  MGARDX_EXEC void Operation4() {
    if (!orthogonal_projection)
      return;
    if (active && item == 0) {
      solve_tridiag();
    }
  }

  // Apply correction and write out
  MGARDX_EXEC void Operation5() {
    if (!active)
      return;
    if (item < NumCoarse_8) {
      if (orthogonal_projection)
        sm_v[base_v + Coarse_Offset_8(item)] += sm_x[base_x + item];
      *coarse(bid * LowDim_Coarse + item) =
          sm_v[base_v + Coarse_Offset_8(item)];
    } else {
      int op_tid = item - NumCoarse_8;
      *coeff(bid * NumCoeff_8 + op_tid) = sm_v[base_v + Coeff_Offset_8(op_tid)];
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

// Fused decompose+quantize variant. See DecomposeQuantize8x8x8Functor: the
// input is read with bounds checks so unpadded inputs can be consumed
// directly, and the 3 per-tile coefficients are quantized on write-out.
template <DIM D, typename T, typename Q, SIZE TILES, SIZE X,
          typename DeviceType>
class DecomposeQuantize8Functor
    : public Decompose8Functor<D, T, TILES, X, DeviceType> {
  using Base = Decompose8Functor<D, T, TILES, X, DeviceType>;

public:
  MGARDX_CONT DecomposeQuantize8Functor() {}
  MGARDX_CONT DecomposeQuantize8Functor(
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
    if (this->active) {
      this->sm_v[this->base_v + this->item] = (T)0;
      if (this->x_gl < (int)this->v.shape(D - 1)) {
        this->sm_v[this->base_v + this->item] = *this->v(this->x_gl);
      }
    }
  }

  MGARDX_EXEC void Operation5() {
    if (!this->active)
      return;
    if (this->item < NumCoarse_8) {
      if (this->orthogonal_projection)
        this->sm_v[this->base_v + Coarse_Offset_8(this->item)] +=
            this->sm_x[this->base_x + this->item];
      *this->coarse(this->bid * LowDim_Coarse + this->item) =
          this->sm_v[this->base_v + Coarse_Offset_8(this->item)];
    } else {
      int op_tid = this->item - NumCoarse_8;
      T t = this->sm_v[this->base_v + Coeff_Offset_8(op_tid)];
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
      *quantized_coeff(this->bid * NumCoeff_8 + op_tid) = quantized_data;
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
class Decompose8Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lwpk_1d";
  MGARDX_CONT
  Decompose8Kernel(SubArray<D, T, DeviceType> v,
                   SubArray<D, T, DeviceType> coarse,
                   SubArray<1, T, DeviceType> coeff,
                   bool orthogonal_projection = true)
      : v(v), coarse(coarse), coeff(coeff),
        orthogonal_projection(orthogonal_projection) {}

  MGARDX_CONT Task<Decompose8Functor<D, T, LowDim_Tiles_1D, 8, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = Decompose8Functor<D, T, LowDim_Tiles_1D, 8, DeviceType>;
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
class DecomposeQuantize8Kernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "lwpk_1d_fq";
  MGARDX_CONT
  DecomposeQuantize8Kernel(SubArray<D, T, DeviceType> v,
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
  Task<DecomposeQuantize8Functor<D, T, Q, LowDim_Tiles_1D, 8, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType =
        DecomposeQuantize8Functor<D, T, Q, LowDim_Tiles_1D, 8, DeviceType>;
    FunctorType functor(v, coarse, quantized_coeff, quantizer, block_quantizers,
                        use_block_quantizers, prep_huffman, dict_size,
                        orthogonal_projection);

    // Same launch geometry as Decompose8Kernel; v may be unpadded here but
    // ceil(shape / 8) matches the padded tile count exactly.
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
