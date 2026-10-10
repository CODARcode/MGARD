/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_BLOCK_DELTA_FUSED_HPP
#define MGARD_X_160_BLOCK_DELTA_FUSED_HPP

#include "BlockDeltaKernels.hpp" // block_delta:: helpers + RuntimeX types/macros

// Single-kernel (decoupled look-back) implementation of BlockDelta for the GPU
// backends. The cross-block byte-offset prefix sum -- the one global dependency
// in the scheme -- is resolved inside one kernel launch via a Merrill/Garland
// style decoupled look-back, instead of a separate ScanSumExtended pass.
//
// Mapping: one thread per data block. Each thread acquires a *dynamic* tile id
// via atomicAdd so logical order matches dispatch order (this is what keeps the
// look-back deadlock-free under the GPU's non-preemptive scheduling -- see the
// design discussion). It then computes its block's byte count, publishes it,
// looks back to obtain its exclusive prefix (= its byte offset), and packs.
//
// Intra-block work is serial per thread (same as the portable path); the point
// here is the fused cross-block scan, not intra-block parallelism. The byte
// layout is identical to the portable path, so streams are cross-compatible.

#if defined(MGARDX_COMPILE_CUDA) || defined(MGARDX_COMPILE_HIP)

namespace mgard_x_160 {
namespace block_delta_fused {

// Per-tile status word: top 2 bits = state, low 62 bits = value (a byte count
// or a prefix sum, both well within 62 bits). State INVALID == 0 so a zeroed
// status array starts "not ready".
enum : unsigned long long {
  ST_INVALID = 0ULL,
  ST_AGGREGATE = 1ULL,
  ST_PREFIX = 2ULL
};
__device__ __forceinline__ unsigned long long
make_status(unsigned long long state, unsigned long long val) {
  return (state << 62) | (val & ((1ULL << 62) - 1));
}
__device__ __forceinline__ unsigned long long st_state(unsigned long long s) {
  return s >> 62;
}
__device__ __forceinline__ unsigned long long st_val(unsigned long long s) {
  return s & ((1ULL << 62) - 1);
}

// Exclusive prefix of byte counts for tile `b` via decoupled look-back.
// `bc` is this tile's own byte count. Publishes AGGREGATE before walking back,
// then PREFIX once the exclusive prefix is known.
__device__ __forceinline__ unsigned long long
lookback_exclusive_prefix(volatile unsigned long long *status, unsigned int b,
                          unsigned long long bc) {
  status[b] = make_status(ST_AGGREGATE, bc);
  __threadfence();
  unsigned long long excl = 0;
  long look = (long)b - 1;
  while (look >= 0) {
    unsigned long long s;
    do {
      s = status[look];
    } while (st_state(s) == ST_INVALID);
    excl += st_val(s);
    if (st_state(s) == ST_PREFIX)
      break; // PREFIX value already folds in everything <= look
    look--;
  }
  status[b] = make_status(ST_PREFIX, excl + bc);
  __threadfence();
  return excl;
}

template <typename T>
__global__ void
encode_kernel(const T *__restrict__ data, SIZE n, SIZE block_size, SIZE nblocks,
              bool use_delta, Byte *__restrict__ bitwidth,
              Byte *__restrict__ packed, volatile unsigned long long *status,
              unsigned int *counter, unsigned long long *total) {
  using UT = typename std::make_unsigned<T>::type;
  unsigned int b = atomicAdd(counter, 1u);
  if (b >= nblocks)
    return;
  SIZE start = b * block_size;
  SIZE len = block_size < (n - start) ? block_size : (n - start);

  // bit-width over (delta+)zigzag
  T prev = 0;
  UT acc = 0;
  for (SIZE i = 0; i < len; i++) {
    T x = data[start + i];
    acc |= block_delta::zigzag<T>(use_delta ? (T)(x - prev) : x);
    prev = x;
  }
  int bw = block_delta::bit_length<UT>(acc);
  bitwidth[b] = (Byte)bw;
  unsigned long long bc = (unsigned long long)block_delta::block_bytes(bw, len);

  // fused cross-block scan
  unsigned long long excl = lookback_exclusive_prefix(status, b, bc);

  // pack into our (disjoint) byte range
  Byte *out = packed + excl;
  prev = 0;
  UT buf = 0;
  int cnt = 0;
  size_t pos = 0;
  for (SIZE i = 0; i < len; i++) {
    T x = data[start + i];
    UT z = block_delta::zigzag<T>(use_delta ? (T)(x - prev) : x);
    prev = x;
    for (int k = 0; k < bw; k++) {
      buf |= (UT)((z >> k) & 1) << cnt;
      if (++cnt == 8) {
        out[pos++] = (Byte)(buf & 0xff);
        buf = 0;
        cnt = 0;
      }
    }
  }
  if (cnt > 0)
    out[pos++] = (Byte)(buf & 0xff);

  if (b == nblocks - 1)
    *total = excl + bc; // last tile holds the full packed size
}

template <typename T>
__global__ void
decode_kernel(const Byte *__restrict__ packed, SIZE n, SIZE block_size,
              SIZE nblocks, bool use_delta, const Byte *__restrict__ bitwidth,
              T *__restrict__ data, volatile unsigned long long *status,
              unsigned int *counter) {
  using UT = typename std::make_unsigned<T>::type;
  unsigned int b = atomicAdd(counter, 1u);
  if (b >= nblocks)
    return;
  SIZE start = b * block_size;
  SIZE len = block_size < (n - start) ? block_size : (n - start);
  int bw = (int)bitwidth[b];
  unsigned long long bc = (unsigned long long)block_delta::block_bytes(bw, len);

  unsigned long long excl = lookback_exclusive_prefix(status, b, bc);

  const Byte *in = packed + excl;
  T prev = 0;
  Byte cur = 0;
  int cnt = 0;
  size_t pos = 0;
  for (SIZE i = 0; i < len; i++) {
    UT z = 0;
    for (int k = 0; k < bw; k++) {
      if (cnt == 0) {
        cur = in[pos++];
        cnt = 8;
      }
      z |= (UT)(cur & 1) << k;
      cur >>= 1;
      cnt--;
    }
    T d = block_delta::unzigzag<T>(z);
    prev = use_delta ? (T)(prev + d) : d;
    data[start + i] = prev;
  }
}

#ifdef MGARDX_COMPILE_CUDA
using gpuStream_t = cudaStream_t;
#else
using gpuStream_t = hipStream_t;
#endif

template <typename T>
inline void launch_encode(const T *data, SIZE n, SIZE block_size, SIZE nblocks,
                          bool use_delta, Byte *bitwidth, Byte *packed,
                          unsigned long long *status, unsigned int *counter,
                          unsigned long long *total, gpuStream_t stream) {
  SIZE tpb = 256;
  SIZE grid = (nblocks - 1) / tpb + 1;
  encode_kernel<T><<<grid, tpb, 0, stream>>>(
      data, n, block_size, nblocks, use_delta, bitwidth, packed,
      (volatile unsigned long long *)status, counter, total);
}

template <typename T>
inline void launch_decode(const Byte *packed, SIZE n, SIZE block_size,
                          SIZE nblocks, bool use_delta, const Byte *bitwidth,
                          T *data, unsigned long long *status,
                          unsigned int *counter, gpuStream_t stream) {
  SIZE tpb = 256;
  SIZE grid = (nblocks - 1) / tpb + 1;
  decode_kernel<T><<<grid, tpb, 0, stream>>>(
      packed, n, block_size, nblocks, use_delta, bitwidth, data,
      (volatile unsigned long long *)status, counter);
}

} // namespace block_delta_fused
} // namespace mgard_x_160

#endif // CUDA || HIP

#endif
