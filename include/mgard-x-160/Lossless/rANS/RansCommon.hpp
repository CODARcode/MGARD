/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_RANS_COMMON_TEMPLATE_HPP
#define MGARD_X_160_RANS_COMMON_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x_160 {
namespace rans {

// Static range-ANS (rANS) over a byte alphabet (256 symbols), 32-bit state,
// byte-wise renormalization. The normalization interval is [RANS_L, RANS_L*256)
// so the state always fits in uint32 (RANS_L*256 == 2^31). Encoding writes
// bytes back-to-front (decreasing pointer); decoding reads them front-to-back.
// These primitives follow the well-known ryg_rans construction.

static constexpr uint32_t RANS_L = 1u << 23; // lower bound of normalization
static constexpr int RANS_ALPHABET = 256;

// Warp-interleaving factor. Streams are grouped into blocks of RANS_NLANES
// consecutive streams that interleave over a contiguous super-segment, so that
// at each step the RANS_NLANES streams (= consecutive threads in a warp) touch
// RANS_NLANES consecutive input/output positions -> coalesced memory access,
// with no extra per-stream metadata (same stream/flush/offset count as the
// non-interleaved layout). Stream p covers symbol positions
//   (p / RANS_NLANES) * (RANS_NLANES * S) + (p % RANS_NLANES) + j * RANS_NLANES
// for j = 0..count_p-1, where S = symbols-per-stream (segment_size).
static constexpr IDX RANS_NLANES = 32;

// Number of symbol positions assigned to interleaved stream p, given total n
// and S symbols-per-stream. Clamped to S so a stream never steals the next
// block's symbols.
MGARDX_CONT_EXEC IDX RansStreamBase(IDX p, IDX S) {
  return (p / RANS_NLANES) * (RANS_NLANES * S) + (p % RANS_NLANES);
}
MGARDX_CONT_EXEC IDX RansStreamCount(IDX p, IDX S, IDX n) {
  IDX base = RansStreamBase(p, S);
  if (base >= n) {
    return 0;
  }
  IDX count = (n - 1 - base) / RANS_NLANES + 1;
  return count > S ? S : count;
}

// Per-symbol precomputed encode constants (Alverson reciprocal, ryg_rans). They
// turn the encode step's integer division (x / freq, x % freq) into a
// multiply-high + shift, which is far cheaper on the GPU and is the encode
// bottleneck fix. Built once per frequency table on the host (see Rans.hpp).
//   x_max     = ((RANS_L >> scale_bits) << 8) * freq   (renorm threshold)
//   rcp_freq  = fixed-point reciprocal of freq
//   rcp_shift = reciprocal shift
//   bias      = additive bias
//   cmpl_freq = (1 << scale_bits) - freq
struct RansEncSymbol {
  uint32_t x_max;
  uint32_t rcp_freq;
  uint32_t bias;
  uint32_t cmpl_freq;
  uint32_t rcp_shift;
};

// Packed 16-byte encode-table entry so the hot encode loop fetches all
// per-symbol constants in a single (uint4) load instead of 5 separate array
// reads. The encode loop is compute/latency-bound, so cutting table loads
// 5x->1x is the main encode speedup. Internal to encode only (not part of the
// bitstream).
struct alignas(16) RansEncPacked {
  uint32_t x_max;
  uint32_t rcp_freq;
  uint32_t bias;
  uint16_t cmpl_freq;
  uint16_t rcp_shift;
};

// Host-side: fill the precomputed encode constants for one symbol. Mirrors
// ryg_rans RansEncSymbolInit; produces a bitstream identical to the division
// form, so the decoder is unchanged.
MGARDX_CONT void RansEncSymbolInit(RansEncSymbol &s, uint32_t cum,
                                   uint32_t freq, uint32_t scale_bits) {
  s.x_max = ((RANS_L >> scale_bits) << 8) * freq;
  s.cmpl_freq = (1u << scale_bits) - freq;
  if (freq < 2) {
    s.rcp_freq = ~0u;
    s.rcp_shift = 0;
    s.bias = cum + (1u << scale_bits) - 1;
  } else {
    uint32_t shift = 0;
    while (freq > (1u << shift)) {
      shift++;
    }
    s.rcp_freq = (uint32_t)(((1ull << (shift + 31)) + freq - 1) / freq);
    s.rcp_shift = shift - 1;
    s.bias = cum;
  }
}

// Split form of the encode step for warp-cooperative encoding, where the
// renorm bytes of all lanes in a step must be gathered and written coalesced
// (so they cannot be written inline). RansEncCollect reduces x and returns its
// renorm bytes (LSB first) in buf; RansEncApply then folds in the symbol.
MGARDX_CONT_EXEC int RansEncCollect(uint32_t &x, uint32_t x_max, Byte buf[4]) {
  int nb = 0;
  while (x >= x_max) {
    buf[nb] = (Byte)(x & 0xFFu);
    nb++;
    x >>= 8;
  }
  return nb;
}
MGARDX_CONT_EXEC uint32_t RansEncApply(uint32_t x, uint32_t rcp_freq,
                                       uint32_t bias, uint32_t cmpl_freq,
                                       uint32_t rcp_shift) {
  uint32_t q = (uint32_t)(((uint64_t)x * rcp_freq) >> 32);
  return x + bias + (q >> rcp_shift) * cmpl_freq;
}

// Encode one symbol with the precomputed constants: renormalize by flushing low
// bytes (back-to-front via the decreasing cursor ptr), then advance the state
// via reciprocal multiply instead of division.
template <typename DeviceType>
MGARDX_EXEC uint32_t RansEncPut(uint32_t x, uint32_t x_max, uint32_t rcp_freq,
                                uint32_t bias, uint32_t cmpl_freq,
                                uint32_t rcp_shift,
                                SubArray<1, Byte, DeviceType> &out, IDX &ptr) {
  if (x >= x_max) {
    do {
      --ptr;
      *out(ptr) = (Byte)(x & 0xFFu);
      x >>= 8;
    } while (x >= x_max);
  }
  uint32_t q = (uint32_t)(((uint64_t)x * rcp_freq) >> 32);
  return x + bias + (q >> rcp_shift) * cmpl_freq;
}

// Flush the final 32-bit state as 4 little-endian bytes at the front of the
// (back-to-front) segment buffer.
template <typename DeviceType>
MGARDX_EXEC void RansEncFlush(uint32_t x, SubArray<1, Byte, DeviceType> &out,
                              IDX &ptr) {
  ptr -= 4;
  *out(ptr + 0) = (Byte)(x >> 0);
  *out(ptr + 1) = (Byte)(x >> 8);
  *out(ptr + 2) = (Byte)(x >> 16);
  *out(ptr + 3) = (Byte)(x >> 24);
}

// Read the 4-byte little-endian state that RansEncFlush wrote.
template <typename DeviceType>
MGARDX_EXEC uint32_t RansDecInit(SubArray<1, Byte, DeviceType> &in, IDX &rp) {
  uint32_t x = (uint32_t)(*in(rp + 0)) | ((uint32_t)(*in(rp + 1)) << 8) |
               ((uint32_t)(*in(rp + 2)) << 16) |
               ((uint32_t)(*in(rp + 3)) << 24);
  rp += 4;
  return x;
}

// Given the current slot (= x & mask) and the decoded symbol's (freq, cum),
// advance the state and renormalize by pulling bytes front-to-back.
template <typename DeviceType>
MGARDX_EXEC uint32_t RansDecAdvance(uint32_t x, uint32_t freq, uint32_t cum,
                                    uint32_t slot, uint32_t scale_bits,
                                    SubArray<1, Byte, DeviceType> &in,
                                    IDX &rp) {
  x = freq * (x >> scale_bits) + slot - cum;
  while (x < RANS_L) {
    x = (x << 8) | (uint32_t)(*in(rp));
    ++rp;
  }
  return x;
}

} // namespace rans
} // namespace mgard_x_160
#endif
