/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_RANS_INTERLEAVED_TEMPLATE_HPP
#define MGARD_X_160_RANS_INTERLEAVED_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"
#include "RansCommon.hpp"

namespace mgard_x_160 {
namespace rans {

// Shared-stream interleaved rANS: a "block" of RANS_NLANES independent rANS
// states shares ONE byte stream, with the states' renorm bytes intermixed in
// LIFO emission order (standard interleaved rANS, generalized to NLANES
// states). This is the layout required to coalesce the encode byte writes (the
// CUDA warp-cooperative kernels below process one block per warp and write the
// per-step bytes coalesced). The SEQUENTIAL reference functors here (one thread
// per block) define the canonical byte order and are the correctness oracle /
// portable fallback; they produce a bitstream identical to the warp kernels.
//
// Block b, lane L owns symbol positions  b*NLANES*S + L + j*NLANES,  j < count,
// where S = symbols-per-lane (segment_size). Encode order: j high->low, then
// L = 0..NLANES-1; states flushed L = 0..NLANES-1 at the end (lowest address).
// Decode mirrors: init states L = NLANES-1..0, then j low->high, L =
// NLANES-1..0.

template <typename DeviceType>
MGARDX_EXEC IDX RansBlockLaneBase(IDX b, IDX L, IDX S) {
  return b * (RANS_NLANES * S) + L;
}
template <typename DeviceType>
MGARDX_EXEC IDX RansBlockLaneCount(IDX b, IDX L, IDX S, IDX n) {
  IDX base = b * (RANS_NLANES * S) + L;
  if (base >= n) {
    return 0;
  }
  IDX count = (n - 1 - base) / RANS_NLANES + 1;
  return count > S ? S : count;
}

// ---------------------------------------------------------------------------
// Sequential reference encode: one thread per block.
// ---------------------------------------------------------------------------
template <typename Q, typename DeviceType>
class InterleavedEncodeFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT InterleavedEncodeFunctor() {}
  MGARDX_CONT InterleavedEncodeFunctor(
      SubArray<1, Q, DeviceType> input, SIZE n, SIZE segment_size,
      SIZE num_blocks, SubArray<1, RansEncPacked, DeviceType> esym,
      IDX block_capacity, SubArray<1, Byte, DeviceType> scratch,
      SubArray<1, uint32_t, DeviceType> seg_len)
      : input(input), n(n), segment_size(segment_size), num_blocks(num_blocks),
        esym(esym), block_capacity(block_capacity), scratch(scratch),
        seg_len(seg_len) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    IDX start = FunctorBase<DeviceType>::GetBlockIdX() *
                    FunctorBase<DeviceType>::GetBlockDimX() +
                FunctorBase<DeviceType>::GetThreadIdX();
    IDX grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                    FunctorBase<DeviceType>::GetBlockDimX();

    for (IDX b = start; b < num_blocks; b += grid_size) {
      uint32_t x[RANS_NLANES];
      for (IDX L = 0; L < RANS_NLANES; L++) {
        x[L] = RANS_L;
      }
      // Lane 0 starts earliest, so it has the most symbols => maxcount.
      IDX maxcount = RansBlockLaneCount<DeviceType>(b, 0, segment_size, n);

      IDX ptr = b * block_capacity + block_capacity; // cursor, fills downward

      for (IDX jj = maxcount; jj > 0; jj--) {
        IDX j = jj - 1;
        for (IDX L = 0; L < RANS_NLANES; L++) {
          IDX base = b * (RANS_NLANES * segment_size) + L;
          IDX count = RansBlockLaneCount<DeviceType>(b, L, segment_size, n);
          if (j < count) {
            IDX pos = base + j * RANS_NLANES;
            uint32_t s = (uint32_t)(*input(pos));
            RansEncPacked e = *esym(s);
            x[L] = RansEncPut<DeviceType>(x[L], e.x_max, e.rcp_freq, e.bias,
                                          (uint32_t)e.cmpl_freq,
                                          (uint32_t)e.rcp_shift, scratch, ptr);
          }
        }
      }
      for (IDX L = 0; L < RANS_NLANES; L++) {
        RansEncFlush<DeviceType>(x[L], scratch, ptr);
      }
      *seg_len(b) = (uint32_t)(b * block_capacity + block_capacity - ptr);
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, Q, DeviceType> input;
  SIZE n;
  SIZE segment_size;
  SIZE num_blocks;
  SubArray<1, RansEncPacked, DeviceType> esym;
  IDX block_capacity;
  SubArray<1, Byte, DeviceType> scratch;
  SubArray<1, uint32_t, DeviceType> seg_len;
};

template <typename Q, typename DeviceType>
class InterleavedEncodeKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "rans interleaved encode";
  MGARDX_CONT InterleavedEncodeKernel(
      SubArray<1, Q, DeviceType> input, SIZE n, SIZE segment_size,
      SIZE num_blocks, SubArray<1, RansEncPacked, DeviceType> esym,
      IDX block_capacity, SubArray<1, Byte, DeviceType> scratch,
      SubArray<1, uint32_t, DeviceType> seg_len)
      : input(input), n(n), segment_size(segment_size), num_blocks(num_blocks),
        esym(esym), block_capacity(block_capacity), scratch(scratch),
        seg_len(seg_len) {}

  MGARDX_CONT Task<InterleavedEncodeFunctor<Q, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = InterleavedEncodeFunctor<Q, DeviceType>;
    FunctorType functor(input, n, segment_size, num_blocks, esym,
                        block_capacity, scratch, seg_len);
    SIZE tbx = 256, gridx;
    gridx = (num_blocks - 1) / tbx + 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(), gridx);
    return Task(functor, 1, 1, gridx, 1, 1, tbx, 0, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, Q, DeviceType> input;
  SIZE n;
  SIZE segment_size;
  SIZE num_blocks;
  SubArray<1, RansEncPacked, DeviceType> esym;
  IDX block_capacity;
  SubArray<1, Byte, DeviceType> scratch;
  SubArray<1, uint32_t, DeviceType> seg_len;
};

// ---------------------------------------------------------------------------
// Sequential reference decode: one thread per block.
// ---------------------------------------------------------------------------
template <typename Q, typename DeviceType>
class InterleavedDecodeFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT InterleavedDecodeFunctor() {}
  MGARDX_CONT InterleavedDecodeFunctor(
      SubArray<1, Byte, DeviceType> stream,
      SubArray<1, uint32_t, DeviceType> seg_offset, SIZE n, SIZE segment_size,
      SIZE num_blocks, SubArray<1, uint32_t, DeviceType> freq,
      SubArray<1, uint32_t, DeviceType> cum,
      SubArray<1, uint16_t, DeviceType> slot2sym, uint32_t scale_bits,
      SubArray<1, Q, DeviceType> output)
      : stream(stream), seg_offset(seg_offset), n(n),
        segment_size(segment_size), num_blocks(num_blocks), freq(freq),
        cum(cum), slot2sym(slot2sym), scale_bits(scale_bits), output(output) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    IDX start = FunctorBase<DeviceType>::GetBlockIdX() *
                    FunctorBase<DeviceType>::GetBlockDimX() +
                FunctorBase<DeviceType>::GetThreadIdX();
    IDX grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                    FunctorBase<DeviceType>::GetBlockDimX();
    uint32_t mask = (1u << scale_bits) - 1;

    for (IDX b = start; b < num_blocks; b += grid_size) {
      IDX rp = (IDX)(*seg_offset(b));
      uint32_t x[RANS_NLANES];
      // Init states in reverse of the flush order (flush wrote L=0..NLANES-1,
      // so the lowest address is L=NLANES-1, read first going forward).
      for (IDX LL = RANS_NLANES; LL > 0; LL--) {
        x[LL - 1] = RansDecInit<DeviceType>(stream, rp);
      }

      IDX maxcount = RansBlockLaneCount<DeviceType>(b, 0, segment_size, n);
      for (IDX j = 0; j < maxcount; j++) {
        for (IDX LL = RANS_NLANES; LL > 0; LL--) {
          IDX L = LL - 1;
          IDX count = RansBlockLaneCount<DeviceType>(b, L, segment_size, n);
          if (j < count) {
            uint32_t slot = x[L] & mask;
            uint32_t s = (uint32_t)(*slot2sym(slot));
            IDX pos = b * (RANS_NLANES * segment_size) + L + j * RANS_NLANES;
            *output(pos) = (Q)s;
            x[L] = RansDecAdvance<DeviceType>(x[L], *freq(s), *cum(s), slot,
                                              scale_bits, stream, rp);
          }
        }
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, Byte, DeviceType> stream;
  SubArray<1, uint32_t, DeviceType> seg_offset;
  SIZE n;
  SIZE segment_size;
  SIZE num_blocks;
  SubArray<1, uint32_t, DeviceType> freq;
  SubArray<1, uint32_t, DeviceType> cum;
  SubArray<1, uint16_t, DeviceType> slot2sym;
  uint32_t scale_bits;
  SubArray<1, Q, DeviceType> output;
};

template <typename Q, typename DeviceType>
class InterleavedDecodeKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "rans interleaved decode";
  MGARDX_CONT InterleavedDecodeKernel(
      SubArray<1, Byte, DeviceType> stream,
      SubArray<1, uint32_t, DeviceType> seg_offset, SIZE n, SIZE segment_size,
      SIZE num_blocks, SubArray<1, uint32_t, DeviceType> freq,
      SubArray<1, uint32_t, DeviceType> cum,
      SubArray<1, uint16_t, DeviceType> slot2sym, uint32_t scale_bits,
      SubArray<1, Q, DeviceType> output)
      : stream(stream), seg_offset(seg_offset), n(n),
        segment_size(segment_size), num_blocks(num_blocks), freq(freq),
        cum(cum), slot2sym(slot2sym), scale_bits(scale_bits), output(output) {}

  MGARDX_CONT Task<InterleavedDecodeFunctor<Q, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = InterleavedDecodeFunctor<Q, DeviceType>;
    FunctorType functor(stream, seg_offset, n, segment_size, num_blocks, freq,
                        cum, slot2sym, scale_bits, output);
    SIZE tbx = 256, gridx;
    gridx = (num_blocks - 1) / tbx + 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(), gridx);
    return Task(functor, 1, 1, gridx, 1, 1, tbx, 0, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, Byte, DeviceType> stream;
  SubArray<1, uint32_t, DeviceType> seg_offset;
  SIZE n;
  SIZE segment_size;
  SIZE num_blocks;
  SubArray<1, uint32_t, DeviceType> freq;
  SubArray<1, uint32_t, DeviceType> cum;
  SubArray<1, uint16_t, DeviceType> slot2sym;
  uint32_t scale_bits;
  SubArray<1, Q, DeviceType> output;
};

// ---------------------------------------------------------------------------
// Warp-cooperative encode: one SubGroup (= RANS_NLANES lanes) per block, lane L
// owns rANS state L. Written against the portable SubGroup abstraction, so the
// per-step renorm bytes of all lanes are gathered with a subgroup prefix-sum
// and written COALESCED, in the exact byte order of the sequential reference
// above (lane 0 at the high end of each step's range). Used only where the
// subgroup size equals RANS_NLANES (CUDA warp); other backends use the
// sequential kernel.
template <typename Q, typename DeviceType>
class InterleavedEncodeWarpFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT InterleavedEncodeWarpFunctor() {}
  MGARDX_CONT InterleavedEncodeWarpFunctor(
      SubArray<1, Q, DeviceType> input, SIZE n, SIZE segment_size,
      SIZE num_blocks, SubArray<1, RansEncPacked, DeviceType> esym,
      IDX block_capacity, SubArray<1, Byte, DeviceType> scratch,
      SubArray<1, uint32_t, DeviceType> seg_len)
      : input(input), n(n), segment_size(segment_size), num_blocks(num_blocks),
        esym(esym), block_capacity(block_capacity), scratch(scratch),
        seg_len(seg_len) {
    Functor<DeviceType>();
  }

  // Exclusive prefix sum of v across ONE RANS_NLANES-lane segment of the
  // subgroup (lanes [seg_base, seg_base+RANS_NLANES)); returns the segment
  // total via the reference parameter. logical_lane is the 0..RANS_NLANES-1
  // position within the segment. A wavefront wider than RANS_NLANES holds
  // several independent segments, each scanning only its own lanes.
  MGARDX_EXEC int SubgroupExclScan(SubGroup<DeviceType> &sg, int logical_lane,
                                   int seg_base, int v, int &total) {
    int incl = v;
    for (int d = 1; d < (int)RANS_NLANES; d <<= 1) {
      int src = (logical_lane - d < 0) ? (seg_base + logical_lane)
                                       : (seg_base + logical_lane - d);
      int t = sg.shfl(incl, src);
      if (logical_lane >= d) {
        incl += t;
      }
    }
    total = sg.shfl(incl, seg_base + (int)RANS_NLANES - 1);
    return incl - v;
  }

  MGARDX_EXEC void Operation1() {
    SubGroup<DeviceType> sg;
    constexpr int W =
        SubGroup<DeviceType>::size(); // 32 (warp) or 64 (wavefront)
    constexpr int SUBBLOCKS =
        W / (int)RANS_NLANES;                // logical blocks per subgroup
    int lane_in_sg = sg.lane();              // 0..W-1
    int seg = lane_in_sg / (int)RANS_NLANES; // which logical block
    int logical_lane = lane_in_sg % (int)RANS_NLANES;
    int seg_base = seg * (int)RANS_NLANES;

    IDX gtid = FunctorBase<DeviceType>::GetBlockIdX() *
                   FunctorBase<DeviceType>::GetBlockDimX() +
               FunctorBase<DeviceType>::GetThreadIdX();
    IDX subgroup_idx = gtid / (IDX)W;
    IDX num_subgroups = (FunctorBase<DeviceType>::GetGridDimX() *
                         FunctorBase<DeviceType>::GetBlockDimX()) /
                        (IDX)W;

    // The loop bound is uniform across the whole subgroup so every lane
    // iterates together (required for the subgroup shuffles), even when the
    // wavefront's several blocks have different lengths.
    for (IDX sgi = subgroup_idx; sgi * (IDX)SUBBLOCKS < num_blocks;
         sgi += num_subgroups) {
      IDX b = sgi * (IDX)SUBBLOCKS + (IDX)seg;
      bool block_active = (b < num_blocks);

      IDX base = b * (RANS_NLANES * segment_size) + (IDX)logical_lane;
      int count = 0;
      if (block_active && base < n) {
        IDX c = (n - 1 - base) / RANS_NLANES + 1;
        count = (int)(c > segment_size ? segment_size : c);
      }
      // Subgroup-wide max symbol count -> uniform inner-loop bound.
      int maxcount = count;
      for (int d = 1; d < W; d <<= 1) {
        int t = sg.shfl(maxcount, lane_in_sg ^ d);
        maxcount = t > maxcount ? t : maxcount;
      }

      uint32_t x = RANS_L;
      IDX ptr = b * block_capacity + block_capacity; // per-block shared cursor

      for (int jj = maxcount; jj > 0; jj--) {
        int j = jj - 1;
        Byte buf[4];
        int nb = 0;
        RansEncPacked e;
        bool active = (j < count); // false for inactive-block lanes
        if (active) {
          IDX pos = base + (IDX)j * RANS_NLANES;
          uint32_t s = (uint32_t)(*input(pos));
          e = *esym(s);
          nb = RansEncCollect(x, e.x_max, buf);
        }
        int total;
        int excl = SubgroupExclScan(sg, logical_lane, seg_base, nb, total);
        for (int k = 0; k < nb; k++) {
          *scratch(ptr - (IDX)excl - 1 - (IDX)k) = buf[k];
        }
        ptr -= (IDX)total;
        if (active) {
          x = RansEncApply(x, e.rcp_freq, e.bias, (uint32_t)e.cmpl_freq,
                           (uint32_t)e.rcp_shift);
        }
        sg.sync();
      }

      // Flush NLANES states (4 LE bytes each), logical lane 0 at the high end.
      // Inactive-block lanes contribute 0 to keep the scan in lockstep, but
      // write nothing.
      int flush_nb = block_active ? 4 : 0;
      int total;
      int excl = SubgroupExclScan(sg, logical_lane, seg_base, flush_nb, total);
      if (block_active) {
        IDX p0 = ptr - (IDX)excl - 4;
        *scratch(p0 + 0) = (Byte)(x >> 0);
        *scratch(p0 + 1) = (Byte)(x >> 8);
        *scratch(p0 + 2) = (Byte)(x >> 16);
        *scratch(p0 + 3) = (Byte)(x >> 24);
        ptr -= (IDX)total;
        if (logical_lane == 0) {
          *seg_len(b) = (uint32_t)(b * block_capacity + block_capacity - ptr);
        }
      }
      sg.sync();
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, Q, DeviceType> input;
  SIZE n;
  SIZE segment_size;
  SIZE num_blocks;
  SubArray<1, RansEncPacked, DeviceType> esym;
  IDX block_capacity;
  SubArray<1, Byte, DeviceType> scratch;
  SubArray<1, uint32_t, DeviceType> seg_len;
};

template <typename Q, typename DeviceType>
class InterleavedEncodeWarpKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "rans interleaved warp encode";
  MGARDX_CONT InterleavedEncodeWarpKernel(
      SubArray<1, Q, DeviceType> input, SIZE n, SIZE segment_size,
      SIZE num_blocks, SubArray<1, RansEncPacked, DeviceType> esym,
      IDX block_capacity, SubArray<1, Byte, DeviceType> scratch,
      SubArray<1, uint32_t, DeviceType> seg_len)
      : input(input), n(n), segment_size(segment_size), num_blocks(num_blocks),
        esym(esym), block_capacity(block_capacity), scratch(scratch),
        seg_len(seg_len) {}

  MGARDX_CONT Task<InterleavedEncodeWarpFunctor<Q, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = InterleavedEncodeWarpFunctor<Q, DeviceType>;
    FunctorType functor(input, n, segment_size, num_blocks, esym,
                        block_capacity, scratch, seg_len);
    SIZE tbx = 256; // 8 subgroups (warps) per block
    SIZE total_threads = num_blocks * RANS_NLANES;
    SIZE gridx = (total_threads - 1) / tbx + 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(), gridx);
    return Task(functor, 1, 1, gridx, 1, 1, tbx, 0, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, Q, DeviceType> input;
  SIZE n;
  SIZE segment_size;
  SIZE num_blocks;
  SubArray<1, RansEncPacked, DeviceType> esym;
  IDX block_capacity;
  SubArray<1, Byte, DeviceType> scratch;
  SubArray<1, uint32_t, DeviceType> seg_len;
};

} // namespace rans
} // namespace mgard_x_160
#endif
