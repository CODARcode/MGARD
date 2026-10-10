/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_RANS_ENCODE_TEMPLATE_HPP
#define MGARD_X_160_RANS_ENCODE_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"
#include "RansCommon.hpp"

namespace mgard_x_160 {
namespace rans {

// One independent rANS stream per segment (interleaving across threads gives
// the parallelism). Thread p encodes input[p*seg_size, ...) in reverse into a
// private scratch region [p*seg_capacity, (p+1)*seg_capacity), filling it
// back-to-front, then records the produced byte length. Bytes live at the high
// end of the region: [seg_capacity - seg_len, seg_capacity).
template <typename Q, typename DeviceType>
class EncodeFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT EncodeFunctor() {}
  MGARDX_CONT
  EncodeFunctor(SubArray<1, Q, DeviceType> input, SIZE n, SIZE segment_size,
                SIZE num_segments, SubArray<1, RansEncPacked, DeviceType> esym,
                IDX seg_capacity, SubArray<1, Byte, DeviceType> scratch,
                SubArray<1, uint32_t, DeviceType> seg_len)
      : input(input), n(n), segment_size(segment_size),
        num_segments(num_segments), esym(esym), seg_capacity(seg_capacity),
        scratch(scratch), seg_len(seg_len) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    IDX start = FunctorBase<DeviceType>::GetBlockIdX() *
                    FunctorBase<DeviceType>::GetBlockDimX() +
                FunctorBase<DeviceType>::GetThreadIdX();
    IDX grid_size = FunctorBase<DeviceType>::GetGridDimX() *
                    FunctorBase<DeviceType>::GetBlockDimX();

    for (IDX p = start; p < num_segments; p += grid_size) {
      // Interleaved symbol mapping: consecutive streams (= consecutive warp
      // lanes) read consecutive positions, so the input loads coalesce.
      IDX base_pos = RansStreamBase(p, segment_size);
      IDX count = RansStreamCount(p, segment_size, n);

      IDX base = p * seg_capacity;
      IDX ptr = base + seg_capacity; // cursor, fills downward

      uint32_t x = RANS_L;
      for (IDX j = count; j > 0; j--) {
        IDX pos = base_pos + (j - 1) * RANS_NLANES;
        uint32_t s = (uint32_t)(*input(pos));
        RansEncPacked e = *esym(s); // single 16-byte load of all constants
        x = RansEncPut<DeviceType>(x, e.x_max, e.rcp_freq, e.bias,
                                   (uint32_t)e.cmpl_freq, (uint32_t)e.rcp_shift,
                                   scratch, ptr);
      }
      RansEncFlush<DeviceType>(x, scratch, ptr);

      *seg_len(p) = (uint32_t)(base + seg_capacity - ptr);
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, Q, DeviceType> input;
  SIZE n;
  SIZE segment_size;
  SIZE num_segments;
  SubArray<1, RansEncPacked, DeviceType> esym;
  IDX seg_capacity;
  SubArray<1, Byte, DeviceType> scratch;
  SubArray<1, uint32_t, DeviceType> seg_len;
};

template <typename Q, typename DeviceType> class EncodeKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "rans encode";
  MGARDX_CONT
  EncodeKernel(SubArray<1, Q, DeviceType> input, SIZE n, SIZE segment_size,
               SIZE num_segments, SubArray<1, RansEncPacked, DeviceType> esym,
               IDX seg_capacity, SubArray<1, Byte, DeviceType> scratch,
               SubArray<1, uint32_t, DeviceType> seg_len)
      : input(input), n(n), segment_size(segment_size),
        num_segments(num_segments), esym(esym), seg_capacity(seg_capacity),
        scratch(scratch), seg_len(seg_len) {}

  MGARDX_CONT Task<EncodeFunctor<Q, DeviceType>> GenTask(int queue_idx) {
    using FunctorType = EncodeFunctor<Q, DeviceType>;
    FunctorType functor(input, n, segment_size, num_segments, esym,
                        seg_capacity, scratch, seg_len);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    // One thread per segment, full grid (no repeat_factor cap): the per-thread
    // rANS loop is a long serial dependent chain, so we want maximum occupancy
    // to hide its latency rather than fewer threads each doing more segments.
    gridx = (num_segments - 1) / tbx + 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(), gridx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, Q, DeviceType> input;
  SIZE n;
  SIZE segment_size;
  SIZE num_segments;
  SubArray<1, RansEncPacked, DeviceType> esym;
  IDX seg_capacity;
  SubArray<1, Byte, DeviceType> scratch;
  SubArray<1, uint32_t, DeviceType> seg_len;
};

// Copy each segment's bytes out of its scratch region (back-to-front layout)
// into a single contiguous stream at seg_offset[p]. One block per segment.
template <typename DeviceType>
class CompactFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT CompactFunctor() {}
  MGARDX_CONT CompactFunctor(SubArray<1, Byte, DeviceType> scratch,
                             SubArray<1, uint32_t, DeviceType> seg_len,
                             SubArray<1, uint32_t, DeviceType> seg_offset,
                             SIZE num_segments, IDX seg_capacity,
                             SubArray<1, Byte, DeviceType> stream)
      : scratch(scratch), seg_len(seg_len), seg_offset(seg_offset),
        num_segments(num_segments), seg_capacity(seg_capacity), stream(stream) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    for (IDX p = FunctorBase<DeviceType>::GetBlockIdX(); p < num_segments;
         p += FunctorBase<DeviceType>::GetGridDimX()) {
      uint32_t len = *seg_len(p);
      IDX src = p * seg_capacity + (seg_capacity - (IDX)len);
      IDX dst = (IDX)(*seg_offset(p));
      for (IDX j = FunctorBase<DeviceType>::GetThreadIdX(); j < len;
           j += FunctorBase<DeviceType>::GetBlockDimX()) {
        *stream(dst + j) = *scratch(src + j);
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, Byte, DeviceType> scratch;
  SubArray<1, uint32_t, DeviceType> seg_len;
  SubArray<1, uint32_t, DeviceType> seg_offset;
  SIZE num_segments;
  IDX seg_capacity;
  SubArray<1, Byte, DeviceType> stream;
};

template <typename DeviceType> class CompactKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "rans compact";
  MGARDX_CONT
  CompactKernel(SubArray<1, Byte, DeviceType> scratch,
                SubArray<1, uint32_t, DeviceType> seg_len,
                SubArray<1, uint32_t, DeviceType> seg_offset, SIZE num_segments,
                IDX seg_capacity, SubArray<1, Byte, DeviceType> stream)
      : scratch(scratch), seg_len(seg_len), seg_offset(seg_offset),
        num_segments(num_segments), seg_capacity(seg_capacity), stream(stream) {
  }

  MGARDX_CONT Task<CompactFunctor<DeviceType>> GenTask(int queue_idx) {
    using FunctorType = CompactFunctor<DeviceType>;
    FunctorType functor(scratch, seg_len, seg_offset, num_segments,
                        seg_capacity, stream);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(),
                     (SIZE)num_segments);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, Byte, DeviceType> scratch;
  SubArray<1, uint32_t, DeviceType> seg_len;
  SubArray<1, uint32_t, DeviceType> seg_offset;
  SIZE num_segments;
  IDX seg_capacity;
  SubArray<1, Byte, DeviceType> stream;
};

} // namespace rans
} // namespace mgard_x_160
#endif
