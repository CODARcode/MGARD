/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_RANS_DECODE_TEMPLATE_HPP
#define MGARD_X_160_RANS_DECODE_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"
#include "RansCommon.hpp"

namespace mgard_x_160 {
namespace rans {

// Decode each segment independently: thread p reads its byte stream starting at
// seg_offset[p], reconstructs the state, then walks symbols forward (the mirror
// of the reverse encode), looking each up via the slot->symbol table and
// scattering into output[p*seg_size, ...).
template <typename Q, typename DeviceType>
class DecodeFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DecodeFunctor() {}
  MGARDX_CONT DecodeFunctor(SubArray<1, Byte, DeviceType> stream,
                            SubArray<1, uint32_t, DeviceType> seg_offset,
                            SIZE n, SIZE segment_size, SIZE num_segments,
                            SubArray<1, uint32_t, DeviceType> freq,
                            SubArray<1, uint32_t, DeviceType> cum,
                            SubArray<1, uint16_t, DeviceType> slot2sym,
                            uint32_t scale_bits,
                            SubArray<1, Q, DeviceType> output)
      : stream(stream), seg_offset(seg_offset), n(n),
        segment_size(segment_size), num_segments(num_segments), freq(freq),
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

    for (IDX p = start; p < num_segments; p += grid_size) {
      // Mirror the interleaved encode mapping: stream p produces symbols for
      // positions base_pos + j*RANS_NLANES (consecutive streams -> consecutive
      // output positions -> coalesced stores).
      IDX base_pos = RansStreamBase(p, segment_size);
      IDX count = RansStreamCount(p, segment_size, n);

      IDX rp = (IDX)(*seg_offset(p));
      uint32_t x = RansDecInit<DeviceType>(stream, rp);

      for (IDX j = 0; j < count; j++) {
        uint32_t slot = x & mask;
        uint32_t s = (uint32_t)(*slot2sym(slot));
        *output(base_pos + j * RANS_NLANES) = (Q)s;
        x = RansDecAdvance<DeviceType>(x, *freq(s), *cum(s), slot, scale_bits,
                                       stream, rp);
      }
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, Byte, DeviceType> stream;
  SubArray<1, uint32_t, DeviceType> seg_offset;
  SIZE n;
  SIZE segment_size;
  SIZE num_segments;
  SubArray<1, uint32_t, DeviceType> freq;
  SubArray<1, uint32_t, DeviceType> cum;
  SubArray<1, uint16_t, DeviceType> slot2sym;
  uint32_t scale_bits;
  SubArray<1, Q, DeviceType> output;
};

template <typename Q, typename DeviceType> class DecodeKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "rans decode";
  MGARDX_CONT
  DecodeKernel(SubArray<1, Byte, DeviceType> stream,
               SubArray<1, uint32_t, DeviceType> seg_offset, SIZE n,
               SIZE segment_size, SIZE num_segments,
               SubArray<1, uint32_t, DeviceType> freq,
               SubArray<1, uint32_t, DeviceType> cum,
               SubArray<1, uint16_t, DeviceType> slot2sym, uint32_t scale_bits,
               SubArray<1, Q, DeviceType> output)
      : stream(stream), seg_offset(seg_offset), n(n),
        segment_size(segment_size), num_segments(num_segments), freq(freq),
        cum(cum), slot2sym(slot2sym), scale_bits(scale_bits), output(output) {}

  MGARDX_CONT Task<DecodeFunctor<Q, DeviceType>> GenTask(int queue_idx) {
    using FunctorType = DecodeFunctor<Q, DeviceType>;
    FunctorType functor(stream, seg_offset, n, segment_size, num_segments, freq,
                        cum, slot2sym, scale_bits, output);

    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = 1;
    tby = 1;
    tbx = 256;
    gridz = 1;
    gridy = 1;
    // One thread per segment, full grid (no repeat_factor cap) to maximize
    // occupancy for the serial per-segment decode chain.
    gridx = (num_segments - 1) / tbx + 1;
    gridx = std::max((SIZE)DeviceRuntime<DeviceType>::GetNumSMs(), gridx);
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, Byte, DeviceType> stream;
  SubArray<1, uint32_t, DeviceType> seg_offset;
  SIZE n;
  SIZE segment_size;
  SIZE num_segments;
  SubArray<1, uint32_t, DeviceType> freq;
  SubArray<1, uint32_t, DeviceType> cum;
  SubArray<1, uint16_t, DeviceType> slot2sym;
  uint32_t scale_bits;
  SubArray<1, Q, DeviceType> output;
};

} // namespace rans
} // namespace mgard_x_160
#endif
