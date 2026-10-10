/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_PARALLEL_DEFLATE_TEMPLATE_HPP
#define MGARD_X_160_PARALLEL_DEFLATE_TEMPLATE_HPP
#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x_160 {

// Number of consecutive symbols packed by a single thread ("group"). Each
// Huffman chunk is split into ceil(chunk_size / group_size) groups so that
// the bit-packing of one chunk is shared by many threads instead of a single
// one. Must stay small enough to keep good parallelism but large enough that
// most output words a group produces are fully owned (written with a plain
// store) rather than shared at the boundaries (written with an atomicOr).
//
// Empirically the pack kernel is fastest when a full warp covers about one
// Huffman chunk, i.e. group_size ~= huff_block_size / 32: for the default
// huff_block_size = 1024 that measured ~3.6x faster than a fixed 256 on NYX
// 512^3 on Hopper. That "one warp per chunk" reasoning does not transfer
// directly to a 64-wide CDNA wavefront, though: group_size = chunk_size/64
// = 16 measured *worse* on MI300 (~73 GB/s) than the original 32 (~74
// GB/s), and a direct sweep (16/32/64/128/256/512) found the real peak at
// group_size = 128 (~86 GB/s) -- larger, not smaller, than Hopper's tuning.
// Fewer, bigger groups apparently win here because each group boundary
// costs an atomicOr-merged word instead of a plain store (see DeflatePack
// below); halving group_size doubles the boundary count for the same data,
// and on MI300 that atomic-merge overhead outweighs the extra parallelism
// from narrower groups. Revisit with a fresh sweep if huff_block_size
// changes (currently group_size = chunk_size/8, matching the measured
// optimum at the default chunk_size = 1024).
template <typename DeviceType>
MGARDX_CONT SIZE GetDeflateGroupSize(SIZE chunk_size) {
  SIZE group_size;
  if constexpr (std::is_same<DeviceType, HIP>::value) {
    group_size = chunk_size / 8;
  } else {
    SIZE warp_size = DeviceRuntime<DeviceType>::GetWarpSize();
    group_size = chunk_size / warp_size;
  }
  return group_size > 0 ? group_size : 1;
}

// Helper: extract the per-symbol bitwidth, stored in the most-significant byte
// of each fixed-length Huffman codeword (codebook[symbol]).
template <typename H> MGARDX_EXEC uint8_t deflate_bitwidth(H word) {
  return *((uint8_t *)&word + (sizeof(H) - 1));
}

// Phase 1 (sizing): each thread sums the bitwidths of the symbols in one group.
// The fixed-length codeword for symbol s is codebook[s], so we look it up on
// the fly instead of reading a materialized per-symbol array (encode is fused
// here). Groups never cross chunk boundaries, so a later per-chunk reduction
// over the scanned group sums yields per-chunk bit lengths.
template <typename Q, typename H, typename DeviceType>
class DeflateGroupBitsFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DeflateGroupBitsFunctor() {}
  MGARDX_CONT DeflateGroupBitsFunctor(
      SubArray<1, Q, DeviceType> data, SubArray<1, H, DeviceType> codebook,
      SubArray<1, size_t, DeviceType> group_bits, size_t primary_count,
      SIZE chunk_size, SIZE groups_per_chunk, SIZE ngroups,
      SIZE deflate_group_size)
      : data(data), codebook(codebook), group_bits(group_bits),
        primary_count(primary_count), chunk_size(chunk_size),
        groups_per_chunk(groups_per_chunk), ngroups(ngroups),
        deflate_group_size(deflate_group_size) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE gid = FunctorBase<DeviceType>::GetBlockIdX() *
                   FunctorBase<DeviceType>::GetBlockDimX() +
               FunctorBase<DeviceType>::GetThreadIdX();
    if (gid >= ngroups)
      return;
    SIZE chunk_id = gid / groups_per_chunk;
    SIZE local = gid % groups_per_chunk;
    size_t sym_base =
        (size_t)chunk_id * chunk_size + (size_t)local * deflate_group_size;
    size_t chunk_end = (size_t)(chunk_id + 1) * chunk_size;
    if (chunk_end > primary_count)
      chunk_end = primary_count;
    size_t sym_end = sym_base + deflate_group_size;
    if (sym_end > chunk_end)
      sym_end = chunk_end;
    size_t bits = 0;
    for (size_t i = sym_base; i < sym_end; i++) {
      bits += deflate_bitwidth<H>(*codebook(*data(i)));
    }
    *group_bits(gid) = bits;
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, Q, DeviceType> data;
  SubArray<1, H, DeviceType> codebook;
  SubArray<1, size_t, DeviceType> group_bits;
  size_t primary_count;
  SIZE chunk_size;
  SIZE groups_per_chunk;
  SIZE ngroups;
  SIZE deflate_group_size;
};

template <typename Q, typename H, typename DeviceType>
class DeflateGroupBitsKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "deflate_group_bits";
  MGARDX_CONT
  DeflateGroupBitsKernel(SubArray<1, Q, DeviceType> data,
                         SubArray<1, H, DeviceType> codebook,
                         SubArray<1, size_t, DeviceType> group_bits,
                         size_t primary_count, SIZE chunk_size,
                         SIZE groups_per_chunk, SIZE ngroups)
      : data(data), codebook(codebook), group_bits(group_bits),
        primary_count(primary_count), chunk_size(chunk_size),
        groups_per_chunk(groups_per_chunk), ngroups(ngroups) {}

  MGARDX_CONT Task<DeflateGroupBitsFunctor<Q, H, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = DeflateGroupBitsFunctor<Q, H, DeviceType>;
    SIZE deflate_group_size = GetDeflateGroupSize<DeviceType>(chunk_size);
    FunctorType functor(data, codebook, group_bits, primary_count, chunk_size,
                        groups_per_chunk, ngroups, deflate_group_size);
    SIZE tbx;
    if constexpr (std::is_same<DeviceType, HIP>::value) {
      // Empirically tuned on MI300: this kernel has no shared memory or
      // cross-thread communication (each thread independently sums bitwidths
      // for its own group), so block size is a pure occupancy knob. A direct
      // sweep (256/512/768/960) found a non-monotonic curve -- 256 and 512
      // both measure ~113 GB/s (combined with DeflatePack below, via the
      // CLI's "Huffman compress" timer), 768 measures ~120-124 GB/s, and 960
      // drops back to ~111 GB/s. 768 is the peak, likely an occupancy sweet
      // spot for this kernel's register/LDS footprint rather than a value
      // that keeps climbing like OutlierSeparator's. This tuning is
      // HIP/CDNA3-specific and untested on other backends, so it is guarded
      // here rather than applied unconditionally. Revisit with a fresh
      // sweep if this kernel's per-thread work changes.
      tbx = 768;
    } else {
      tbx = 256;
    }
    size_t sm_size = functor.shared_memory_size();
    SIZE gridx = (ngroups - 1) / tbx + 1;
    return Task(functor, 1, 1, gridx, 1, 1, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, Q, DeviceType> data;
  SubArray<1, H, DeviceType> codebook;
  SubArray<1, size_t, DeviceType> group_bits;
  size_t primary_count;
  SIZE chunk_size;
  SIZE groups_per_chunk;
  SIZE ngroups;
};

// Phase 2 (per-chunk meta): from the extended exclusive scan of group bit sums,
// compute each chunk's total bit length and its output length in H-words.
template <typename H, typename DeviceType>
class DeflateChunkMetaFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DeflateChunkMetaFunctor() {}
  MGARDX_CONT
  DeflateChunkMetaFunctor(SubArray<1, size_t, DeviceType> group_offsets,
                          SubArray<1, size_t, DeviceType> chunk_bits,
                          SubArray<1, size_t, DeviceType> chunk_words,
                          SIZE nchunk, SIZE groups_per_chunk)
      : group_offsets(group_offsets), chunk_bits(chunk_bits),
        chunk_words(chunk_words), nchunk(nchunk),
        groups_per_chunk(groups_per_chunk) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE chunk_id = FunctorBase<DeviceType>::GetBlockIdX() *
                        FunctorBase<DeviceType>::GetBlockDimX() +
                    FunctorBase<DeviceType>::GetThreadIdX();
    if (chunk_id >= nchunk)
      return;
    size_t total_bits = *group_offsets((chunk_id + 1) * groups_per_chunk) -
                        *group_offsets(chunk_id * groups_per_chunk);
    *chunk_bits(chunk_id) = total_bits;
    *chunk_words(chunk_id) = (total_bits + sizeof(H) * 8 - 1) / (sizeof(H) * 8);
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, size_t, DeviceType> group_offsets;
  SubArray<1, size_t, DeviceType> chunk_bits;
  SubArray<1, size_t, DeviceType> chunk_words;
  SIZE nchunk;
  SIZE groups_per_chunk;
};

template <typename H, typename DeviceType>
class DeflateChunkMetaKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "deflate_chunk_meta";
  MGARDX_CONT
  DeflateChunkMetaKernel(SubArray<1, size_t, DeviceType> group_offsets,
                         SubArray<1, size_t, DeviceType> chunk_bits,
                         SubArray<1, size_t, DeviceType> chunk_words,
                         SIZE nchunk, SIZE groups_per_chunk)
      : group_offsets(group_offsets), chunk_bits(chunk_bits),
        chunk_words(chunk_words), nchunk(nchunk),
        groups_per_chunk(groups_per_chunk) {}

  MGARDX_CONT Task<DeflateChunkMetaFunctor<H, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = DeflateChunkMetaFunctor<H, DeviceType>;
    FunctorType functor(group_offsets, chunk_bits, chunk_words, nchunk,
                        groups_per_chunk);
    SIZE tbx = 256;
    size_t sm_size = functor.shared_memory_size();
    SIZE gridx = (nchunk - 1) / tbx + 1;
    return Task(functor, 1, 1, gridx, 1, 1, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, size_t, DeviceType> group_offsets;
  SubArray<1, size_t, DeviceType> chunk_bits;
  SubArray<1, size_t, DeviceType> chunk_words;
  SIZE nchunk;
  SIZE groups_per_chunk;
};

// Phase 3 (pack): each thread densely packs one group's symbols, MSB-first,
// directly into the final compressed buffer at the chunk's word offset plus the
// group's intra-chunk bit offset. Output words fully covered by the group are
// written with a plain store; the (at most two) words shared with neighbouring
// groups are merged with atomicOr. The destination region must be zeroed first
// so the atomicOr merges only contribute new bits.
template <typename Q, typename H, typename DeviceType>
class DeflatePackFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DeflatePackFunctor() {}
  MGARDX_CONT
  DeflatePackFunctor(SubArray<1, Q, DeviceType> data,
                     SubArray<1, H, DeviceType> codebook,
                     SubArray<1, size_t, DeviceType> group_offsets,
                     SubArray<1, size_t, DeviceType> chunk_word_offsets,
                     SubArray<1, H, DeviceType> condensed, size_t primary_count,
                     SIZE chunk_size, SIZE groups_per_chunk, SIZE ngroups,
                     SIZE deflate_group_size)
      : data(data), codebook(codebook), group_offsets(group_offsets),
        chunk_word_offsets(chunk_word_offsets), condensed(condensed),
        primary_count(primary_count), chunk_size(chunk_size),
        groups_per_chunk(groups_per_chunk), ngroups(ngroups),
        deflate_group_size(deflate_group_size) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE gid = FunctorBase<DeviceType>::GetBlockIdX() *
                   FunctorBase<DeviceType>::GetBlockDimX() +
               FunctorBase<DeviceType>::GetThreadIdX();
    if (gid >= ngroups)
      return;
    SIZE chunk_id = gid / groups_per_chunk;
    SIZE local = gid % groups_per_chunk;
    size_t sym_base =
        (size_t)chunk_id * chunk_size + (size_t)local * deflate_group_size;
    if (sym_base >= primary_count)
      return;
    size_t chunk_end = (size_t)(chunk_id + 1) * chunk_size;
    if (chunk_end > primary_count)
      chunk_end = primary_count;
    size_t sym_end = sym_base + deflate_group_size;
    if (sym_end > chunk_end)
      sym_end = chunk_end;

    const size_t bpw = sizeof(H) * 8;
    size_t group_start_bit =
        *group_offsets(gid) - *group_offsets(chunk_id * groups_per_chunk);
    size_t chunk_base_word = *chunk_word_offsets(chunk_id);

    size_t lsb_pos = bpw - (group_start_bit % bpw);
    size_t cur = chunk_base_word + group_start_bit / bpw;
    H buffer = 0;
    bool first_word = true;
    bool left_shared = (group_start_bit % bpw) != 0;

    for (size_t i = sym_base; i < sym_end; i++) {
      H word = *codebook(*data(i)); // encode fused in: codeword for symbol
      uint8_t bitwidth = deflate_bitwidth<H>(word);
      *((uint8_t *)&word + (sizeof(H) - 1)) = 0x0; // clear bitwidth byte
      if (lsb_pos == bpw)
        buffer = 0x0; // start of a fresh output word
      if (bitwidth <= lsb_pos) {
        lsb_pos -= bitwidth;
        buffer |= word << lsb_pos;
        if (lsb_pos == 0) {
          // completed an output word
          if (first_word && left_shared) {
            Atomic<H, AtomicGlobalMemory, AtomicDeviceScope, DeviceType>::Or(
                condensed(cur), buffer);
          } else {
            *condensed(cur) = buffer;
          }
          first_word = false;
          cur++;
          lsb_pos = bpw;
          buffer = 0x0;
        }
      } else {
        // code straddles two output words
        H _1 = word >> (bitwidth - lsb_pos);
        H _2 = word << (bpw - (bitwidth - lsb_pos));
        buffer |= _1;
        if (first_word && left_shared) {
          Atomic<H, AtomicGlobalMemory, AtomicDeviceScope, DeviceType>::Or(
              condensed(cur), buffer);
        } else {
          *condensed(cur) = buffer;
        }
        first_word = false;
        cur++;
        buffer = _2;
        lsb_pos = bpw - (bitwidth - lsb_pos);
      }
    }
    // Trailing partial word is shared with the next group (or, for the last
    // group in a chunk, lands in a word owned solely by this chunk). Either way
    // it must be merged, not stored, since neighbours contribute the rest.
    if (lsb_pos != bpw) {
      Atomic<H, AtomicGlobalMemory, AtomicDeviceScope, DeviceType>::Or(
          condensed(cur), buffer);
    }
  }

  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, Q, DeviceType> data;
  SubArray<1, H, DeviceType> codebook;
  SubArray<1, size_t, DeviceType> group_offsets;
  SubArray<1, size_t, DeviceType> chunk_word_offsets;
  SubArray<1, H, DeviceType> condensed;
  size_t primary_count;
  SIZE chunk_size;
  SIZE groups_per_chunk;
  SIZE ngroups;
  SIZE deflate_group_size;
};

template <typename Q, typename H, typename DeviceType>
class DeflatePackKernel : public Kernel {
public:
  constexpr static bool EnableAutoTuning() { return false; }
  constexpr static std::string_view Name = "deflate_pack";
  MGARDX_CONT
  DeflatePackKernel(SubArray<1, Q, DeviceType> data,
                    SubArray<1, H, DeviceType> codebook,
                    SubArray<1, size_t, DeviceType> group_offsets,
                    SubArray<1, size_t, DeviceType> chunk_word_offsets,
                    SubArray<1, H, DeviceType> condensed, size_t primary_count,
                    SIZE chunk_size, SIZE groups_per_chunk, SIZE ngroups)
      : data(data), codebook(codebook), group_offsets(group_offsets),
        chunk_word_offsets(chunk_word_offsets), condensed(condensed),
        primary_count(primary_count), chunk_size(chunk_size),
        groups_per_chunk(groups_per_chunk), ngroups(ngroups) {}

  MGARDX_CONT Task<DeflatePackFunctor<Q, H, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = DeflatePackFunctor<Q, H, DeviceType>;
    SIZE deflate_group_size = GetDeflateGroupSize<DeviceType>(chunk_size);
    FunctorType functor(data, codebook, group_offsets, chunk_word_offsets,
                        condensed, primary_count, chunk_size, groups_per_chunk,
                        ngroups, deflate_group_size);
    // Same empirical sweep, reasoning, and HIP-only guard as
    // DeflateGroupBitsKernel above -- 768 is the measured occupancy sweet
    // spot on MI300 (256/512 ~113 GB/s, 768 ~120-124 GB/s, 960 back down to
    // ~111 GB/s), untested on other backends.
    SIZE tbx;
    if constexpr (std::is_same<DeviceType, HIP>::value) {
      tbx = 768;
    } else {
      tbx = 256;
    }
    size_t sm_size = functor.shared_memory_size();
    SIZE gridx = (ngroups - 1) / tbx + 1;
    return Task(functor, 1, 1, gridx, 1, 1, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, Q, DeviceType> data;
  SubArray<1, H, DeviceType> codebook;
  SubArray<1, size_t, DeviceType> group_offsets;
  SubArray<1, size_t, DeviceType> chunk_word_offsets;
  SubArray<1, H, DeviceType> condensed;
  size_t primary_count;
  SIZE chunk_size;
  SIZE groups_per_chunk;
  SIZE ngroups;
};

} // namespace mgard_x_160

#endif
