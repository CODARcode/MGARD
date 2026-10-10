/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_DECODE_TEMPLATE_HPP
#define MGARD_X_160_DECODE_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"

namespace mgard_x_160 {

template <typename Q, typename H, bool CACHE_SINGLETION, typename DeviceType>
class DecodeFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT DecodeFunctor() {}
  MGARDX_CONT DecodeFunctor(SubArray<1, H, DeviceType> densely,
                            SubArray<1, size_t, DeviceType> dH_meta,
                            SubArray<1, Q, DeviceType> bcode, SIZE len,
                            int chunk_size, int n_chunk,
                            SubArray<1, uint8_t, DeviceType> singleton,
                            size_t singleton_size)
      : densely(densely), dH_meta(dH_meta), bcode(bcode), len(len),
        chunk_size(chunk_size), n_chunk(n_chunk), singleton(singleton),
        singleton_size(singleton_size) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    if (CACHE_SINGLETION) {
      // Opt 3: cooperatively stage only the small, hottest decodebook tables
      // (first[] and entry[], together sizeof(H)*2*word_bw bytes ~ 1KB) into
      // shared memory. These are touched on every decoded bit. The large keys[]
      // table is left in global memory so caching never inflates the per-block
      // shared-memory footprint and hurts occupancy (which is what made caching
      // the whole decodebook a net loss for large dictionaries). All threads
      // participate; the framework inserts a block sync before Operation2.
      _s_singleton = (uint8_t *)FunctorBase<DeviceType>::GetSharedMemory();
      size_t fe_bytes = sizeof(H) * (2 * sizeof(H) * 8);
      uint8_t *src = singleton((IDX)0);
      SIZE tid = FunctorBase<DeviceType>::GetThreadIdX();
      SIZE nthreads = FunctorBase<DeviceType>::GetBlockDimX();
      for (SIZE b = tid; b < (SIZE)fe_bytes; b += nthreads) {
        _s_singleton[b] = src[b];
      }
    } else {
      _s_singleton = singleton((IDX)0);
    }
  }

  MGARDX_EXEC void Operation2() {
    size_t chunk_id = FunctorBase<DeviceType>::GetBlockIdX() *
                          FunctorBase<DeviceType>::GetBlockDimX() +
                      FunctorBase<DeviceType>::GetThreadIdX();
    if (chunk_id >= n_chunk)
      return;

    SIZE densely_offset = *dH_meta(n_chunk + chunk_id);
    SIZE bcode_offset = chunk_size * chunk_id;
    size_t total_bw = *dH_meta(chunk_id);

    const size_t word_bw = sizeof(H) * 8;
    // first[]/entry[] come from _s_singleton (shared when cached, else global);
    // keys[] always stays in global memory.
    auto first = reinterpret_cast<H *>(_s_singleton);
    auto entry = first + sizeof(H) * 8;
    auto keys = reinterpret_cast<Q *>(singleton((IDX)0) +
                                      sizeof(H) * (2 * sizeof(H) * 8));

    // Opt 2: hold the current densely word in a register and refetch from
    // global memory only when the bit cursor crosses into the next word,
    // instead of re-loading the same word once per bit (up to word_bw redundant
    // loads).
    size_t cached_word_idx = 0;
    H cached_word = *densely(densely_offset);

    H v = (cached_word >> (word_bw - 1)) & 0x1; // get the first bit
    size_t l = 1;
    size_t i = 0;
    size_t idx_bcoded = 0;
    while (i < total_bw) {
      while (v < first[l]) { // append next i_cb bit
        ++i;
        size_t idx_word = i / word_bw;
        if (idx_word != cached_word_idx) {
          cached_word = *densely(densely_offset + idx_word);
          cached_word_idx = idx_word;
        }
        H next_bit = (cached_word >> (word_bw - 1 - (i % word_bw))) & 0x1;
        v = (v << 1) | next_bit;
        ++l;
      }
      *bcode(bcode_offset + idx_bcoded) = keys[entry[l] + v - first[l]];
      idx_bcoded++;
      {
        ++i;
        size_t idx_word = i / word_bw;
        if (idx_word != cached_word_idx) {
          cached_word = *densely(densely_offset + idx_word);
          cached_word_idx = idx_word;
        }
        H next_bit = (cached_word >> (word_bw - 1 - (i % word_bw))) & 0x1;
        v = next_bit;
      }
      l = 1;
    }
  }

  MGARDX_CONT size_t shared_memory_size() {
    if (CACHE_SINGLETION) {
      // Only first[]/entry[] are cached (see Operation1), not the full table.
      return sizeof(H) * (2 * sizeof(H) * 8);
    } else {
      return 0;
    }
  }

private:
  SubArray<1, H, DeviceType> densely;
  SubArray<1, size_t, DeviceType> dH_meta;
  SubArray<1, Q, DeviceType> bcode;
  SIZE len;
  int chunk_size;
  int n_chunk;
  SubArray<1, uint8_t, DeviceType> singleton;
  size_t singleton_size;

  uint8_t *_s_singleton;
};

template <typename Q, typename H, bool CACHE_SINGLETION, typename DeviceType>
class DecodeKernel : public Kernel {
public:
  constexpr static DIM NumDim = 1;
  using DataType = H;
  constexpr static std::string_view Name = "decode";
  MGARDX_CONT
  DecodeKernel(SubArray<1, H, DeviceType> densely,
               SubArray<1, size_t, DeviceType> dH_meta,
               SubArray<1, Q, DeviceType> bcode, SIZE len, int chunk_size,
               int n_chunk, SubArray<1, uint8_t, DeviceType> singleton,
               size_t singleton_size)
      : densely(densely), dH_meta(dH_meta), bcode(bcode), len(len),
        chunk_size(chunk_size), n_chunk(n_chunk), singleton(singleton),
        singleton_size(singleton_size) {}

  // NOTE: decode is one independent thread per chunk and is dominated by warp
  // straggler divergence (each thread loops until its own chunk's bits are
  // exhausted, and the warp runs until its slowest chunk finishes). Larger
  // (full-warp) blocks raise occupancy but put 32 chunks under one straggler
  // group instead of 16, which measured ~15% SLOWER on NYX 512^3. The autotuned
  // (half-warp) default is intentionally left in place.
  template <SIZE R, SIZE C, SIZE F>
  MGARDX_CONT Task<DecodeFunctor<Q, H, CACHE_SINGLETION, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = DecodeFunctor<Q, H, CACHE_SINGLETION, DeviceType>;
    FunctorType functor(densely, dH_meta, bcode, len, chunk_size, n_chunk,
                        singleton, singleton_size);

    int nchunk = (len - 1) / chunk_size + 1;
    SIZE tbx, tby, tbz, gridx, gridy, gridz;
    size_t sm_size = functor.shared_memory_size();
    tbz = R;
    tby = C;
    tbx = F;
    gridz = 1;
    gridy = 1;
    gridx = (nchunk - 1) / tbx + 1;
    return Task(functor, gridz, gridy, gridx, tbz, tby, tbx, sm_size, queue_idx,
                std::string(Name));
  }

private:
  SubArray<1, H, DeviceType> densely;
  SubArray<1, size_t, DeviceType> dH_meta;
  SubArray<1, Q, DeviceType> bcode;
  SIZE len;
  int chunk_size;
  int n_chunk;
  SubArray<1, uint8_t, DeviceType> singleton;
  size_t singleton_size;
};

template <typename Q, typename H, typename DeviceType>
void Decode(SubArray<1, H, DeviceType> densely,
            SubArray<1, size_t, DeviceType> dH_meta,
            SubArray<1, Q, DeviceType> bcode, SIZE len, int chunk_size,
            int n_chunk, SubArray<1, uint8_t, DeviceType> singleton,
            size_t singleton_size, int queue_idx) {
  // Opt 3 caches only first[]/entry[] (~1KB) in shared memory, which always
  // fits and never hurts occupancy, so the cached path is used unconditionally.
  if (DeviceRuntime<DeviceType>::PrintKernelConfig) {
    std::cout << log::log_info
              << "Decode: caching first[]/entry[] in shared memory\n";
  }
  DeviceLauncher<DeviceType>::Execute(
      DecodeKernel<Q, H, true, DeviceType>(densely, dH_meta, bcode, len,
                                           chunk_size, n_chunk, singleton,
                                           singleton_size),
      queue_idx);
}

} // namespace mgard_x_160

#endif