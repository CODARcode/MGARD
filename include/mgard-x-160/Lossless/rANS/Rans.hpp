/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_RANS_TEMPLATE_HPP
#define MGARD_X_160_RANS_TEMPLATE_HPP

#include <vector>

#include "../../RuntimeX/RuntimeX.h"
#include "../ParallelHuffman/Histogram.hpp"
#include "RansCommon.hpp"
#include "RansDecode.hpp"
#include "RansEncode.hpp"
#include "RansInterleaved.hpp"

namespace mgard_x_160 {
namespace rans {

// Static rANS entropy coder over a symbol alphabet of arbitrary size (set at
// Resize): 256 for a byte stream, or dict_size for the Huffman-style primary
// quantized-symbol stream. Unlike Huffman it codes fractional bits, so the
// compressed size is the order-0 entropy of the symbol stream (no 1-bit/symbol
// floor and no integer-code-length rounding loss). Parallelism comes from
// splitting the input into independent segments, each its own rANS stream.
//
// Q is the symbol storage type (Byte, or the unsigned quantized type). Symbols
// must lie in [0, alphabet) and alphabet must be <= 65536 (slot table is
// uint16). The compressed stream is always bytes.
template <typename Q, typename DeviceType> class Rans {
public:
  Rans() : initialized(false) {}

  // Choose M = 2^scale_bits with generous headroom over the alphabet so the
  // normalized frequencies (each present symbol >= 1) keep enough precision:
  // too small an M rounds the per-symbol probabilities coarsely and erases
  // rANS's fractional-bit advantage over Huffman. Aim for ~16x the alphabet,
  // capped at 2^16 (the max that keeps the 32-bit state valid with byte
  // renormalization).
  static uint32_t ScaleBitsFor(int alphabet) {
    uint32_t sb = 12;
    while (((uint32_t)1 << sb) < (uint32_t)alphabet * 16) {
      sb++;
    }
    if (sb > 16) {
      sb = 16;
    }
    return sb;
  }

  void Resize(SIZE max_size, int alphabet, int queue_idx,
              SIZE segment_size_override = 0, bool interleaved_mode = false) {
    this->initialized = true;
    this->max_size = max_size;
    this->alphabet = alphabet;
    this->scale_bits = ScaleBitsFor(alphabet);
    this->interleaved = interleaved_mode;
    this->segment_size = segment_size_override > 0 ? segment_size_override
                                                   : DEFAULT_SEGMENT_SIZE;

    // Two layouts share the same scratch/compact/offset machinery:
    //  - non-interleaved: one stream per lane (num_segments = blocks*NLANES),
    //    each stream's scratch holds up to segment_size symbols.
    //  - interleaved (warp-coalesced): one shared stream per block of NLANES
    //    lanes (num_segments = blocks), each block's scratch holds NLANES
    //    lanes' worth of bytes plus NLANES state flushes.
    SIZE block_symbols = (SIZE)RANS_NLANES * segment_size;
    SIZE max_blocks = (max_size + block_symbols - 1) / block_symbols;
    SIZE max_segments;
    if (interleaved) {
      this->seg_capacity =
          (IDX)RANS_NLANES * segment_size * 2 + (IDX)RANS_NLANES * 8 + 16;
      max_segments = max_blocks;
    } else {
      this->seg_capacity = (IDX)segment_size * 2 + 16;
      max_segments = max_blocks * (SIZE)RANS_NLANES;
    }
    if (max_segments == 0) {
      max_segments = 1;
    }
    SIZE table_size = (SIZE)1 << scale_bits;

    freq32.resize({(SIZE)alphabet}, queue_idx);
    freq_d.resize({(SIZE)alphabet}, queue_idx);
    cum_d.resize({(SIZE)alphabet}, queue_idx);
    esym_d.resize({(SIZE)alphabet}, queue_idx);
    slot2sym_d.resize({table_size}, queue_idx);
    seg_len_d.resize({max_segments}, queue_idx);
    scratch.resize({(SIZE)(max_segments * seg_capacity)}, queue_idx);

    hcounts.resize(alphabet);
    hfreq.resize(alphabet);
    hcum.resize(alphabet);
    hnorm.resize(alphabet);
    hesym.resize(alphabet);
    hslot.resize(table_size);
    hseg_len.resize(max_segments);
    hseg_off.resize(max_segments);

    MemoryManager<DeviceType>::MallocHost(signature_verify, 7 * sizeof(char),
                                          queue_idx);
  }

  // Normalize the raw symbol counts so they sum to exactly M = 2^scale_bits,
  // with every present symbol getting frequency >= 1, then build the cumulative
  // table, the slot->symbol lookup, and the reciprocal-multiply encode tables.
  void BuildTables(SIZE n, int queue_idx) {
    uint32_t M = 1u << scale_bits;

    MemoryManager<DeviceType>::Copy1D(hcounts.data(), freq32.data(), alphabet,
                                      queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    uint32_t sum = 0;
    int argmax = -1;
    uint32_t max_freq = 0;
    for (int s = 0; s < alphabet; s++) {
      uint32_t f = 0;
      if (hcounts[s] > 0) {
        double scaled = (double)hcounts[s] * (double)M / (double)n;
        f = (uint32_t)(scaled + 0.5);
        if (f == 0) {
          f = 1;
        }
      }
      hnorm[s] = f;
      sum += f;
      if (f > max_freq) {
        max_freq = f;
        argmax = s;
      }
    }

    // Reconcile the rounded sum to exactly M by nudging the largest bins (never
    // dropping a present symbol below 1).
    while (sum > M) {
      int best = -1;
      uint32_t best_f = 1;
      for (int s = 0; s < alphabet; s++) {
        if (hnorm[s] > best_f) {
          best_f = hnorm[s];
          best = s;
        }
      }
      if (best < 0) {
        break;
      }
      hnorm[best]--;
      sum--;
    }
    while (sum < M) {
      hnorm[argmax]++;
      sum++;
    }

    uint32_t c = 0;
    for (int s = 0; s < alphabet; s++) {
      hfreq[s] = hnorm[s];
      hcum[s] = c;
      for (uint32_t k = 0; k < hnorm[s]; k++) {
        hslot[c + k] = (uint16_t)s;
      }
      RansEncPacked &ep = hesym[s];
      if (hnorm[s] > 0) {
        RansEncSymbol es;
        RansEncSymbolInit(es, c, hnorm[s], scale_bits);
        ep.x_max = es.x_max;
        ep.rcp_freq = es.rcp_freq;
        ep.bias = es.bias;
        ep.cmpl_freq = (uint16_t)es.cmpl_freq;
        ep.rcp_shift = (uint16_t)es.rcp_shift;
      } else {
        ep.x_max = 0;
        ep.rcp_freq = 0;
        ep.bias = 0;
        ep.cmpl_freq = 0;
        ep.rcp_shift = 0;
      }
      c += hnorm[s];
    }

    freq_d.load(hfreq.data(), 0, queue_idx);
    cum_d.load(hcum.data(), 0, queue_idx);
    slot2sym_d.load(hslot.data(), 0, queue_idx);
    esym_d.load(hesym.data(), 0, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  }

  bool Compress(Array<1, Q, DeviceType> &input_data,
                Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE n = input_data.shape(0);
    SIZE block_symbols = (SIZE)RANS_NLANES * segment_size;
    SIZE num_blocks = (n + block_symbols - 1) / block_symbols;
    // The compaction unit ("segment") is one shared stream per block when
    // interleaved, else one stream per lane.
    SIZE num_segments =
        interleaved ? num_blocks : num_blocks * (SIZE)RANS_NLANES;
    if (n == 0) {
      num_segments = 0;
    }

    if (n > 0) {
      MemoryManager<DeviceType>::Memset1D(freq32.data(), alphabet, 0,
                                          queue_idx);
      Histogram<Q, uint32_t, DeviceType>(SubArray(input_data), SubArray(freq32),
                                         n, alphabet, queue_idx);
      BuildTables(n, queue_idx);
    }

    if (num_segments > 0) {
      seg_len_d.resize({num_segments}, queue_idx);
      if (interleaved) {
        if constexpr ((int)SubGroup<DeviceType>::size() % (int)RANS_NLANES ==
                      0) {
          // Warp-cooperative coalesced encode for subgroups that are a multiple
          // of RANS_NLANES: a 32-lane CUDA warp handles one logical block, a
          // 64-lane CDNA wavefront handles two. The bitstream is identical
          // (always 32 logical lanes), so it stays cross-backend compatible.
          DeviceLauncher<DeviceType>::Execute(
              InterleavedEncodeWarpKernel<Q, DeviceType>(
                  SubArray(input_data), n, segment_size, num_blocks,
                  SubArray(esym_d), seg_capacity, SubArray(scratch),
                  SubArray(seg_len_d)),
              queue_idx);
        } else {
          // Portable sequential reference (size-1 subgroup backends).
          DeviceLauncher<DeviceType>::Execute(
              InterleavedEncodeKernel<Q, DeviceType>(
                  SubArray(input_data), n, segment_size, num_blocks,
                  SubArray(esym_d), seg_capacity, SubArray(scratch),
                  SubArray(seg_len_d)),
              queue_idx);
        }
      } else {
        DeviceLauncher<DeviceType>::Execute(
            EncodeKernel<Q, DeviceType>(SubArray(input_data), n, segment_size,
                                        num_segments, SubArray(esym_d),
                                        seg_capacity, SubArray(scratch),
                                        SubArray(seg_len_d)),
            queue_idx);
      }

      MemoryManager<DeviceType>::Copy1D(hseg_len.data(), seg_len_d.data(),
                                        num_segments, queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    }

    uint64_t stream_bytes = 0;
    for (SIZE p = 0; p < num_segments; p++) {
      hseg_off[p] = (uint32_t)stream_bytes;
      stream_bytes += hseg_len[p];
    }

    SIZE byte_offset = 0;
    advance_with_align<Byte>(byte_offset, 7);                // signature
    advance_with_align<SIZE>(byte_offset, 1);                // scale_bits
    advance_with_align<SIZE>(byte_offset, 1);                // alphabet
    advance_with_align<SIZE>(byte_offset, 1);                // original_length
    advance_with_align<SIZE>(byte_offset, 1);                // segment_size
    advance_with_align<SIZE>(byte_offset, 1);                // interleaved
    advance_with_align<SIZE>(byte_offset, 1);                // num_segments
    advance_with_align<SIZE>(byte_offset, 1);                // stream_bytes
    advance_with_align<uint16_t>(byte_offset, alphabet);     // norm freq
    advance_with_align<uint32_t>(byte_offset, num_segments); // seg offsets
    advance_with_align<Byte>(byte_offset, (SIZE)stream_bytes); // stream

    compressed_data.resize({byte_offset}, queue_idx);
    SubArray<1, Byte, DeviceType> compressed_subarray(compressed_data);

    SIZE scale_bits_s = scale_bits, alphabet_s = alphabet,
         original_length_s = n, segment_size_s = segment_size,
         num_segments_s = num_segments, stream_bytes_s = stream_bytes,
         interleaved_s = interleaved ? 1 : 0;

    byte_offset = 0;
    SerializeArray<Byte>(compressed_subarray, signature, 7, byte_offset,
                         queue_idx);
    SerializeArray<SIZE>(compressed_subarray, &scale_bits_s, 1, byte_offset,
                         queue_idx);
    SerializeArray<SIZE>(compressed_subarray, &alphabet_s, 1, byte_offset,
                         queue_idx);
    SerializeArray<SIZE>(compressed_subarray, &original_length_s, 1,
                         byte_offset, queue_idx);
    SerializeArray<SIZE>(compressed_subarray, &segment_size_s, 1, byte_offset,
                         queue_idx);
    SerializeArray<SIZE>(compressed_subarray, &interleaved_s, 1, byte_offset,
                         queue_idx);
    SerializeArray<SIZE>(compressed_subarray, &num_segments_s, 1, byte_offset,
                         queue_idx);
    SerializeArray<SIZE>(compressed_subarray, &stream_bytes_s, 1, byte_offset,
                         queue_idx);
    SerializeArray<uint16_t>(compressed_subarray, hnorm.data(), alphabet,
                             byte_offset, queue_idx);

    align_byte_offset<uint32_t>(byte_offset);
    SubArray<1, uint32_t, DeviceType> seg_offset_sub(
        {num_segments}, (uint32_t *)(compressed_data.data() + byte_offset));
    SerializeArray<uint32_t>(compressed_subarray, hseg_off.data(), num_segments,
                             byte_offset, queue_idx);

    align_byte_offset<Byte>(byte_offset);
    SubArray<1, Byte, DeviceType> stream_sub(
        {(SIZE)stream_bytes}, (Byte *)(compressed_data.data() + byte_offset));

    if (num_segments > 0) {
      DeviceLauncher<DeviceType>::Execute(
          CompactKernel<DeviceType>(SubArray(scratch), SubArray(seg_len_d),
                                    seg_offset_sub, num_segments, seg_capacity,
                                    stream_sub),
          queue_idx);
    }
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      log::info(
          "rANS compression ratio: " + std::to_string(n * sizeof(Q)) + "/" +
          std::to_string(compressed_data.shape(0)) + " (" +
          std::to_string((double)n * sizeof(Q) / compressed_data.shape(0)) +
          ")");
      timer.end();
      timer.print("rANS compress", n * sizeof(Q));
      timer.clear();
    }
    return true;
  }

  void Serialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {}

  bool Verify(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    SubArray compressed_subarray(compressed_data);
    SIZE byte_offset = 0;
    DeserializeArray<Byte>(compressed_subarray, signature_verify, 7,
                           byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    for (int i = 0; i < 7; i++) {
      if (signature[i] != signature_verify[i]) {
        return false;
      }
    }
    return true;
  }

  void Deserialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    if (!Verify(compressed_data, queue_idx)) {
      throw std::runtime_error("rANS signature mismatch.");
    }
    SubArray<1, Byte, DeviceType> compressed_subarray(compressed_data);
    Byte *signature_ptr = nullptr;
    SIZE scale_bits_s, alphabet_s, original_length_s, segment_size_s,
        num_segments_s, stream_bytes_s, interleaved_s;
    SIZE *p_scale = &scale_bits_s, *p_alpha = &alphabet_s,
         *p_len = &original_length_s, *p_seg = &segment_size_s,
         *p_num = &num_segments_s, *p_stream = &stream_bytes_s,
         *p_il = &interleaved_s;

    SIZE byte_offset = 0;
    DeserializeArray<Byte>(compressed_subarray, signature_ptr, 7, byte_offset,
                           true, queue_idx);
    DeserializeArray<SIZE>(compressed_subarray, p_scale, 1, byte_offset, false,
                           queue_idx);
    DeserializeArray<SIZE>(compressed_subarray, p_alpha, 1, byte_offset, false,
                           queue_idx);
    DeserializeArray<SIZE>(compressed_subarray, p_len, 1, byte_offset, false,
                           queue_idx);
    DeserializeArray<SIZE>(compressed_subarray, p_seg, 1, byte_offset, false,
                           queue_idx);
    DeserializeArray<SIZE>(compressed_subarray, p_il, 1, byte_offset, false,
                           queue_idx);
    DeserializeArray<SIZE>(compressed_subarray, p_num, 1, byte_offset, false,
                           queue_idx);
    DeserializeArray<SIZE>(compressed_subarray, p_stream, 1, byte_offset, false,
                           queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    scale_bits = scale_bits_s;
    alphabet = alphabet_s;
    original_length = original_length_s;
    segment_size = segment_size_s;
    num_segments = num_segments_s;
    stream_bytes = stream_bytes_s;
    interleaved = interleaved_s != 0;

    if ((int)hnorm.size() < alphabet) {
      hnorm.resize(alphabet);
      hfreq.resize(alphabet);
      hcum.resize(alphabet);
      hslot.resize((size_t)1 << scale_bits);
    }
    uint16_t *hnorm_ptr = hnorm.data();
    DeserializeArray<uint16_t>(compressed_subarray, hnorm_ptr, alphabet,
                               byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    seg_offset_ptr = nullptr;
    DeserializeArray<uint32_t>(compressed_subarray, seg_offset_ptr,
                               num_segments, byte_offset, true, queue_idx);
    stream_ptr = nullptr;
    DeserializeArray<Byte>(compressed_subarray, stream_ptr, stream_bytes,
                           byte_offset, true, queue_idx);
  }

  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  Array<1, Q, DeviceType> &decompressed_data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    decompressed_data.resize({(SIZE)original_length}, queue_idx);

    if (num_segments > 0) {
      uint32_t c = 0;
      for (int s = 0; s < alphabet; s++) {
        hfreq[s] = hnorm[s];
        hcum[s] = c;
        for (uint32_t k = 0; k < hnorm[s]; k++) {
          hslot[c + k] = (uint16_t)s;
        }
        c += hnorm[s];
      }
      if ((int)freq_d.shape(0) < alphabet) {
        freq_d.resize({(SIZE)alphabet}, queue_idx);
        cum_d.resize({(SIZE)alphabet}, queue_idx);
        slot2sym_d.resize({(SIZE)1 << scale_bits}, queue_idx);
      }
      freq_d.load(hfreq.data(), 0, queue_idx);
      cum_d.load(hcum.data(), 0, queue_idx);
      slot2sym_d.load(hslot.data(), 0, queue_idx);

      SubArray<1, uint32_t, DeviceType> seg_offset_sub({(SIZE)num_segments},
                                                       seg_offset_ptr);
      SubArray<1, Byte, DeviceType> stream_sub({(SIZE)stream_bytes},
                                               stream_ptr);

      if (interleaved) {
        // num_segments == num_blocks here.
        DeviceLauncher<DeviceType>::Execute(
            InterleavedDecodeKernel<Q, DeviceType>(
                stream_sub, seg_offset_sub, original_length, segment_size,
                num_segments, SubArray(freq_d), SubArray(cum_d),
                SubArray(slot2sym_d), scale_bits, SubArray(decompressed_data)),
            queue_idx);
      } else {
        DeviceLauncher<DeviceType>::Execute(
            DecodeKernel<Q, DeviceType>(
                stream_sub, seg_offset_sub, original_length, segment_size,
                num_segments, SubArray(freq_d), SubArray(cum_d),
                SubArray(slot2sym_d), scale_bits, SubArray(decompressed_data)),
            queue_idx);
      }
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("rANS decompress", original_length * sizeof(Q));
      timer.clear();
    }
  }

  static constexpr SIZE DEFAULT_SEGMENT_SIZE = 2048;

  bool initialized;
  bool interleaved = false;
  SIZE max_size = 0;
  int alphabet = 256;
  uint32_t scale_bits = 12;
  SIZE segment_size = DEFAULT_SEGMENT_SIZE;
  IDX seg_capacity = 0;
  SIZE original_length = 0;
  SIZE num_segments = 0;
  SIZE stream_bytes = 0;

  uint32_t *seg_offset_ptr = nullptr;
  Byte *stream_ptr = nullptr;
  Byte signature[7] = {'M', 'G', 'X', 'R', 'A', 'N', 'S'};
  Byte *signature_verify;

  Array<1, uint32_t, DeviceType> freq32;
  Array<1, uint32_t, DeviceType> freq_d;
  Array<1, uint32_t, DeviceType> cum_d;
  Array<1, RansEncPacked, DeviceType> esym_d;
  Array<1, uint16_t, DeviceType> slot2sym_d;
  Array<1, uint32_t, DeviceType> seg_len_d;
  Array<1, Byte, DeviceType> scratch;

  std::vector<uint32_t> hcounts;
  std::vector<uint32_t> hfreq;
  std::vector<uint32_t> hcum;
  std::vector<uint16_t> hnorm;
  std::vector<RansEncPacked> hesym;
  std::vector<uint16_t> hslot;
  std::vector<uint32_t> hseg_len;
  std::vector<uint32_t> hseg_off;
};

} // namespace rans
} // namespace mgard_x_160
#endif
