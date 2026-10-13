/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_LZ4_KERNELS_HPP
#define MGARD_X_160_LZ4_KERNELS_HPP

#include "../../RuntimeX/RuntimeX.h"

// Portable (multi-kernel) building blocks for a self-contained LZ4 backend --
// a byte-oriented LZ77 dictionary codec, no external nvcomp dependency.
//
// The input byte stream is split into independent chunks of `chunk_size`. Each
// chunk is compressed/decompressed in isolation (its back-references never
// cross a chunk boundary), which makes the whole thing trivially parallel: one
// thread per chunk, no atomics, no cross-chunk state. This mirrors how nvcomp
// batches LZ4 and how BlockDelta gives each data-block a disjoint byte range.
//
// Compression is two phases because a chunk's compressed size is not known
// until it is actually compressed:
//   1) LZ4ChunkCompress : compress chunk c into a worst-case-sized scratch slot
//                         (scratch + c*max_chunk_out) and record comp_bytes[c].
//   2) (host) exclusive scan comp_bytes -> byte_offset (final contiguous
//   layout) 3) LZ4Condense      : copy each chunk's comp_bytes[c] from its
//   scratch slot
//                         to packed[byte_offset[c]] (the gather/compaction
//                         pass).
// Decompression is single-phase: rebuild byte_offset from the stored
// comp_bytes, then one thread per chunk parses tokens and copies.
//
// NOTE (perf): 1 thread per chunk is the portable correctness reference. With
// large chunks it is low-occupancy and keeps a per-thread hash table in global
// scratch. The optimization is one thread *block* per chunk (cooperative
// match-find + a shared-memory hash table), and/or a fused decoupled-look-back
// pass like BlockDeltaFused. Use a small chunk_size (4-8 KB) for this path.

namespace mgard_x_160 {

namespace lz4 {

// ---- format / tuning constants -------------------------------------------
enum : int {
  MIN_MATCH = 4,        // a back-reference must cover >= 4 bytes to pay off
  LAST_LITERALS = 5,    // final 5 bytes of a chunk are always emitted literally
  MFLIMIT = 12,         // last match must start >= 12 bytes before chunk end
  MAX_DISTANCE = 65535, // 16-bit offset -> 64 KB window
  HASH_LOG = 12,        // per-chunk hash table: 1<<HASH_LOG entries
};
static constexpr uint32_t HASH_SIZE = 1u << HASH_LOG;
static constexpr uint32_t HASH_EMPTY = 0xFFFFFFFFu;

// Worst-case compressed size of `s` input bytes (LZ4_compressBound).
MGARDX_CONT_EXEC size_t compress_bound(size_t s) { return s + s / 255 + 16; }

MGARDX_EXEC uint32_t read4(const Byte *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

MGARDX_EXEC uint32_t hash4(uint32_t seq) {
  return (seq * 2654435761u) >> (32 - HASH_LOG);
}

// Write an LZ4 length in the 255-continuation form. Returns the new write pos.
MGARDX_EXEC int write_len(Byte *out, int op, int length) {
  while (length >= 255) {
    out[op++] = (Byte)255;
    length -= 255;
  }
  out[op++] = (Byte)length;
  return op;
}

} // namespace lz4

// ---------------------------------------------------------------------------
// ONE cooperative LZ4 codec, written against the portable SubGroup abstraction
// (mgard_x_160::SubGroup<DeviceType> / SubGroupScalar, defined in the
// DeviceAdapter headers). A "sub-group" is a set of lanes running in lockstep
// that can shfl / ballot / ffs / sync. At size 1 (CPU) every collective is an
// identity and this exact code degenerates into the greedy serial
// encoder/decoder. Warp-size is handled via SG::size()/full_mask()/the mask
// typedef, so it is correct for any width -- the match scan examines size()
// positions per step (a wider sub-group gives slightly different, still-valid
// matches => CR can vary marginally with width; decode output is
// width-independent).
// ---------------------------------------------------------------------------
namespace lz4_coop {

// Cooperative LZ4 encode of one chunk. `ht` is sub-group-shared scratch of
// lz4::HASH_SIZE uint16 slots (0 = empty, else position+1; needs chunk_size
// <= 65535). All lanes of `sg` call this together; returns the byte count.
// At size 1 this is exactly the greedy serial encoder.
template <typename SG>
MGARDX_EXEC int compress_chunk(SG sg, const Byte *in, int len, Byte *out,
                               uint16_t *ht) {
  const int W = SG::size();
  const int lane = sg.lane();
  const int MF = lz4::MFLIMIT, LL = lz4::LAST_LITERALS, MM = lz4::MIN_MATCH;
  const int MAXD = lz4::MAX_DISTANCE;
  const int HS = (int)lz4::HASH_SIZE;

  for (int i = lane; i < HS; i += W)
    ht[i] = 0;
  sg.sync();

  int ip = 0, anchor = 0, op = 0;
  while (ip < len - MF) {
    const int pos = ip + lane;
    // Phase 1: does this lane's position have a valid 4-byte match? (no extend)
    int rref = -1;
    if (pos < len - MF) {
      uint32_t seq = lz4::read4(in + pos);
      uint16_t slot = ht[lz4::hash4(seq)];
      if (slot != 0) {
        int r = (int)slot - 1;
        if (pos - r <= MAXD && lz4::read4(in + r) == seq)
          rref = r;
      }
    }
    typename SG::mask_t ballot = sg.ballot(rref >= 0);

    if (ballot == 0) {
      if (pos < len - 3)
        ht[lz4::hash4(lz4::read4(in + pos))] = (uint16_t)(pos + 1);
      sg.sync();
      ip += W;
      continue;
    }

    const int j = sg.ffs(ballot) - 1; // first (left-most) matching lane
    const int mpos = ip + j;
    const int r_j = sg.shfl(rref, j);
    const int off_j = mpos - r_j;

    // Phase 2: cooperative match-length extension (W bytes per step).
    int ml = MM;
    {
      int base = MM;
      while (true) {
        int q = base + lane, a = mpos + q, b = r_j + q;
        int eq = (a < len - LL && in[b] == in[a]) ? 1 : 0;
        typename SG::mask_t good = sg.ballot(eq);
        if (good == sg.full_mask())
          base += W;
        else {
          ml = base + (sg.ffs(~good & sg.full_mask()) - 1);
          break;
        }
      }
    }

    // Record literal positions we passed [ip, mpos] for later windows.
    if (lane <= j && pos < len - 3)
      ht[lz4::hash4(lz4::read4(in + pos))] = (uint16_t)(pos + 1);
    sg.sync();

    // Emit: token | litlen-ext | literals | offset | matchlen-ext.
    const int litlen = mpos - anchor;
    const int numLitExt = litlen < 15 ? 0 : ((litlen - 15) / 255 + 1);
    const int litStart = op + 1 + numLitExt;
    const int mlf = ml - MM;
    const int numMlExt = mlf < 15 ? 0 : ((mlf - 15) / 255 + 1);
    const int after = litStart + litlen + 2 + numMlExt;
    if (lane == 0) {
      int tHi = litlen < 15 ? litlen : 15, tLo = mlf < 15 ? mlf : 15;
      out[op] = (Byte)((tHi << 4) | tLo);
      if (litlen >= 15)
        lz4::write_len(out, op + 1, litlen - 15);
      int offPos = litStart + litlen;
      out[offPos] = (Byte)(off_j & 0xff);
      out[offPos + 1] = (Byte)((off_j >> 8) & 0xff);
      if (mlf >= 15)
        lz4::write_len(out, offPos + 2, mlf - 15);
    }
    for (int k = lane; k < litlen; k += W) // cooperative literal copy
      out[litStart + k] = in[anchor + k];
    sg.sync();

    op = after;
    ip = mpos + ml;
    anchor = ip;
  }

  // Terminal literal-only sequence [anchor, len).
  const int litlen = len - anchor;
  const int numLitExt = litlen < 15 ? 0 : ((litlen - 15) / 255 + 1);
  const int litStart = op + 1 + numLitExt;
  if (lane == 0) {
    int tHi = litlen < 15 ? litlen : 15;
    out[op] = (Byte)(tHi << 4);
    if (litlen >= 15)
      lz4::write_len(out, op + 1, litlen - 15);
  }
  for (int k = lane; k < litlen; k += W)
    out[litStart + k] = in[anchor + k];
  sg.sync();
  return litStart + litlen;
}

// Cooperative decode of one chunk. Lane 0 parses the serial token stream; the
// sub-group cooperatively copies literals and the match. Overlap-safe via the
// periodic seed out[op+k] = out[op-offset + (k mod offset)] (reads only the
// already-decoded seed). Driven by output length. Size 1 = serial decoder.
template <typename SG>
MGARDX_EXEC void decompress_chunk(SG sg, const Byte *in, Byte *out,
                                  int outLen) {
  const int W = SG::size();
  const int lane = sg.lane();
  const int MM = lz4::MIN_MATCH;
  int ip = 0, op = 0;
  while (op < outLen) {
    int litlen = 0, lit_in = 0, off = 0, mlen = 0, is_term = 0;
    if (lane == 0) {
      Byte token = in[ip++];
      litlen = token >> 4;
      if (litlen == 15) {
        Byte b;
        do {
          b = in[ip++];
          litlen += b;
        } while (b == 255);
      }
      lit_in = ip;
      ip += litlen;
      if (op + litlen >= outLen) {
        is_term = 1;
      } else {
        off = (int)in[ip] | ((int)in[ip + 1] << 8);
        ip += 2;
        mlen = (token & 0xf) + MM;
        if ((token & 0xf) == 15) {
          Byte b;
          do {
            b = in[ip++];
            mlen += b;
          } while (b == 255);
        }
      }
    }
    litlen = sg.shfl(litlen, 0);
    lit_in = sg.shfl(lit_in, 0);
    is_term = sg.shfl(is_term, 0);
    off = sg.shfl(off, 0);
    mlen = sg.shfl(mlen, 0);

    for (int k = lane; k < litlen; k += W)
      out[op + k] = in[lit_in + k];
    op += litlen;
    sg.sync();
    if (is_term)
      break;

    const int seed = op - off;
    if (off >= mlen) {
      for (int k = lane; k < mlen; k += W)
        out[op + k] = out[seed + k];
    } else {
      for (int k = lane; k < mlen; k += W)
        out[op + k] = out[seed + (k % off)];
    }
    op += mlen;
    sg.sync();
  }
}

} // namespace lz4_coop

// ---------------------------------------------------------------------------
// Kernel 1: compress each chunk into its scratch slot; record comp_bytes[c].
// ---------------------------------------------------------------------------
template <typename DeviceType>
class LZ4ChunkCompressFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT LZ4ChunkCompressFunctor() {}
  MGARDX_CONT
  LZ4ChunkCompressFunctor(SubArray<1, Byte, DeviceType> input, SIZE n,
                          SIZE chunk_size, SIZE nchunks, SIZE max_chunk_out,
                          SubArray<1, uint16_t, DeviceType> htable,
                          SubArray<1, Byte, DeviceType> scratch,
                          SubArray<1, size_t, DeviceType> comp_bytes)
      : input(input), n(n), chunk_size(chunk_size), nchunks(nchunks),
        max_chunk_out(max_chunk_out), htable(htable), scratch(scratch),
        comp_bytes(comp_bytes) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE c = (FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX()) +
             FunctorBase<DeviceType>::GetThreadIdX();
    if (c >= nchunks)
      return;
    SIZE start = c * chunk_size;
    int len = (int)(chunk_size < (n - start) ? chunk_size : (n - start));
    Byte *out = scratch((size_t)c * max_chunk_out);
    uint16_t *ht = htable((size_t)c * lz4::HASH_SIZE);
    int outlen =
        lz4_coop::compress_chunk(SubGroupScalar{}, input(start), len, out, ht);
    *comp_bytes(c) = (size_t)outlen;
  }

  MGARDX_EXEC void Operation2() {}
  MGARDX_EXEC void Operation3() {}
  MGARDX_EXEC void Operation4() {}
  MGARDX_EXEC void Operation5() {}
  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, Byte, DeviceType> input;
  SIZE n, chunk_size, nchunks, max_chunk_out;
  SubArray<1, uint16_t, DeviceType> htable;
  SubArray<1, Byte, DeviceType> scratch;
  SubArray<1, size_t, DeviceType> comp_bytes;
};

template <typename DeviceType> class LZ4ChunkCompressKernel : public Kernel {
public:
  constexpr static DIM NumDim = 1;
  constexpr static bool EnableAutoTuning() { return false; }
  using DataType = Byte;
  constexpr static std::string_view Name = "lz4_chunk_compress";
  MGARDX_CONT LZ4ChunkCompressKernel(SubArray<1, Byte, DeviceType> input,
                                     SIZE n, SIZE chunk_size, SIZE nchunks,
                                     SIZE max_chunk_out,
                                     SubArray<1, uint16_t, DeviceType> htable,
                                     SubArray<1, Byte, DeviceType> scratch,
                                     SubArray<1, size_t, DeviceType> comp_bytes)
      : input(input), n(n), chunk_size(chunk_size), nchunks(nchunks),
        max_chunk_out(max_chunk_out), htable(htable), scratch(scratch),
        comp_bytes(comp_bytes) {}

  MGARDX_CONT Task<LZ4ChunkCompressFunctor<DeviceType>> GenTask(int queue_idx) {
    using FunctorType = LZ4ChunkCompressFunctor<DeviceType>;
    FunctorType functor(input, n, chunk_size, nchunks, max_chunk_out, htable,
                        scratch, comp_bytes);
    SIZE tbx = 256, tby = 1, tbz = 1;
    SIZE gridx = (nchunks - 1) / tbx + 1;
    return Task(functor, 1, 1, gridx, tbz, tby, tbx,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<1, Byte, DeviceType> input;
  SIZE n, chunk_size, nchunks, max_chunk_out;
  SubArray<1, uint16_t, DeviceType> htable;
  SubArray<1, Byte, DeviceType> scratch;
  SubArray<1, size_t, DeviceType> comp_bytes;
};

// ---------------------------------------------------------------------------
// Kernel 2: gather each chunk's compressed bytes into the contiguous output.
// ---------------------------------------------------------------------------
template <typename DeviceType>
class LZ4CondenseFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT LZ4CondenseFunctor() {}
  MGARDX_CONT LZ4CondenseFunctor(SubArray<1, Byte, DeviceType> scratch,
                                 SIZE nchunks, SIZE max_chunk_out,
                                 SubArray<1, size_t, DeviceType> comp_bytes,
                                 SubArray<1, size_t, DeviceType> byte_offset,
                                 SubArray<1, Byte, DeviceType> packed)
      : scratch(scratch), nchunks(nchunks), max_chunk_out(max_chunk_out),
        comp_bytes(comp_bytes), byte_offset(byte_offset), packed(packed) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE c = (FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX()) +
             FunctorBase<DeviceType>::GetThreadIdX();
    if (c >= nchunks)
      return;
    Byte *src = scratch((size_t)c * max_chunk_out);
    Byte *dst = packed(*byte_offset(c));
    size_t cb = *comp_bytes(c);
    // TUNING: serial per-thread copy. The cooperative (block-per-chunk) version
    // copies this with the whole block striding over cb.
    for (size_t k = 0; k < cb; k++)
      dst[k] = src[k];
  }

  MGARDX_EXEC void Operation2() {}
  MGARDX_EXEC void Operation3() {}
  MGARDX_EXEC void Operation4() {}
  MGARDX_EXEC void Operation5() {}
  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, Byte, DeviceType> scratch;
  SIZE nchunks, max_chunk_out;
  SubArray<1, size_t, DeviceType> comp_bytes;
  SubArray<1, size_t, DeviceType> byte_offset;
  SubArray<1, Byte, DeviceType> packed;
};

template <typename DeviceType> class LZ4CondenseKernel : public Kernel {
public:
  constexpr static DIM NumDim = 1;
  constexpr static bool EnableAutoTuning() { return false; }
  using DataType = Byte;
  constexpr static std::string_view Name = "lz4_condense";
  MGARDX_CONT LZ4CondenseKernel(SubArray<1, Byte, DeviceType> scratch,
                                SIZE nchunks, SIZE max_chunk_out,
                                SubArray<1, size_t, DeviceType> comp_bytes,
                                SubArray<1, size_t, DeviceType> byte_offset,
                                SubArray<1, Byte, DeviceType> packed)
      : scratch(scratch), nchunks(nchunks), max_chunk_out(max_chunk_out),
        comp_bytes(comp_bytes), byte_offset(byte_offset), packed(packed) {}

  MGARDX_CONT Task<LZ4CondenseFunctor<DeviceType>> GenTask(int queue_idx) {
    using FunctorType = LZ4CondenseFunctor<DeviceType>;
    FunctorType functor(scratch, nchunks, max_chunk_out, comp_bytes,
                        byte_offset, packed);
    SIZE tbx = 256, tby = 1, tbz = 1;
    SIZE gridx = (nchunks - 1) / tbx + 1;
    return Task(functor, 1, 1, gridx, tbz, tby, tbx,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<1, Byte, DeviceType> scratch;
  SIZE nchunks, max_chunk_out;
  SubArray<1, size_t, DeviceType> comp_bytes;
  SubArray<1, size_t, DeviceType> byte_offset;
  SubArray<1, Byte, DeviceType> packed;
};

// ---------------------------------------------------------------------------
// Kernel 3: decode each chunk back into the contiguous output stream.
// ---------------------------------------------------------------------------
template <typename DeviceType>
class LZ4ChunkDecompressFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT LZ4ChunkDecompressFunctor() {}
  MGARDX_CONT
  LZ4ChunkDecompressFunctor(SubArray<1, Byte, DeviceType> packed,
                            SubArray<1, size_t, DeviceType> byte_offset, SIZE n,
                            SIZE chunk_size, SIZE nchunks,
                            SubArray<1, Byte, DeviceType> output)
      : packed(packed), byte_offset(byte_offset), n(n), chunk_size(chunk_size),
        nchunks(nchunks), output(output) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE c = (FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX()) +
             FunctorBase<DeviceType>::GetThreadIdX();
    if (c >= nchunks)
      return;
    SIZE start = c * chunk_size;
    int len = (int)(chunk_size < (n - start) ? chunk_size : (n - start));
    lz4_coop::decompress_chunk(SubGroupScalar{}, packed(*byte_offset(c)),
                               output(start), len);
  }

  MGARDX_EXEC void Operation2() {}
  MGARDX_EXEC void Operation3() {}
  MGARDX_EXEC void Operation4() {}
  MGARDX_EXEC void Operation5() {}
  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, Byte, DeviceType> packed;
  SubArray<1, size_t, DeviceType> byte_offset;
  SIZE n, chunk_size, nchunks;
  SubArray<1, Byte, DeviceType> output;
};

template <typename DeviceType> class LZ4ChunkDecompressKernel : public Kernel {
public:
  constexpr static DIM NumDim = 1;
  constexpr static bool EnableAutoTuning() { return false; }
  using DataType = Byte;
  constexpr static std::string_view Name = "lz4_chunk_decompress";
  MGARDX_CONT
  LZ4ChunkDecompressKernel(SubArray<1, Byte, DeviceType> packed,
                           SubArray<1, size_t, DeviceType> byte_offset, SIZE n,
                           SIZE chunk_size, SIZE nchunks,
                           SubArray<1, Byte, DeviceType> output)
      : packed(packed), byte_offset(byte_offset), n(n), chunk_size(chunk_size),
        nchunks(nchunks), output(output) {}

  MGARDX_CONT Task<LZ4ChunkDecompressFunctor<DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = LZ4ChunkDecompressFunctor<DeviceType>;
    FunctorType functor(packed, byte_offset, n, chunk_size, nchunks, output);
    SIZE tbx = 256, tby = 1, tbz = 1;
    SIZE gridx = (nchunks - 1) / tbx + 1;
    return Task(functor, 1, 1, gridx, tbz, tby, tbx,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<1, Byte, DeviceType> packed;
  SubArray<1, size_t, DeviceType> byte_offset;
  SIZE n, chunk_size, nchunks;
  SubArray<1, Byte, DeviceType> output;
};

} // namespace mgard_x_160

#endif
