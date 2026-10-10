/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_BLOCK_DELTA_KERNELS_HPP
#define MGARD_X_160_BLOCK_DELTA_KERNELS_HPP

#include "../../RuntimeX/RuntimeX.h"

// Portable (multi-kernel) building blocks for the BlockDelta lossless backend.
//
// Three encoding modes (mirroring cuSZp), selected per call:
//   Fixed   (0): zigzag(value)         -> fixed-length pack at per-block width
//   Delta   (1): zigzag(value-prev)    -> fixed-length pack          (default)
//   Outlier (2): Delta, but values exceeding a per-block budget width are
//                peeled into a side record list so a few large values don't
//                widen the whole block.
//
// Each block is padded to a whole byte and owns a disjoint byte range of the
// output (no atomics; trivially parallel). For Fixed/Delta a block's byte range
// is just its packed bitstream. For Outlier it is laid out as:
//   [outlier_count : 2B][main FLE : ceil(bw*len/8)B][records : oc * (2B pos +
//   sizeof(T)B value)]
// where outlier slots in the main stream store 0 and the full zigzag value
// lives in the record. Records are written in increasing position order so
// decode can patch them with an O(len+oc) single scan.
//
// All functors use a 1-thread-per-data-block mapping (the portable correctness
// reference). The CUDA/HIP fused path implements Fixed/Delta in one kernel.

namespace mgard_x_160 {

namespace block_delta {

enum : Byte { MODE_FIXED = 0, MODE_DELTA = 1, MODE_OUTLIER = 2 };

// Bytes per outlier record: 2-byte intra-block position + the value.
template <typename T> MGARDX_CONT_EXEC constexpr int outlier_rec_bytes() {
  return 2 + (int)sizeof(T);
}

template <typename T>
MGARDX_CONT_EXEC typename std::make_unsigned<T>::type zigzag(T v) {
  using UT = typename std::make_unsigned<T>::type;
  constexpr int bits = sizeof(T) * 8;
  return (UT)((v << 1) ^ (v >> (bits - 1)));
}

template <typename T>
MGARDX_CONT_EXEC T unzigzag(typename std::make_unsigned<T>::type z) {
  return (T)((z >> 1) ^ (~(z & 1) + 1)); // (z>>1) ^ -(z&1)
}

template <typename UT> MGARDX_CONT_EXEC int bit_length(UT acc) {
  int n = 0;
  while (acc) {
    n++;
    acc >>= 1;
  }
  return n;
}

// Bytes to hold `nvalues` values of `bits` bits each, padded to a byte.
MGARDX_CONT_EXEC size_t block_bytes(int bits, SIZE nvalues) {
  return (size_t)(((size_t)bits * nvalues + 7) / 8);
}

// Per-block sizing shared by host/device. Computes the block's fixed-length
// width `bw`, byte count `bc`, and (Outlier only) outlier count `oc`.
template <typename T>
MGARDX_CONT_EXEC void size_block(const T *data, SIZE start, SIZE len, Byte mode,
                                 int &bw, size_t &bc, int &oc) {
  using UT = typename std::make_unsigned<T>::type;
  bool use_delta = (mode != MODE_FIXED);
  T prev = 0;
  UT acc = 0;
  int hist[65];
  if (mode == MODE_OUTLIER) {
    for (int k = 0; k < 65; k++)
      hist[k] = 0;
  }
  for (SIZE i = 0; i < len; i++) {
    T x = data[start + i];
    UT z = zigzag<T>(use_delta ? (T)(x - prev) : x);
    prev = x;
    acc |= z;
    if (mode == MODE_OUTLIER)
      hist[bit_length<UT>(z)]++;
  }
  int bw_max = bit_length<UT>(acc);
  if (mode != MODE_OUTLIER) {
    bw = bw_max;
    oc = 0;
    bc = block_bytes(bw, len);
    return;
  }
  // suffix sums: suf[k] = #{ bit_length(z) >= k }
  int suf[66];
  suf[65] = 0;
  for (int k = 64; k >= 0; k--)
    suf[k] = suf[k + 1] + hist[k];
  int best_bw = bw_max, best_oc = 0;
  size_t best_cost = (size_t)-1;
  for (int b = 0; b <= bw_max; b++) {
    int o = suf[b + 1]; // values needing > b bits
    size_t cost = 2 + block_bytes(b, len) + (size_t)o * outlier_rec_bytes<T>();
    if (cost < best_cost) {
      best_cost = cost;
      best_bw = b;
      best_oc = o;
    }
  }
  bw = best_bw;
  oc = best_oc;
  bc = 2 + block_bytes(best_bw, len) + (size_t)best_oc * outlier_rec_bytes<T>();
}

} // namespace block_delta

// ---------------------------------------------------------------------------
// Kernel 1: per-block bit-width + byte-count (+ outlier count) -- encode
// sizing.
// ---------------------------------------------------------------------------
template <typename T, typename DeviceType>
class BlockBitwidthFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT BlockBitwidthFunctor() {}
  MGARDX_CONT BlockBitwidthFunctor(SubArray<1, T, DeviceType> data, SIZE n,
                                   SIZE block_size, SIZE nblocks, Byte mode,
                                   SubArray<1, Byte, DeviceType> bitwidth,
                                   SubArray<1, size_t, DeviceType> bytecount,
                                   SubArray<1, uint16_t, DeviceType> oc)
      : data(data), n(n), block_size(block_size), nblocks(nblocks), mode(mode),
        bitwidth(bitwidth), bytecount(bytecount), oc(oc) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE b = (FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX()) +
             FunctorBase<DeviceType>::GetThreadIdX();
    if (b >= nblocks)
      return;
    SIZE start = b * block_size;
    SIZE len = block_size < (n - start) ? block_size : (n - start);
    int bw;
    size_t bc;
    int ocount;
    block_delta::size_block<T>(data.data(), start, len, mode, bw, bc, ocount);
    *bitwidth(b) = (Byte)bw;
    *bytecount(b) = bc;
    *oc(b) = (uint16_t)ocount;
  }

  MGARDX_EXEC void Operation2() {}
  MGARDX_EXEC void Operation3() {}
  MGARDX_EXEC void Operation4() {}
  MGARDX_EXEC void Operation5() {}
  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, T, DeviceType> data;
  SIZE n, block_size, nblocks;
  Byte mode;
  SubArray<1, Byte, DeviceType> bitwidth;
  SubArray<1, size_t, DeviceType> bytecount;
  SubArray<1, uint16_t, DeviceType> oc;
};

template <typename T, typename DeviceType>
class BlockBitwidthKernel : public Kernel {
public:
  constexpr static DIM NumDim = 1;
  constexpr static bool EnableAutoTuning() { return false; }
  using DataType = T;
  constexpr static std::string_view Name = "block_delta_bitwidth";
  MGARDX_CONT BlockBitwidthKernel(SubArray<1, T, DeviceType> data, SIZE n,
                                  SIZE block_size, SIZE nblocks, Byte mode,
                                  SubArray<1, Byte, DeviceType> bitwidth,
                                  SubArray<1, size_t, DeviceType> bytecount,
                                  SubArray<1, uint16_t, DeviceType> oc)
      : data(data), n(n), block_size(block_size), nblocks(nblocks), mode(mode),
        bitwidth(bitwidth), bytecount(bytecount), oc(oc) {}

  MGARDX_CONT Task<BlockBitwidthFunctor<T, DeviceType>> GenTask(int queue_idx) {
    using FunctorType = BlockBitwidthFunctor<T, DeviceType>;
    FunctorType functor(data, n, block_size, nblocks, mode, bitwidth, bytecount,
                        oc);
    SIZE tbx = 256, tby = 1, tbz = 1;
    SIZE gridx = (nblocks - 1) / tbx + 1;
    return Task(functor, 1, 1, gridx, tbz, tby, tbx,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<1, T, DeviceType> data;
  SIZE n, block_size, nblocks;
  Byte mode;
  SubArray<1, Byte, DeviceType> bitwidth;
  SubArray<1, size_t, DeviceType> bytecount;
  SubArray<1, uint16_t, DeviceType> oc;
};

// ---------------------------------------------------------------------------
// Kernel 2: pack each block into its (disjoint) byte range of the output.
// ---------------------------------------------------------------------------
template <typename T, typename DeviceType>
class BlockPackFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT BlockPackFunctor() {}
  MGARDX_CONT BlockPackFunctor(SubArray<1, T, DeviceType> data, SIZE n,
                               SIZE block_size, SIZE nblocks, Byte mode,
                               SubArray<1, Byte, DeviceType> bitwidth,
                               SubArray<1, size_t, DeviceType> byte_offset,
                               SubArray<1, Byte, DeviceType> packed)
      : data(data), n(n), block_size(block_size), nblocks(nblocks), mode(mode),
        bitwidth(bitwidth), byte_offset(byte_offset), packed(packed) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    using UT = typename std::make_unsigned<T>::type;
    SIZE b = (FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX()) +
             FunctorBase<DeviceType>::GetThreadIdX();
    if (b >= nblocks)
      return;
    int bw = (int)*bitwidth(b);
    SIZE start = b * block_size;
    SIZE len = block_size < (n - start) ? block_size : (n - start);
    bool use_delta = (mode != block_delta::MODE_FIXED);
    bool outlier = (mode == block_delta::MODE_OUTLIER);
    Byte *out = packed(*byte_offset(b));
    size_t main_start = outlier ? 2 : 0;

    // main fixed-length stream
    T prev = 0;
    UT buf = 0;
    int cnt = 0;
    size_t pos = main_start;
    for (SIZE i = 0; i < len; i++) {
      T x = *data(start + i);
      UT z = block_delta::zigzag<T>(use_delta ? (T)(x - prev) : x);
      prev = x;
      UT sv =
          (outlier && bw < (int)(sizeof(T) * 8) && z >= ((UT)1 << bw)) ? 0 : z;
      for (int k = 0; k < bw; k++) {
        buf |= (UT)((sv >> k) & 1) << cnt;
        if (++cnt == 8) {
          out[pos++] = (Byte)(buf & 0xff);
          buf = 0;
          cnt = 0;
        }
      }
    }
    if (cnt > 0)
      out[pos++] = (Byte)(buf & 0xff);

    if (outlier) {
      // records + header
      int oc = 0;
      size_t rp = main_start + block_delta::block_bytes(bw, len);
      prev = 0;
      for (SIZE i = 0; i < len; i++) {
        T x = *data(start + i);
        UT z = block_delta::zigzag<T>((T)(x - prev));
        prev = x;
        if (bw < (int)(sizeof(T) * 8) && z >= ((UT)1 << bw)) {
          out[rp] = (Byte)(i & 0xff);
          out[rp + 1] = (Byte)((i >> 8) & 0xff);
          rp += 2;
          for (int k = 0; k < (int)sizeof(T); k++)
            out[rp + k] = (Byte)((z >> (8 * k)) & 0xff);
          rp += sizeof(T);
          oc++;
        }
      }
      out[0] = (Byte)(oc & 0xff);
      out[1] = (Byte)((oc >> 8) & 0xff);
    }
  }

  MGARDX_EXEC void Operation2() {}
  MGARDX_EXEC void Operation3() {}
  MGARDX_EXEC void Operation4() {}
  MGARDX_EXEC void Operation5() {}
  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, T, DeviceType> data;
  SIZE n, block_size, nblocks;
  Byte mode;
  SubArray<1, Byte, DeviceType> bitwidth;
  SubArray<1, size_t, DeviceType> byte_offset;
  SubArray<1, Byte, DeviceType> packed;
};

template <typename T, typename DeviceType>
class BlockPackKernel : public Kernel {
public:
  constexpr static DIM NumDim = 1;
  constexpr static bool EnableAutoTuning() { return false; }
  using DataType = T;
  constexpr static std::string_view Name = "block_delta_pack";
  MGARDX_CONT BlockPackKernel(SubArray<1, T, DeviceType> data, SIZE n,
                              SIZE block_size, SIZE nblocks, Byte mode,
                              SubArray<1, Byte, DeviceType> bitwidth,
                              SubArray<1, size_t, DeviceType> byte_offset,
                              SubArray<1, Byte, DeviceType> packed)
      : data(data), n(n), block_size(block_size), nblocks(nblocks), mode(mode),
        bitwidth(bitwidth), byte_offset(byte_offset), packed(packed) {}

  MGARDX_CONT Task<BlockPackFunctor<T, DeviceType>> GenTask(int queue_idx) {
    using FunctorType = BlockPackFunctor<T, DeviceType>;
    FunctorType functor(data, n, block_size, nblocks, mode, bitwidth,
                        byte_offset, packed);
    SIZE tbx = 256, tby = 1, tbz = 1;
    SIZE gridx = (nblocks - 1) / tbx + 1;
    return Task(functor, 1, 1, gridx, tbz, tby, tbx,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<1, T, DeviceType> data;
  SIZE n, block_size, nblocks;
  Byte mode;
  SubArray<1, Byte, DeviceType> bitwidth;
  SubArray<1, size_t, DeviceType> byte_offset;
  SubArray<1, Byte, DeviceType> packed;
};

// ---------------------------------------------------------------------------
// Kernel 3: rebuild per-block byte-count from bit-width (+oc) -- decode sizing.
// ---------------------------------------------------------------------------
template <typename T, typename DeviceType>
class BlockBytecountFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT BlockBytecountFunctor() {}
  MGARDX_CONT BlockBytecountFunctor(SIZE n, SIZE block_size, SIZE nblocks,
                                    Byte mode,
                                    SubArray<1, Byte, DeviceType> bitwidth,
                                    SubArray<1, uint16_t, DeviceType> oc,
                                    SubArray<1, size_t, DeviceType> bytecount)
      : n(n), block_size(block_size), nblocks(nblocks), mode(mode),
        bitwidth(bitwidth), oc(oc), bytecount(bytecount) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    SIZE b = (FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX()) +
             FunctorBase<DeviceType>::GetThreadIdX();
    if (b >= nblocks)
      return;
    SIZE start = b * block_size;
    SIZE len = block_size < (n - start) ? block_size : (n - start);
    int bw = (int)*bitwidth(b);
    if (mode == block_delta::MODE_OUTLIER) {
      *bytecount(b) = 2 + block_delta::block_bytes(bw, len) +
                      (size_t)(*oc(b)) * block_delta::outlier_rec_bytes<T>();
    } else {
      *bytecount(b) = block_delta::block_bytes(bw, len);
    }
  }

  MGARDX_EXEC void Operation2() {}
  MGARDX_EXEC void Operation3() {}
  MGARDX_EXEC void Operation4() {}
  MGARDX_EXEC void Operation5() {}
  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SIZE n, block_size, nblocks;
  Byte mode;
  SubArray<1, Byte, DeviceType> bitwidth;
  SubArray<1, uint16_t, DeviceType> oc;
  SubArray<1, size_t, DeviceType> bytecount;
};

template <typename T, typename DeviceType>
class BlockBytecountKernel : public Kernel {
public:
  constexpr static DIM NumDim = 1;
  constexpr static bool EnableAutoTuning() { return false; }
  using DataType = Byte;
  constexpr static std::string_view Name = "block_delta_bytecount";
  MGARDX_CONT BlockBytecountKernel(SIZE n, SIZE block_size, SIZE nblocks,
                                   Byte mode,
                                   SubArray<1, Byte, DeviceType> bitwidth,
                                   SubArray<1, uint16_t, DeviceType> oc,
                                   SubArray<1, size_t, DeviceType> bytecount)
      : n(n), block_size(block_size), nblocks(nblocks), mode(mode),
        bitwidth(bitwidth), oc(oc), bytecount(bytecount) {}

  MGARDX_CONT Task<BlockBytecountFunctor<T, DeviceType>>
  GenTask(int queue_idx) {
    using FunctorType = BlockBytecountFunctor<T, DeviceType>;
    FunctorType functor(n, block_size, nblocks, mode, bitwidth, oc, bytecount);
    SIZE tbx = 256, tby = 1, tbz = 1;
    SIZE gridx = (nblocks - 1) / tbx + 1;
    return Task(functor, 1, 1, gridx, tbz, tby, tbx,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SIZE n, block_size, nblocks;
  Byte mode;
  SubArray<1, Byte, DeviceType> bitwidth;
  SubArray<1, uint16_t, DeviceType> oc;
  SubArray<1, size_t, DeviceType> bytecount;
};

// ---------------------------------------------------------------------------
// Kernel 4: unpack each block back into the signed quantized stream.
// ---------------------------------------------------------------------------
template <typename T, typename DeviceType>
class BlockUnpackFunctor : public Functor<DeviceType> {
public:
  MGARDX_CONT BlockUnpackFunctor() {}
  MGARDX_CONT BlockUnpackFunctor(SubArray<1, Byte, DeviceType> packed, SIZE n,
                                 SIZE block_size, SIZE nblocks, Byte mode,
                                 SubArray<1, Byte, DeviceType> bitwidth,
                                 SubArray<1, size_t, DeviceType> byte_offset,
                                 SubArray<1, T, DeviceType> data)
      : packed(packed), n(n), block_size(block_size), nblocks(nblocks),
        mode(mode), bitwidth(bitwidth), byte_offset(byte_offset), data(data) {
    Functor<DeviceType>();
  }

  MGARDX_EXEC void Operation1() {
    using UT = typename std::make_unsigned<T>::type;
    SIZE b = (FunctorBase<DeviceType>::GetBlockIdX() *
              FunctorBase<DeviceType>::GetBlockDimX()) +
             FunctorBase<DeviceType>::GetThreadIdX();
    if (b >= nblocks)
      return;
    int bw = (int)*bitwidth(b);
    SIZE start = b * block_size;
    SIZE len = block_size < (n - start) ? block_size : (n - start);
    bool use_delta = (mode != block_delta::MODE_FIXED);
    bool outlier = (mode == block_delta::MODE_OUTLIER);
    Byte *in = packed(*byte_offset(b));

    size_t main_start = 0;
    int oc = 0;
    if (outlier) {
      oc = (int)in[0] | ((int)in[1] << 8);
      main_start = 2;
    }
    size_t rec_start = main_start + block_delta::block_bytes(bw, len);
    constexpr int RECSZ = 2 + (int)sizeof(T);
    int cursor = 0;
    long next_pos = -1;
    if (outlier && oc > 0)
      next_pos = (long)in[rec_start] | ((long)in[rec_start + 1] << 8);

    T prev = 0;
    Byte cur = 0;
    int cnt = 0;
    size_t pos = main_start;
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
      if (outlier && (long)i == next_pos) {
        size_t ro = rec_start + (size_t)cursor * RECSZ + 2;
        UT v = 0;
        for (int k = 0; k < (int)sizeof(T); k++)
          v |= (UT)in[ro + k] << (8 * k);
        z = v;
        cursor++;
        next_pos =
            (cursor < oc)
                ? ((long)in[rec_start + (size_t)cursor * RECSZ] |
                   ((long)in[rec_start + (size_t)cursor * RECSZ + 1] << 8))
                : -1;
      }
      T d = block_delta::unzigzag<T>(z);
      prev = use_delta ? (T)(prev + d) : d;
      *data(start + i) = prev;
    }
  }

  MGARDX_EXEC void Operation2() {}
  MGARDX_EXEC void Operation3() {}
  MGARDX_EXEC void Operation4() {}
  MGARDX_EXEC void Operation5() {}
  MGARDX_CONT size_t shared_memory_size() { return 0; }

private:
  SubArray<1, Byte, DeviceType> packed;
  SIZE n, block_size, nblocks;
  Byte mode;
  SubArray<1, Byte, DeviceType> bitwidth;
  SubArray<1, size_t, DeviceType> byte_offset;
  SubArray<1, T, DeviceType> data;
};

template <typename T, typename DeviceType>
class BlockUnpackKernel : public Kernel {
public:
  constexpr static DIM NumDim = 1;
  constexpr static bool EnableAutoTuning() { return false; }
  using DataType = T;
  constexpr static std::string_view Name = "block_delta_unpack";
  MGARDX_CONT BlockUnpackKernel(SubArray<1, Byte, DeviceType> packed, SIZE n,
                                SIZE block_size, SIZE nblocks, Byte mode,
                                SubArray<1, Byte, DeviceType> bitwidth,
                                SubArray<1, size_t, DeviceType> byte_offset,
                                SubArray<1, T, DeviceType> data)
      : packed(packed), n(n), block_size(block_size), nblocks(nblocks),
        mode(mode), bitwidth(bitwidth), byte_offset(byte_offset), data(data) {}

  MGARDX_CONT Task<BlockUnpackFunctor<T, DeviceType>> GenTask(int queue_idx) {
    using FunctorType = BlockUnpackFunctor<T, DeviceType>;
    FunctorType functor(packed, n, block_size, nblocks, mode, bitwidth,
                        byte_offset, data);
    SIZE tbx = 256, tby = 1, tbz = 1;
    SIZE gridx = (nblocks - 1) / tbx + 1;
    return Task(functor, 1, 1, gridx, tbz, tby, tbx,
                functor.shared_memory_size(), queue_idx, std::string(Name));
  }

private:
  SubArray<1, Byte, DeviceType> packed;
  SIZE n, block_size, nblocks;
  Byte mode;
  SubArray<1, Byte, DeviceType> bitwidth;
  SubArray<1, size_t, DeviceType> byte_offset;
  SubArray<1, T, DeviceType> data;
};

} // namespace mgard_x_160

#endif
