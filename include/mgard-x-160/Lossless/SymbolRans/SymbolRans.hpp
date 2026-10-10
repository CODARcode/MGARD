/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_SYMBOL_RANS_TEMPLATE_HPP
#define MGARD_X_160_SYMBOL_RANS_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"
#include "../ParallelHuffman/OutlierSeparator.hpp"
#include "../rANS/Rans.hpp"

namespace mgard_x_160 {

// Symbol-alphabet rANS lossless backend. Reuses Huffman's outlier front-end
// (values outside [0, dict_size) stored sparsely and zeroed in place) then
// entropy-codes the primary symbol stream with rANS over the dict_size
// alphabet. On the same symbol model rANS codes fractional bits, so its output
// is always <= Huffman's (no integer-code-length rounding loss) and uncapped --
// the architecturally correct way to beat Huffman on ratio. Q = unsigned
// quantized type, S = signed.
template <typename Q, typename S, typename DeviceType> class SymbolRans {
public:
  SymbolRans() : initialized(false) {}

  // SYMBOL_RANS_SEGMENT can be overridden at compile time to sweep the rANS
  // segment size (smaller -> more parallel streams / occupancy, but more
  // per-segment overhead -> lower CR).
#ifndef SYMBOL_RANS_SEGMENT
#define SYMBOL_RANS_SEGMENT 0
#endif

  void Resize(SIZE n, int dict_size, double outlier_ratio, int queue_idx) {
    initialized = true;
    this->max_size = n;
    this->dict_size = dict_size;
    SIZE max_outliers = (SIZE)((double)n * outlier_ratio) + 1;
    outlier_count_d.resize({1}, queue_idx);
    outlier_idx_d.resize({max_outliers}, queue_idx);
    outlier_val_d.resize({max_outliers}, queue_idx);
    // NOTE: the shared-stream interleaved layout (interleaved=true) is the
    // foundation for warp-cooperative coalesced encode, but its CUDA fast-path
    // (warp kernels) is not implemented yet -- the sequential interleaved
    // kernels would be a severe CUDA regression (one thread per block). So the
    // symbol path uses the validated per-stream layout (block-interleaved
    // symbol mapping already gives coalesced reads + a decode that beats
    // Huffman). Flip to true once the warp kernels land.
    rans.Resize(n, dict_size, queue_idx, (SIZE)SYMBOL_RANS_SEGMENT,
                /*interleaved=*/false);
    MemoryManager<DeviceType>::MallocHost(signature_verify, 7 * sizeof(char),
                                          queue_idx);
  }

  void Compress(Array<1, S, DeviceType> &original_data,
                Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    SIZE n = original_data.shape(0);

    // Separate out-of-dictionary outliers (zeroed in place in original_data).
    ATOMIC_IDX zero = 0;
    MemoryManager<DeviceType>::Copy1D(outlier_count_d.data(), &zero, 1,
                                      queue_idx);
    DeviceLauncher<DeviceType>::Execute(
        OutlierSeparatorKernel<S, MGARDX_SEPARATE_OUTLIER, DeviceType>(
            SubArray(original_data), dict_size, SubArray(outlier_count_d),
            SubArray(outlier_idx_d), SubArray(outlier_val_d)),
        queue_idx);
    MemoryManager<DeviceType>::Copy1D(&outlier_count, outlier_count_d.data(), 1,
                                      queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    if (outlier_count > outlier_val_d.shape(0)) {
      throw std::runtime_error("SymbolRans: not enough outlier workspace.");
    }

    // The in-range primary stream (outliers now zero) reinterpreted as
    // unsigned.
    Array<1, Q, DeviceType> primary({n}, (Q *)original_data.data());
    rans.Compress(primary, rans_stream, queue_idx);
    SIZE rans_bytes = rans_stream.shape(0);

    // Layout: signature | n | dict_size | outlier_count | rans_bytes |
    //         outlier_idx[] | outlier_val[] | rANS stream
    SIZE byte_offset = 0;
    advance_with_align<Byte>(byte_offset, 7);
    advance_with_align<SIZE>(byte_offset, 1);
    advance_with_align<SIZE>(byte_offset, 1);
    advance_with_align<SIZE>(byte_offset, 1);
    advance_with_align<SIZE>(byte_offset, 1);
    advance_with_align<ATOMIC_IDX>(byte_offset, outlier_count);
    advance_with_align<S>(byte_offset, outlier_count);
    advance_with_align<Byte>(byte_offset, rans_bytes);

    compressed_data.resize({byte_offset}, queue_idx);
    SubArray<1, Byte, DeviceType> cs(compressed_data);

    SIZE n_s = n, dict_s = dict_size, oc_s = outlier_count, rb_s = rans_bytes;
    byte_offset = 0;
    SerializeArray<Byte>(cs, signature, 7, byte_offset, queue_idx);
    SerializeArray<SIZE>(cs, &n_s, 1, byte_offset, queue_idx);
    SerializeArray<SIZE>(cs, &dict_s, 1, byte_offset, queue_idx);
    SerializeArray<SIZE>(cs, &oc_s, 1, byte_offset, queue_idx);
    SerializeArray<SIZE>(cs, &rb_s, 1, byte_offset, queue_idx);
    SerializeArray<ATOMIC_IDX>(cs, outlier_idx_d.data(), outlier_count,
                               byte_offset, queue_idx);
    SerializeArray<S>(cs, outlier_val_d.data(), outlier_count, byte_offset,
                      queue_idx);
    align_byte_offset<Byte>(byte_offset);
    MemoryManager<DeviceType>::Copy1D(compressed_data.data() + byte_offset,
                                      rans_stream.data(), rans_bytes,
                                      queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  }

  void Serialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {}
  void Deserialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
  }

  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  Array<1, S, DeviceType> &decompressed_data, int queue_idx) {
    SubArray<1, Byte, DeviceType> cs(compressed_data);
    Byte *sig = nullptr;
    SIZE n_s, dict_s, oc_s, rb_s;
    SIZE *p_n = &n_s, *p_d = &dict_s, *p_oc = &oc_s, *p_rb = &rb_s;
    SIZE byte_offset = 0;
    DeserializeArray<Byte>(cs, sig, 7, byte_offset, true, queue_idx);
    DeserializeArray<SIZE>(cs, p_n, 1, byte_offset, false, queue_idx);
    DeserializeArray<SIZE>(cs, p_d, 1, byte_offset, false, queue_idx);
    DeserializeArray<SIZE>(cs, p_oc, 1, byte_offset, false, queue_idx);
    DeserializeArray<SIZE>(cs, p_rb, 1, byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    SIZE n = n_s;
    dict_size = dict_s;
    outlier_count = oc_s;
    SIZE rans_bytes = rb_s;

    ATOMIC_IDX *oidx_ptr = nullptr;
    S *oval_ptr = nullptr;
    DeserializeArray<ATOMIC_IDX>(cs, oidx_ptr, outlier_count, byte_offset, true,
                                 queue_idx);
    DeserializeArray<S>(cs, oval_ptr, outlier_count, byte_offset, true,
                        queue_idx);
    align_byte_offset<Byte>(byte_offset);
    Array<1, Byte, DeviceType> rans_alias({rans_bytes},
                                          compressed_data.data() + byte_offset);

    decompressed_data.resize({n}, queue_idx);
    Array<1, Q, DeviceType> primary({n}, (Q *)decompressed_data.data());
    rans.Deserialize(rans_alias, queue_idx);
    rans.Decompress(rans_alias, primary, queue_idx);

    // Scatter the outliers back into the decoded stream.
    SubArray<1, ATOMIC_IDX, DeviceType> oidx({(SIZE)outlier_count}, oidx_ptr);
    SubArray<1, S, DeviceType> oval({(SIZE)outlier_count}, oval_ptr);
    DeviceLauncher<DeviceType>::Execute(
        OutlierSeparatorKernel<S, MGARDX_RESTORE_OUTLIER, DeviceType>(
            SubArray(decompressed_data), dict_size, SubArray(outlier_count_d),
            oidx, oval),
        queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  }

  static size_t EstimateMemoryFootprint(SIZE n, int dict_size,
                                        double outlier_ratio) {
    size_t size = 0;
    size += (size_t)(n * outlier_ratio) * (sizeof(ATOMIC_IDX) + sizeof(S));
    size += (size_t)n * 2 + 16 * ((size_t)n / 2048 + 1); // rANS scratch ~2x
    return size;
  }

  bool initialized;
  SIZE max_size = 0;
  int dict_size = 0;
  ATOMIC_IDX outlier_count = 0;
  Byte signature[7] = {'M', 'G', 'X', 'S', 'R', 'A', 'N'};
  Byte *signature_verify;

  rans::Rans<Q, DeviceType> rans;
  Array<1, Byte, DeviceType> rans_stream;
  Array<1, ATOMIC_IDX, DeviceType> outlier_count_d;
  Array<1, ATOMIC_IDX, DeviceType> outlier_idx_d;
  Array<1, S, DeviceType> outlier_val_d;
};

} // namespace mgard_x_160
#endif
