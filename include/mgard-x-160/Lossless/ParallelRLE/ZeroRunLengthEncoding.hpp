/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_ZERO_RUN_LENGTH_ENCODING_TEMPLATE_HPP
#define MGARD_X_160_ZERO_RUN_LENGTH_ENCODING_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"
#include "ZeroDecode.hpp"
#include "ZeroEncode.hpp"
#include "ZeroStartMarks.hpp"
#include "ZeroStartPositions.hpp"

namespace mgard_x_160 {

namespace parallel_rle {

// Zero-run-length encoding (RLE0). Only zeros are run-encoded; nonzero values
// are stored as literals together with the count of zeros that immediately
// precede each one. Trailing zeros are implied by the stored original length.
// This is the sparse-data front end intended to feed an entropy backend (e.g.
// Huffman/rANS) on the two output streams (counts, symbols) separately.
//
// Limitation: a single zero gap must fit in C_run (use uint32_t unless gaps can
// exceed 2^32). Unlike the full RLE, gaps are not split at C_run boundaries.
template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class ZeroRunLengthEncoding {
public:
  ZeroRunLengthEncoding() : initialized(false) {}

  ZeroRunLengthEncoding(SIZE max_size)
      : initialized(true), max_size(max_size) {}

  void Resize(SIZE max_size, int queue_idx) {
    this->initialized = true;
    this->max_size = max_size;
    start_marks.resize({max_size}, queue_idx);
    scanned_start_marks.resize({max_size}, queue_idx);
    start_positions.resize({max_size}, queue_idx);
    MemoryManager<DeviceType>::MallocHost(signature_verify, 7 * sizeof(char),
                                          queue_idx);
    DeviceCollective<DeviceType>::ScanSumInclusive(
        max_size, SubArray<1, C_global, DeviceType>(),
        SubArray<1, C_global, DeviceType>(), this->scan_workspace, false,
        queue_idx);
  }

  // Run the mark + scan stages and read back the number of nonzeros (= number
  // of stored symbols). Leaves scanned_start_marks populated for the caller.
  C_global CountSymbols(Array<1, T_symbol, DeviceType> &original_data,
                        int queue_idx) {
    SIZE original_length = original_data.shape(0);

    start_marks.resize({original_length}, queue_idx);
    scanned_start_marks.resize({original_length}, queue_idx);
    start_positions.resize({original_length}, queue_idx);

    DeviceLauncher<DeviceType>::Execute(
        ZeroStartMarksKernel<T_symbol, C_run, C_global, DeviceType>(
            SubArray(original_data), SubArray(start_marks)),
        queue_idx);

    DeviceCollective<DeviceType>::ScanSumInclusive(
        original_length, SubArray(start_marks), SubArray(scanned_start_marks),
        scan_workspace, true, queue_idx);

    C_global _num_symbols = 0;
    MemoryManager<DeviceType>::Copy1D(
        &_num_symbols, scanned_start_marks.data() + original_length - 1, 1,
        queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    return _num_symbols;
  }

  double EstimateCR(Array<1, T_symbol, DeviceType> &original_data,
                    int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE original_length = original_data.shape(0);
    C_global _num_symbols = CountSymbols(original_data, queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Zero RLE estimate CR", original_length * sizeof(T_symbol));
      timer.clear();
    }

    return (double)(original_length * sizeof(T_symbol)) /
           (_num_symbols * (sizeof(T_symbol) + sizeof(C_run)) + 30);
  }

  bool Compress(Array<1, T_symbol, DeviceType> &original_data,
                Array<1, Byte, DeviceType> &compressed_data, float target_cr,
                int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE original_length = original_data.shape(0);
    C_global _num_symbols = CountSymbols(original_data, queue_idx);

    if (target_cr > 0) {
      double est_cr = (double)(original_length * sizeof(T_symbol)) /
                      (_num_symbols * (sizeof(T_symbol) + sizeof(C_run)) + 30);
      log::info("Zero RLE estimated CR: " + std::to_string(est_cr) +
                " (target: " + std::to_string(target_cr) + ")");
      if (est_cr < target_cr) {
        return false;
      }
    }

    num_symbols = _num_symbols;

    // Compact nonzero positions, then derive (gap, value) pairs from them.
    if (num_symbols > 0) {
      DeviceLauncher<DeviceType>::Execute(
          ZeroStartPositionsKernel<T_symbol, C_run, C_global, DeviceType>(
              SubArray(scanned_start_marks), SubArray(start_positions)),
          queue_idx);
    }

    SIZE byte_offset = 0;
    advance_with_align<Byte>(byte_offset, 7); // signature
    advance_with_align<SIZE>(byte_offset, 1); // num_symbols
    advance_with_align<SIZE>(byte_offset, 1); // original_length
    advance_with_align<C_run>(byte_offset, num_symbols);
    advance_with_align<T_symbol>(byte_offset, num_symbols);

    SIZE output_size = byte_offset;
    compressed_data.resize({output_size}, queue_idx);
    SubArray<1, Byte, DeviceType> compressed_subarray(compressed_data);

    byte_offset = 0;
    SerializeArray<Byte>(compressed_subarray, signature, 7, byte_offset,
                         queue_idx);
    SerializeArray<SIZE>(compressed_subarray, &num_symbols, 1, byte_offset,
                         queue_idx);
    SerializeArray<SIZE>(compressed_subarray, &original_length, 1, byte_offset,
                         queue_idx);

    SubArray<1, C_run, DeviceType> counts(
        {num_symbols}, (C_run *)(compressed_data.data() + byte_offset));
    advance_with_align<C_run>(byte_offset, num_symbols);
    SubArray<1, T_symbol, DeviceType> symbols(
        {num_symbols}, (T_symbol *)(compressed_data.data() + byte_offset));
    advance_with_align<T_symbol>(byte_offset, num_symbols);

    if (num_symbols > 0) {
      DeviceLauncher<DeviceType>::Execute(
          ZeroEncodeKernel<T_symbol, C_run, C_global, DeviceType>(
              num_symbols, SubArray(original_data), SubArray(start_positions),
              counts, symbols),
          queue_idx);
    }

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      log::info("Zero RLE compression ratio: " +
                std::to_string(original_length * sizeof(T_symbol)) + "/" +
                std::to_string(compressed_data.shape(0)) + " (" +
                std::to_string((double)original_length * sizeof(T_symbol) /
                               compressed_data.shape(0)) +
                ")");
      timer.end();
      timer.print("Zero RLE compress", original_length * sizeof(T_symbol));
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
      throw std::runtime_error("Zero RLE signature mismatch.");
    }
    SubArray<1, Byte, DeviceType> compressed_subarray(compressed_data);
    Byte *signature_ptr = nullptr;
    num_symbols_ptr = &num_symbols;
    original_length_ptr = &original_length;
    SIZE byte_offset = 0;
    DeserializeArray<Byte>(compressed_subarray, signature_ptr, 7, byte_offset,
                           true, queue_idx);
    DeserializeArray<SIZE>(compressed_subarray, num_symbols_ptr, 1, byte_offset,
                           false, queue_idx);
    DeserializeArray<SIZE>(compressed_subarray, original_length_ptr, 1,
                           byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    DeserializeArray<C_run>(compressed_subarray, counts_ptr, num_symbols,
                            byte_offset, true, queue_idx);
    DeserializeArray<T_symbol>(compressed_subarray, symbols_ptr, num_symbols,
                               byte_offset, true, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
  }

  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  Array<1, T_symbol, DeviceType> &decompressed_data,
                  int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SubArray<1, C_run, DeviceType> counts({(SIZE)num_symbols}, counts_ptr);
    SubArray<1, T_symbol, DeviceType> symbols({(SIZE)num_symbols}, symbols_ptr);
    decompressed_data.resize({(SIZE)original_length}, queue_idx);

    // Trailing/interior zeros are never written, so start from an all-zero
    // buffer and scatter only the nonzeros.
    MemoryManager<DeviceType>::Memset1D(decompressed_data.data(),
                                        original_length, 0, queue_idx);

    if (num_symbols > 0) {
      // strides[i] = counts[i] + 1; inclusive scan -> position[i] + 1.
      SubArray<1, C_global, DeviceType> strides(start_marks);
      DeviceLauncher<DeviceType>::Execute(
          ZeroStrideKernel<T_symbol, C_run, C_global, DeviceType>(counts,
                                                                  strides),
          queue_idx);

      DeviceCollective<DeviceType>::ScanSumInclusive(
          num_symbols, strides, SubArray(start_positions), scan_workspace, true,
          queue_idx);

      DeviceLauncher<DeviceType>::Execute(
          ZeroScatterKernel<T_symbol, C_run, C_global, DeviceType>(
              symbols, SubArray(start_positions), SubArray(decompressed_data)),
          queue_idx);
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Zero RLE decompress", original_length * sizeof(T_symbol));
      timer.clear();
    }
  }

  bool initialized;
  SIZE max_size;
  SIZE original_length = 0;
  SIZE num_symbols = 0;
  SIZE *num_symbols_ptr = nullptr;
  SIZE *original_length_ptr = nullptr;
  C_run *counts_ptr = nullptr;
  T_symbol *symbols_ptr = nullptr;
  Byte signature[7] = {'M', 'G', 'X', 'Z', 'R', 'L', '0'};
  Byte *signature_verify;

  Array<1, C_global, DeviceType> start_marks;
  Array<1, C_global, DeviceType> scanned_start_marks;
  Array<1, C_global, DeviceType> start_positions;
  Array<1, Byte, DeviceType> scan_workspace;
};

} // namespace parallel_rle
} // namespace mgard_x_160
#endif
