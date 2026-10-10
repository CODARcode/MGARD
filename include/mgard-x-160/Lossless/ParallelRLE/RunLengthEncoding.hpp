/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_160_RUN_LENGTH_ENCODING_TEMPLATE_HPP
#define MGARD_X_160_RUN_LENGTH_ENCODING_TEMPLATE_HPP

#include "../../RuntimeX/RuntimeX.h"
#include "Convert.hpp"
#include "Decode.hpp"
#include "Encode.hpp"
#include "StartMarks.hpp"
#include "StartPositions.hpp"

namespace mgard_x_160 {

namespace parallel_rle {

template <typename T_symbol, typename C_run, typename C_global,
          typename DeviceType>
class RunLengthEncoding {
public:
  RunLengthEncoding() : initialized(false) {}

  RunLengthEncoding(SIZE max_size) : initialized(true), max_size(max_size) {}

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

  static size_t EstimateMemoryFootprint(SIZE n) {
    size_t memory_footprint = 0;
    memory_footprint += n * sizeof(SIZE) * 3;
    Array<1, Byte, DeviceType> tmp_workspace;
    DeviceCollective<DeviceType>::ScanSumInclusive(
        n, SubArray<1, C_global, DeviceType>(),
        SubArray<1, C_global, DeviceType>(), tmp_workspace, false, 0);
    memory_footprint += tmp_workspace.shape(0);
    return 0;
  }

  double EstimateCR(Array<1, T_symbol, DeviceType> &original_data,
                    int queue_idx) {
    Timer timer;
    // Timer timer_each;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SIZE original_length = original_data.shape(0);

    start_marks.resize({original_length}, queue_idx);
    scanned_start_marks.resize({original_length}, queue_idx);
    start_positions.resize({original_length}, queue_idx);

    // timer_each.start();

    DeviceLauncher<DeviceType>::Execute(
        StartMarksKernel<T_symbol, C_run, C_global, DeviceType>(
            SubArray(original_data), SubArray(start_marks)),
        queue_idx);

    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // timer_each.end(); timer_each.print("StartMarksKernel", original_length *
    // sizeof(T_symbol)); timer_each.clear(); timer_each.start();
    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // PrintSubarray("StartMarksKernel", SubArray(start_marks));

    DeviceCollective<DeviceType>::ScanSumInclusive(
        original_length, SubArray(start_marks), SubArray(scanned_start_marks),
        scan_workspace, true, queue_idx);

    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // timer_each.end(); timer_each.print("ScanSumInclusive", original_length *
    // sizeof(T_symbol)); timer_each.clear(); timer_each.start();

    C_global _total_run_length = 0;
    MemoryManager<DeviceType>::Copy1D(
        &_total_run_length, scanned_start_marks.data() + original_length - 1, 1,
        queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("RLE estimate CR", original_length * sizeof(T_symbol));
      timer.clear();
    }

    return (double)(original_length * sizeof(T_symbol)) /
           (_total_run_length * (sizeof(T_symbol) + sizeof(C_run)) + 30);
  }

  bool Compress(Array<1, T_symbol, DeviceType> &original_data,
                Array<1, Byte, DeviceType> &compressed_data, float target_cr,
                int queue_idx) {
    Timer timer;
    // Timer timer_each;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    // PrintSubarray("original_data", SubArray(original_data));

    SIZE original_length = original_data.shape(0);

    start_marks.resize({original_length}, queue_idx);
    scanned_start_marks.resize({original_length}, queue_idx);
    start_positions.resize({original_length}, queue_idx);

    // timer_each.start();

    DeviceLauncher<DeviceType>::Execute(
        StartMarksKernel<T_symbol, C_run, C_global, DeviceType>(
            SubArray(original_data), SubArray(start_marks)),
        queue_idx);

    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // timer_each.end(); timer_each.print("StartMarksKernel", original_length *
    // sizeof(T_symbol)); timer_each.clear(); timer_each.start();
    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // PrintSubarray("StartMarksKernel", SubArray(start_marks));

    DeviceCollective<DeviceType>::ScanSumInclusive(
        original_length, SubArray(start_marks), SubArray(scanned_start_marks),
        scan_workspace, true, queue_idx);

    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // timer_each.end(); timer_each.print("ScanSumInclusive", original_length *
    // sizeof(T_symbol)); timer_each.clear(); timer_each.start();

    C_global _total_run_length = 0;
    MemoryManager<DeviceType>::Copy1D(
        &_total_run_length, scanned_start_marks.data() + original_length - 1, 1,
        queue_idx);

    if (target_cr > 0) {
      double est_cr =
          (double)(original_length * sizeof(T_symbol)) /
          (_total_run_length * (sizeof(T_symbol) + sizeof(C_run)) + 30);
      log::info("RLE estimated CR: " + std::to_string(est_cr) +
                " (target: " + std::to_string(target_cr) + ")");
      if (est_cr < target_cr) {
        return false;
      }
    }

    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // PrintSubarray("scanned_start_marks", SubArray(scanned_start_marks));
    DeviceLauncher<DeviceType>::Execute(
        StartPositionsKernel<T_symbol, C_run, C_global, DeviceType>(
            SubArray(scanned_start_marks), SubArray(start_positions)),
        queue_idx);

    // wait for total_run_length to be copied
    total_run_length = _total_run_length;
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // timer_each.end(); timer_each.print("StartPositionsKernel",
    // original_length * sizeof(T_symbol)); timer_each.clear();
    // timer_each.start();
    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // PrintSubarray("start_positions", SubArray(start_positions));

    SIZE byte_offset = 0;
    advance_with_align<Byte>(byte_offset, 7); // signature
    advance_with_align<SIZE>(byte_offset, 1);
    advance_with_align<SIZE>(byte_offset, 1);
    advance_with_align<C_run>(byte_offset, total_run_length);
    advance_with_align<T_symbol>(byte_offset, total_run_length);

    SIZE output_size = byte_offset;
    compressed_data.resize({output_size}, queue_idx);
    SubArray<1, Byte, DeviceType> compressed_subarray(compressed_data);

    byte_offset = 0;
    SerializeArray<Byte>(compressed_subarray, signature, 7, byte_offset,
                         queue_idx);
    SerializeArray<SIZE>(compressed_subarray, &total_run_length, 1, byte_offset,
                         queue_idx);
    SerializeArray<SIZE>(compressed_subarray, &original_length, 1, byte_offset,
                         queue_idx);

    SubArray<1, C_run, DeviceType> counts(
        {total_run_length}, (C_run *)(compressed_data.data() + byte_offset));
    advance_with_align<C_run>(byte_offset, total_run_length);
    SubArray<1, T_symbol, DeviceType> symbols(
        {total_run_length}, (T_symbol *)(compressed_data.data() + byte_offset));
    advance_with_align<T_symbol>(byte_offset, total_run_length);

    DeviceLauncher<DeviceType>::Execute(
        EncodeKernel<T_symbol, C_run, C_global, DeviceType>(
            total_run_length, SubArray(original_data),
            SubArray(start_positions), counts, symbols),
        queue_idx);

    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // timer_each.end(); timer_each.print("EncodeKernel", original_length *
    // sizeof(T_symbol)); timer_each.clear(); timer_each.start();

    // PrintSubarray("counts", counts);
    // PrintSubarray("symbols", symbols);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      log::info("RLE compression ratio: " +
                std::to_string(original_length * sizeof(T_symbol)) + "/" +
                std::to_string(compressed_data.shape(0)) + " (" +
                std::to_string((double)original_length * sizeof(T_symbol) /
                               compressed_data.shape(0)) +
                ")");
      timer.print("RLE compress", original_length * sizeof(T_symbol));
      timer.clear();
    }

    return true;

    // C_run * counts_host = new C_run[total_run_length];
    // MemoryManager<DeviceType>::Copy1D(counts_host, counts.data(),
    // total_run_length, queue_idx);
    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // C_run max_run = 0;
    // for (int i = 0; i < total_run_length; i++) {
    //   if (counts_host[i] > max_run) {
    //     max_run = counts_host[i];
    //   }
    // }

    // std::cout << "total_run_length: " << total_run_length << std::endl;
    // std::cout << "max_run: " << max_run << std::endl;
    // IDX MAX_RUN = (IDX)1u << sizeof(C_run) * 8;
    // std::cout << "max_run allowed: " << MAX_RUN << std::endl;
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
      throw std::runtime_error("RLE signature mismatch.");
    }
    SubArray<1, Byte, DeviceType> compressed_subarray(compressed_data);
    Byte *signature_ptr = nullptr;
    total_run_length_ptr = &total_run_length;
    original_length_ptr = &original_length;
    SIZE byte_offset = 0;
    DeserializeArray<Byte>(compressed_subarray, signature_ptr, 7, byte_offset,
                           true, queue_idx);
    DeserializeArray<SIZE>(compressed_subarray, total_run_length_ptr, 1,
                           byte_offset, false, queue_idx);
    DeserializeArray<SIZE>(compressed_subarray, original_length_ptr, 1,
                           byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    DeserializeArray<C_run>(compressed_subarray, counts_ptr, total_run_length,
                            byte_offset, true, queue_idx);
    DeserializeArray<T_symbol>(compressed_subarray, symbols_ptr,
                               total_run_length, byte_offset, true, queue_idx);

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // PrintSubarray("counts", SubArray<1, C_run,
    // DeviceType>({total_run_length}, counts_ptr)); PrintSubarray("symbols",
    // SubArray<1, T_symbol, DeviceType>({total_run_length}, symbols_ptr));
    // std::cout << "total_run_length: " << total_run_length << std::endl;
    // std::cout << "original_length: " << original_length << std::endl;
  }

  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  Array<1, T_symbol, DeviceType> &decompressed_data,
                  int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SubArray<1, Byte, DeviceType> compressed_subarray(compressed_data);
    SubArray<1, C_run, DeviceType> counts({total_run_length}, counts_ptr);
    SubArray<1, T_symbol, DeviceType> symbols({total_run_length}, symbols_ptr);
    decompressed_data.resize({(SIZE)original_length}, queue_idx);

    // reuse array
    SubArray counts_SIZE(start_marks);
    DeviceLauncher<DeviceType>::Execute(
        ConvertKernel<T_symbol, C_run, C_global, DeviceType>(counts,
                                                             counts_SIZE),
        queue_idx);

    DeviceCollective<DeviceType>::ScanSumInclusive(
        total_run_length, counts_SIZE, SubArray(start_positions),
        scan_workspace, true, queue_idx);

    //   DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // PrintSubarray("start_positions", SubArray(start_positions));

    DeviceLauncher<DeviceType>::Execute(
        DecodeKernel<T_symbol, C_run, C_global, DeviceType>(
            counts, symbols, SubArray(start_positions),
            SubArray(decompressed_data)),
        queue_idx);

    // DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // PrintSubarray("decompressed_data", SubArray(decompressed_data));

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("RLE decompress", original_length * sizeof(T_symbol));
      timer.clear();
    }
  }

  bool initialized;
  SIZE max_size;
  SIZE original_length = 0;
  SIZE total_run_length = 0;
  SIZE *total_run_length_ptr = nullptr;
  SIZE *original_length_ptr = nullptr;
  C_run *counts_ptr = nullptr;
  T_symbol *symbols_ptr = nullptr;
  Byte signature[7] = {'M', 'G', 'X', 'R', 'L', 'E', 'C'};
  Byte *signature_verify;

  Array<1, C_global, DeviceType> start_marks;
  Array<1, C_global, DeviceType> scanned_start_marks;
  Array<1, C_global, DeviceType> start_positions;
  Array<1, Byte, DeviceType> scan_workspace;
};

} // namespace parallel_rle
} // namespace mgard_x_160
#endif