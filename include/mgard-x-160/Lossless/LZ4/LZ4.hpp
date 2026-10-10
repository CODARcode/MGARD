/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_LZ4_HPP
#define MGARD_X_160_LZ4_HPP

#include "../../RuntimeX/Utilities/Serializer.hpp"
#include "LZ4Fused.hpp" // CUDA warp-per-chunk compress kernel (optional fast path)
#include "LZ4Kernels.hpp"

namespace mgard_x_160 {

// Portable, nvcomp-free LZ4 (warp-cooperative on GPU via LZ4Fused, the size-1
// SubGroupScalar functor on CPU). In-place Compress/Decompress(Array<1,Byte>&):
// the composed pipeline feeds it the Huffman output (Huffman_LZ4) or the raw
// quantized byte stream (standalone LZ4), and gets it back. The buffer is
// fully self-describing -- Compress writes the header; Decompress parses it --
// so there is no separate Serialize stage (unlike BlockDelta, the upstream
// caller already serializes Huffman before handing us the byte stream).
//
// Container layout written by Compress:
//   signature(8) | uncompressed_total:size_t | chunk_size:size_t |
//   nchunks:size_t | comp_bytes[nchunks]:size_t | packed_bytes:size_t |
//   packed[]
// comp_bytes[] is the per-chunk compressed length; Decompress exclusive-scans
// it to recover each chunk's offset into packed[] (same trick as BlockDelta).
template <typename DeviceType> class LZ4 {
public:
  LZ4() : initialized(false) {}

  // The warp-per-chunk CUDA compress kernel (LZ4Fused) replaces the portable
  // 1-thread-per-chunk functor when available; it keeps its hash table in
  // shared memory, so the global htable_array is not needed on that path. True
  // for the GPU backend (CUDA/HIP) that has the fused warp-per-chunk path
  // compiled in this TU. SERIAL/OpenMP/other always use the portable functor.
  static constexpr bool fused_backend() {
    bool r = false;
#if defined(MGARDX_COMPILE_CUDA)
    r = r || std::is_same<DeviceType, CUDA>::value;
#endif
#if defined(MGARDX_COMPILE_HIP)
    r = r || std::is_same<DeviceType, HIP>::value;
#endif
#if defined(MGARDX_COMPILE_SYCL)
    r = r || std::is_same<DeviceType, SYCL>::value;
#endif
    return r;
  }

  bool uses_fused() const {
#if defined(MGARDX_COMPILE_CUDA) || defined(MGARDX_COMPILE_HIP) ||             \
    defined(MGARDX_COMPILE_SYCL)
    return fused_backend() && lz4_fused::fused_ok(chunk_size);
#else
    return false;
#endif
  }

  LZ4(SIZE n, SIZE chunk_size) {
    Resize(n, chunk_size, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  void Resize(SIZE n, SIZE chunk_size, int queue_idx) {
    this->initialized = true;
    this->max_size = n;
    this->chunk_size = chunk_size;
    SIZE max_nchunks = (n - 1) / chunk_size + 1;
    this->max_chunk_out = (SIZE)lz4::compress_bound(chunk_size);

    input_data = Array<1, Byte, DeviceType>({n});
    // Global per-chunk hash table only feeds the portable functor path; the
    // fused CUDA kernel uses shared memory instead, so skip the (large) alloc.
    if (!uses_fused())
      htable_array = Array<1, uint16_t, DeviceType>(
          {(SIZE)((size_t)max_nchunks * lz4::HASH_SIZE)});
    scratch_array = Array<1, Byte, DeviceType>(
        {(SIZE)((size_t)max_nchunks * max_chunk_out)});
    comp_bytes_array = Array<1, size_t, DeviceType>({max_nchunks});
    byte_offset_array = Array<1, size_t, DeviceType>({max_nchunks + 1});
    DeviceCollective<DeviceType>::ScanSumExtended(
        max_nchunks, SubArray<1, size_t, DeviceType>(),
        SubArray<1, size_t, DeviceType>(), scan_workspace, false, queue_idx);
  }

  static size_t EstimateMemoryFootprint(SIZE n, SIZE chunk_size) {
    SIZE max_nchunks = (n - 1) / chunk_size + 1;
    size_t max_chunk_out = lz4::compress_bound(chunk_size);
    size_t size = n;                                                 // input
    size += (size_t)max_nchunks * lz4::HASH_SIZE * sizeof(uint16_t); // htable
    size += (size_t)max_nchunks * max_chunk_out;                     // scratch
    size += (size_t)max_nchunks * sizeof(size_t);       // comp_bytes
    size += (size_t)(max_nchunks + 1) * sizeof(size_t); // offsets
    return size;
  }

  void Compress(Array<1, Byte, DeviceType> &data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    // Snapshot the input; `data` is reused as the compressed output.
    SIZE input_count = data.shape(0);
    n = input_count;
    nchunks = (n - 1) / chunk_size + 1;
    input_data.resize({n}, queue_idx);
    MemoryManager<DeviceType>::Copy1D(input_data.data(), data.data(), n,
                                      queue_idx);

    SubArray<1, Byte, DeviceType> input_subarray({(SIZE)n}, input_data.data());
    SubArray<1, Byte, DeviceType> scratch_subarray(
        {(SIZE)((size_t)nchunks * max_chunk_out)}, scratch_array.data());
    SubArray<1, size_t, DeviceType> comp_bytes_subarray(
        {(SIZE)nchunks}, comp_bytes_array.data());
    SubArray<1, size_t, DeviceType> byte_offset_subarray(
        {(SIZE)nchunks + 1}, byte_offset_array.data());

    // 1) compress each chunk into its scratch slot; record per-chunk size.
    //    CUDA: warp-per-chunk fused kernel (shared-memory hash, parallel match
    //    scan). Other backends / oversized chunks: portable 1-thread functor.
    bool did_fused = false;
#if defined(MGARDX_COMPILE_CUDA) || defined(MGARDX_COMPILE_HIP) ||             \
    defined(MGARDX_COMPILE_SYCL)
    if constexpr (fused_backend()) {
      if (lz4_fused::fused_ok(chunk_size)) {
        auto stream = DeviceRuntime<DeviceType>::GetQueue(queue_idx);
        lz4_fused::launch_compress(input_subarray.data(), (SIZE)n,
                                   (int)chunk_size, (SIZE)nchunks,
                                   (SIZE)max_chunk_out, scratch_subarray.data(),
                                   comp_bytes_subarray.data(), stream);
        did_fused = true;
      }
    }
#endif
    if (!did_fused) {
      SubArray<1, uint16_t, DeviceType> htable_subarray(
          {(SIZE)((size_t)nchunks * lz4::HASH_SIZE)}, htable_array.data());
      DeviceLauncher<DeviceType>::Execute(
          LZ4ChunkCompressKernel<DeviceType>(
              input_subarray, (SIZE)n, (SIZE)chunk_size, (SIZE)nchunks,
              (SIZE)max_chunk_out, htable_subarray, scratch_subarray,
              comp_bytes_subarray),
          queue_idx);
    }

    // 2) exclusive scan of per-chunk sizes -> offsets into packed[] (+ total).
    DeviceCollective<DeviceType>::ScanSumExtended(
        (SIZE)nchunks, comp_bytes_subarray, byte_offset_subarray,
        scan_workspace, true, queue_idx);
    MemoryManager<DeviceType>::Copy1D(
        &packed_bytes, byte_offset_subarray.data() + nchunks, 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    // 3) size the output and 4) gather chunks into the packed region.
    SIZE packed_byte_offset;
    SIZE compressed_size =
        ComputeLayout(nchunks, packed_bytes, packed_byte_offset);
    data.resize({compressed_size}, queue_idx);
    SubArray<1, Byte, DeviceType> out_subarray(data);
    SubArray<1, Byte, DeviceType> packed_subarray(
        {(SIZE)packed_bytes}, (Byte *)out_subarray(packed_byte_offset));
    DeviceLauncher<DeviceType>::Execute(
        LZ4CondenseKernel<DeviceType>(scratch_subarray, (SIZE)nchunks,
                                      (SIZE)max_chunk_out, comp_bytes_subarray,
                                      byte_offset_subarray, packed_subarray),
        queue_idx);

    // Header (everything except packed[], which the condense kernel wrote).
    size_t n_v = n, chunk_v = chunk_size, nchunks_v = nchunks,
           packed_v = packed_bytes;
    SIZE byte_offset = 0;
    SerializeArray<Byte>(out_subarray, signature, kSignatureLen, byte_offset,
                         queue_idx);
    SerializeArray<size_t>(out_subarray, &n_v, 1, byte_offset, queue_idx);
    SerializeArray<size_t>(out_subarray, &chunk_v, 1, byte_offset, queue_idx);
    SerializeArray<size_t>(out_subarray, &nchunks_v, 1, byte_offset, queue_idx);
    SerializeArray<size_t>(out_subarray, comp_bytes_array.data(), nchunks,
                           byte_offset, queue_idx);
    SerializeArray<size_t>(out_subarray, &packed_v, 1, byte_offset, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    log::info("LZ4 compress ratio: " + std::to_string(input_count) + "/" +
              std::to_string(compressed_size) + " (" +
              std::to_string((double)input_count / compressed_size) + ")");
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("LZ4 compress", input_count);
      timer.clear();
    }
  }

  void Decompress(Array<1, Byte, DeviceType> &data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    // Snapshot the compressed input; `data` is reused as the output.
    input_data.resize({data.shape(0)}, queue_idx);
    MemoryManager<DeviceType>::Copy1D(input_data.data(), data.data(),
                                      data.shape(0), queue_idx);
    SubArray<1, Byte, DeviceType> in_subarray(input_data);

    // Parse header. comp_bytes_ptr / packed_ptr are zero-copy into in_subarray.
    SIZE byte_offset = 0;
    Byte *sig = signature_verify;
    DeserializeArray<Byte>(in_subarray, sig, kSignatureLen, byte_offset, false,
                           queue_idx);
    size_t *n_ptr = &n, *chunk_ptr = &chunk_size_runtime,
           *nchunks_ptr = &nchunks, *packed_ptr_sz = &packed_bytes;
    DeserializeArray<size_t>(in_subarray, n_ptr, 1, byte_offset, false,
                             queue_idx);
    DeserializeArray<size_t>(in_subarray, chunk_ptr, 1, byte_offset, false,
                             queue_idx);
    DeserializeArray<size_t>(in_subarray, nchunks_ptr, 1, byte_offset, false,
                             queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    for (int i = 0; i < kSignatureLen; i++)
      if (signature_verify[i] != signature[i])
        throw std::runtime_error("LZ4 signature mismatch.");

    size_t *comp_bytes_ptr = nullptr;
    DeserializeArray<size_t>(in_subarray, comp_bytes_ptr, nchunks, byte_offset,
                             true, queue_idx); // zero-copy device pointer
    DeserializeArray<size_t>(in_subarray, packed_ptr_sz, 1, byte_offset, false,
                             queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    Byte *packed_ptr = nullptr;
    DeserializeArray<Byte>(in_subarray, packed_ptr, packed_bytes, byte_offset,
                           true, queue_idx);

    SubArray<1, size_t, DeviceType> comp_bytes_subarray({(SIZE)nchunks},
                                                        comp_bytes_ptr);
    SubArray<1, size_t, DeviceType> byte_offset_subarray(
        {(SIZE)nchunks + 1}, byte_offset_array.data());
    SubArray<1, Byte, DeviceType> packed_subarray({(SIZE)packed_bytes},
                                                  packed_ptr);

    // Rebuild per-chunk offsets into packed[] from the stored sizes.
    DeviceCollective<DeviceType>::ScanSumExtended(
        (SIZE)nchunks, comp_bytes_subarray, byte_offset_subarray,
        scan_workspace, true, queue_idx);

    data.resize({(SIZE)n}, queue_idx);
    SubArray<1, Byte, DeviceType> out_subarray(data);
    // CUDA: warp-per-chunk fused decoder (cooperative literal/match copy).
    // Other backends / oversized chunks: portable 1-thread decode functor.
    bool did_fused = false;
#if defined(MGARDX_COMPILE_CUDA) || defined(MGARDX_COMPILE_HIP) ||             \
    defined(MGARDX_COMPILE_SYCL)
    if constexpr (fused_backend()) {
      if (lz4_fused::fused_ok(chunk_size_runtime)) {
        auto stream = DeviceRuntime<DeviceType>::GetQueue(queue_idx);
        lz4_fused::launch_decompress(packed_subarray.data(),
                                     byte_offset_subarray.data(), (SIZE)n,
                                     (int)chunk_size_runtime, (SIZE)nchunks,
                                     out_subarray.data(), stream);
        did_fused = true;
      }
    }
#endif
    if (!did_fused) {
      DeviceLauncher<DeviceType>::Execute(
          LZ4ChunkDecompressKernel<DeviceType>(
              packed_subarray, byte_offset_subarray, (SIZE)n,
              (SIZE)chunk_size_runtime, (SIZE)nchunks, out_subarray),
          queue_idx);
    }
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("LZ4 decompress", n);
      timer.clear();
    }
  }

private:
  // Layout walker; mirrors the header order written by Compress. Returns the
  // total compressed size and, via the out-param, where packed[] begins.
  SIZE ComputeLayout(size_t nchunks, size_t packed_bytes,
                     SIZE &packed_byte_offset) {
    SIZE off = 0;
    advance_with_align<Byte>(off, kSignatureLen); // signature
    advance_with_align<size_t>(off, 1);           // n
    advance_with_align<size_t>(off, 1);           // chunk_size
    advance_with_align<size_t>(off, 1);           // nchunks
    advance_with_align<size_t>(off, nchunks);     // comp_bytes[]
    advance_with_align<size_t>(off, 1);           // packed_bytes
    align_byte_offset<Byte>(off);
    packed_byte_offset = off;
    advance_with_align<Byte>(off, packed_bytes); // packed[]
    return off;
  }

public:
  bool initialized;
  SIZE max_size;
  SIZE chunk_size;
  SIZE max_chunk_out;
  size_t n = 0;
  size_t nchunks = 0;
  size_t packed_bytes = 0;
  size_t chunk_size_runtime = 0; // chunk_size read back on Decompress

private:
  static constexpr int kSignatureLen = 8;
  Byte signature[8] = {'M', 'G', 'X', 'L', 'Z', '4', 'P', '\0'};
  Byte signature_verify[8] = {0};

  Array<1, Byte, DeviceType> input_data;
  Array<1, uint16_t, DeviceType> htable_array;
  Array<1, Byte, DeviceType> scratch_array;
  Array<1, size_t, DeviceType> comp_bytes_array;
  Array<1, size_t, DeviceType> byte_offset_array;
  Array<1, Byte, DeviceType> scan_workspace;
};

} // namespace mgard_x_160

#endif
