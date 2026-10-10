/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 */

#ifndef MGARD_X_160_BLOCK_DELTA_HPP
#define MGARD_X_160_BLOCK_DELTA_HPP

#include "../../RuntimeX/Utilities/Serializer.hpp"
#include "../../Utilities/Types.h" // block_delta_mode_type
#include "../LosslessCompressorInterface.hpp"
#include "BlockDeltaFused.hpp"   // CUDA/HIP single-kernel decoupled look-back
#include "BlockDeltaKernels.hpp" // pulls in RuntimeX (types + macros) first

namespace mgard_x_160 {

// True for the GPU backends that get the fused single-kernel (decoupled
// look-back) implementation. Everything else uses the portable multi-kernel
// path. The branch is resolved at compile time, so only one path is
// instantiated per backend.
template <typename DeviceType> struct is_gpu_device {
  static constexpr bool value = std::is_same<DeviceType, CUDA>::value ||
                                std::is_same<DeviceType, HIP>::value;
};

// BlockDelta lossless backend: a non-entropy alternative to Huffman that
// operates on the same signed quantized-integer stream. See BlockDeltaKernels
// for the encoding scheme. Self-contained: Compress writes a complete buffer
// (header + per-block bit-widths + packed stream); Serialize is therefore a
// no-op and Deserialize parses the header back.
template <typename T, typename DeviceType> class BlockDeltaLossless {
public:
  BlockDeltaLossless() : initialized(false) {}

  BlockDeltaLossless(
      SIZE max_size, int block_size,
      block_delta_mode_type mode = block_delta_mode_type::Delta) {
    Resize(max_size, block_size, mode, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);
  }

  void Resize(SIZE max_size, int block_size, block_delta_mode_type mode,
              int queue_idx) {
    this->initialized = true;
    this->max_size = max_size;
    this->block_size = block_size;
    this->mode = mode;
    SIZE max_nblocks = (max_size - 1) / block_size + 1;
    bitwidth_array = Array<1, Byte, DeviceType>({max_nblocks});
    bytecount_array = Array<1, size_t, DeviceType>({max_nblocks});
    byte_offset_array = Array<1, size_t, DeviceType>({max_nblocks + 1});
    // Per-block outlier counts (Outlier mode only; tiny, always allocated).
    oc_array = Array<1, uint16_t, DeviceType>({max_nblocks});
    // Scan workspace for the multi-kernel path (extended exclusive scan over
    // nblocks). Always sized: the GPU backend still uses the portable path for
    // Outlier mode, which scans.
    DeviceCollective<DeviceType>::ScanSumExtended(
        max_nblocks, SubArray<1, size_t, DeviceType>(),
        SubArray<1, size_t, DeviceType>(), scan_workspace, false, queue_idx);
    if constexpr (is_gpu_device<DeviceType>::value) {
      // Decoupled-look-back state for the fused path: one status word per block
      // plus a global tile counter and a total-size slot.
      status_array = Array<1, unsigned long long, DeviceType>({max_nblocks});
      counter_array = Array<1, unsigned int, DeviceType>({1});
      total_array = Array<1, unsigned long long, DeviceType>({1});
    }
  }

  static size_t EstimateMemoryFootprint(SIZE primary_count, int block_size) {
    SIZE max_nblocks = (primary_count - 1) / block_size + 1;
    size_t size = max_nblocks * sizeof(Byte);   // bitwidth
    size += max_nblocks * sizeof(size_t);       // bytecount
    size += (max_nblocks + 1) * sizeof(size_t); // byte_offset
    size += max_nblocks * sizeof(uint16_t);     // outlier counts
    return size;
  }

  // ---- public LosslessCompressorInterface-style entry points --------------

  void Compress(Array<1, T, DeviceType> &original_data,
                Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    if constexpr (is_gpu_device<DeviceType>::value) {
      CompressFused(original_data, compressed_data, queue_idx);
    } else {
      CompressPortable(original_data, compressed_data, queue_idx);
    }
  }

  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  Array<1, T, DeviceType> &decompressed_data, int queue_idx) {
    if constexpr (is_gpu_device<DeviceType>::value) {
      DecompressFused(compressed_data, decompressed_data, queue_idx);
    } else {
      DecompressPortable(compressed_data, decompressed_data, queue_idx);
    }
  }

  // Memory-movement stage (no computation): copy the bit-width array out of the
  // workspace and write the scalar header into the compressed buffer. Compress
  // already wrote the packed bitstream in place; this only fills the metadata
  // sections around it. Kept separate from Compress so it can overlap with the
  // next subdomain's kernels in the pipeline (same split as Huffman).
  void Serialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    SubArray<1, Byte, DeviceType> cd(compressed_data);
    size_t n_v = n, nblocks_v = nblocks, bitwidth_bytes_v = nblocks,
           oc_bytes_v = nblocks * sizeof(uint16_t),
           packed_bytes_v = packed_bytes;
    int block_size_v = block_size;
    Byte mode_v = (Byte)mode;
    SIZE byte_offset = 0;
    SerializeArray<Byte>(cd, signature, kSignatureLen, byte_offset, queue_idx);
    SerializeArray<size_t>(cd, &n_v, 1, byte_offset, queue_idx);
    SerializeArray<int>(cd, &block_size_v, 1, byte_offset, queue_idx);
    SerializeArray<Byte>(cd, &mode_v, 1, byte_offset, queue_idx);
    SerializeArray<size_t>(cd, &nblocks_v, 1, byte_offset, queue_idx);
    SerializeArray<size_t>(cd, &bitwidth_bytes_v, 1, byte_offset, queue_idx);
    SerializeArray<Byte>(cd, bitwidth_array.data(), nblocks, byte_offset,
                         queue_idx);
    if (mode == block_delta_mode_type::Outlier) {
      SerializeArray<size_t>(cd, &oc_bytes_v, 1, byte_offset, queue_idx);
      SerializeArray<uint16_t>(cd, oc_array.data(), nblocks, byte_offset,
                               queue_idx);
    }
    SerializeArray<size_t>(cd, &packed_bytes_v, 1, byte_offset, queue_idx);
    // packed[] is already in place (written by Compress); nothing to copy.
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("BlockDelta serialize", compressed_data.shape(0));
      timer.clear();
    }
  }

  void Deserialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    ParseHeader(compressed_data, queue_idx);
  }

private:
  // Layout walker; mirrors the section order written by Serialize. Returns the
  // total compressed size and, via the out-params, where the bit-width array,
  // the outlier-count array (Outlier mode only), and the packed bitstream land.
  // Layout: signature | n | block_size | mode | nblocks | bitwidth_bytes |
  //         bitwidth[] | [Outlier: oc_bytes | oc[]] | packed_bytes | packed[]
  SIZE ComputeLayout(Byte mode_v, size_t nblocks, size_t packed_bytes,
                     SIZE &bitwidth_byte_offset, SIZE &oc_byte_offset,
                     SIZE &packed_byte_offset) {
    SIZE off = 0;
    advance_with_align<Byte>(off, kSignatureLen); // signature
    advance_with_align<size_t>(off, 1);           // n
    advance_with_align<int>(off, 1);              // block_size
    advance_with_align<Byte>(off, 1);             // mode
    advance_with_align<size_t>(off, 1);           // nblocks
    advance_with_align<size_t>(off, 1);           // bitwidth_bytes
    align_byte_offset<Byte>(off);
    bitwidth_byte_offset = off;
    advance_with_align<Byte>(off, nblocks); // bitwidth[]
    oc_byte_offset = 0;
    if (mode_v == (Byte)block_delta_mode_type::Outlier) {
      advance_with_align<size_t>(off, 1); // oc_bytes
      align_byte_offset<uint16_t>(off);
      oc_byte_offset = off;
      advance_with_align<uint16_t>(off, nblocks); // oc[]
    }
    advance_with_align<size_t>(off, 1); // packed_bytes
    align_byte_offset<Byte>(off);
    packed_byte_offset = off;
    advance_with_align<Byte>(off, packed_bytes);
    return off;
  }

  void CompressPortable(Array<1, T, DeviceType> &original_data,
                        Array<1, Byte, DeviceType> &compressed_data,
                        int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    n = original_data.shape(0);
    nblocks = (SIZE)((n - 1) / block_size + 1);
    SubArray<1, T, DeviceType> data_subarray(original_data);
    SubArray<1, Byte, DeviceType> bitwidth_subarray({(SIZE)nblocks},
                                                    bitwidth_array.data());
    SubArray<1, size_t, DeviceType> bytecount_subarray({(SIZE)nblocks},
                                                       bytecount_array.data());
    SubArray<1, size_t, DeviceType> byte_offset_subarray(
        {(SIZE)nblocks + 1}, byte_offset_array.data());
    SubArray<1, uint16_t, DeviceType> oc_subarray({(SIZE)nblocks},
                                                  oc_array.data());
    Byte mode_v = (Byte)mode;

    // 1) per-block bit-width + byte-count (+ outlier count)
    DeviceLauncher<DeviceType>::Execute(
        BlockBitwidthKernel<T, DeviceType>(
            data_subarray, (SIZE)n, (SIZE)block_size, (SIZE)nblocks, mode_v,
            bitwidth_subarray, bytecount_subarray, oc_subarray),
        queue_idx);

    // 2) exclusive scan of byte-counts -> per-block byte offsets (+ total)
    DeviceCollective<DeviceType>::ScanSumExtended(
        (SIZE)nblocks, bytecount_subarray, byte_offset_subarray, scan_workspace,
        true, queue_idx);
    MemoryManager<DeviceType>::Copy1D(
        &packed_bytes, byte_offset_subarray.data() + nblocks, 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    // 3) size the output buffer. Computation only: the metadata (signature,
    // scalar header, bit-width array, outlier counts) is *not* written here --
    // that memory movement is deferred to Serialize so it can overlap with the
    // next subdomain's computation in the pipeline.
    SIZE bitwidth_byte_offset, oc_byte_offset, packed_byte_offset;
    SIZE compressed_size =
        ComputeLayout(mode_v, nblocks, packed_bytes, bitwidth_byte_offset,
                      oc_byte_offset, packed_byte_offset);
    compressed_data.resize({compressed_size}, queue_idx);
    SubArray<1, Byte, DeviceType> compressed_subarray(compressed_data);

    // 4) pack into the packed region (a kernel writing to its final location,
    // mirroring how Huffman's deflate writes its bitstream in place).
    SubArray<1, Byte, DeviceType> packed_subarray(
        {(SIZE)packed_bytes}, (Byte *)compressed_subarray(packed_byte_offset));
    DeviceLauncher<DeviceType>::Execute(
        BlockPackKernel<T, DeviceType>(data_subarray, (SIZE)n, (SIZE)block_size,
                                       (SIZE)nblocks, mode_v, bitwidth_subarray,
                                       byte_offset_subarray, packed_subarray),
        queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    log::info("BlockDelta compress ratio: " + std::to_string(n * sizeof(T)) +
              "/" + std::to_string(compressed_size) + " (" +
              std::to_string((double)n * sizeof(T) / compressed_size) + ")");
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("BlockDelta compress", n * sizeof(T));
      timer.clear();
    }
  }

  void ParseHeader(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    SubArray<1, Byte, DeviceType> compressed_subarray(compressed_data);
    SIZE byte_offset = 0;

    Byte *sig = signature_verify;
    DeserializeArray<Byte>(compressed_subarray, sig, kSignatureLen, byte_offset,
                           false, queue_idx);
    size_t *n_ptr = &n, *nblocks_ptr = &nblocks,
           *bitwidth_bytes_ptr = &bitwidth_bytes,
           *packed_bytes_ptr = &packed_bytes;
    int *block_size_ptr = &block_size;
    Byte mode_v = 0, *mode_ptr = &mode_v;
    DeserializeArray<size_t>(compressed_subarray, n_ptr, 1, byte_offset, false,
                             queue_idx);
    DeserializeArray<int>(compressed_subarray, block_size_ptr, 1, byte_offset,
                          false, queue_idx);
    DeserializeArray<Byte>(compressed_subarray, mode_ptr, 1, byte_offset, false,
                           queue_idx);
    DeserializeArray<size_t>(compressed_subarray, nblocks_ptr, 1, byte_offset,
                             false, queue_idx);
    DeserializeArray<size_t>(compressed_subarray, bitwidth_bytes_ptr, 1,
                             byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    this->mode = (block_delta_mode_type)mode_v;
    for (int i = 0; i < kSignatureLen; i++) {
      if (signature_verify[i] != signature[i]) {
        throw std::runtime_error("BlockDelta signature mismatch.");
      }
    }
    // zero-copy device pointers into the compressed buffer
    DeserializeArray<Byte>(compressed_subarray, bitwidth_ptr, bitwidth_bytes,
                           byte_offset, true, queue_idx);
    if (this->mode == block_delta_mode_type::Outlier) {
      size_t oc_bytes = 0, *oc_bytes_ptr = &oc_bytes;
      DeserializeArray<size_t>(compressed_subarray, oc_bytes_ptr, 1,
                               byte_offset, false, queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      DeserializeArray<uint16_t>(compressed_subarray, oc_ptr,
                                 oc_bytes / sizeof(uint16_t), byte_offset, true,
                                 queue_idx);
    }
    DeserializeArray<size_t>(compressed_subarray, packed_bytes_ptr, 1,
                             byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    DeserializeArray<Byte>(compressed_subarray, packed_ptr, packed_bytes,
                           byte_offset, true, queue_idx);
  }

  void DecompressPortable(Array<1, Byte, DeviceType> &compressed_data,
                          Array<1, T, DeviceType> &decompressed_data,
                          int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    decompressed_data.resize({(SIZE)n}, queue_idx);
    SubArray<1, T, DeviceType> data_subarray(decompressed_data);
    SubArray<1, Byte, DeviceType> bitwidth_subarray({(SIZE)nblocks},
                                                    bitwidth_ptr);
    SubArray<1, Byte, DeviceType> packed_subarray({(SIZE)packed_bytes},
                                                  packed_ptr);
    SubArray<1, size_t, DeviceType> bytecount_subarray({(SIZE)nblocks},
                                                       bytecount_array.data());
    SubArray<1, size_t, DeviceType> byte_offset_subarray(
        {(SIZE)nblocks + 1}, byte_offset_array.data());
    // Outlier mode reads its per-block counts from the (zero-copy) stream
    // pointer; other modes don't touch it.
    SubArray<1, uint16_t, DeviceType> oc_subarray(
        {(SIZE)nblocks},
        mode == block_delta_mode_type::Outlier ? oc_ptr : oc_array.data());
    Byte mode_v = (Byte)mode;

    // Rebuild per-block byte offsets from the stored bit-widths (+ counts).
    DeviceLauncher<DeviceType>::Execute(
        BlockBytecountKernel<T, DeviceType>(
            (SIZE)n, (SIZE)block_size, (SIZE)nblocks, mode_v, bitwidth_subarray,
            oc_subarray, bytecount_subarray),
        queue_idx);
    DeviceCollective<DeviceType>::ScanSumExtended(
        (SIZE)nblocks, bytecount_subarray, byte_offset_subarray, scan_workspace,
        true, queue_idx);

    DeviceLauncher<DeviceType>::Execute(
        BlockUnpackKernel<T, DeviceType>(
            packed_subarray, (SIZE)n, (SIZE)block_size, (SIZE)nblocks, mode_v,
            bitwidth_subarray, byte_offset_subarray, data_subarray),
        queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("BlockDelta decompress", n * sizeof(T));
      timer.clear();
    }
  }

  // -------- Fused (CUDA/HIP) path -----------------------------------------
  // Single kernel: the cross-block byte-offset scan is resolved in-kernel via
  // decoupled look-back (see BlockDeltaFused.hpp). The byte layout is identical
  // to the portable path, so streams are interchangeable across backends.
  void CompressFused(Array<1, T, DeviceType> &original_data,
                     Array<1, Byte, DeviceType> &compressed_data,
                     int queue_idx) {
#if defined(MGARDX_COMPILE_CUDA) || defined(MGARDX_COMPILE_HIP)
    // Outlier mode has variable per-block side records that don't fit the
    // single-pass look-back cleanly; use the portable multi-kernel path (which
    // also runs on the GPU). Fixed and Delta use the fused single kernel.
    if (mode == block_delta_mode_type::Outlier) {
      CompressPortable(original_data, compressed_data, queue_idx);
      return;
    }
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    n = original_data.shape(0);
    nblocks = (SIZE)((n - 1) / block_size + 1);

    // Worst case: a block keeps full width -> packed <= n * sizeof(T). The
    // bit-width and packed offsets depend only on nblocks, so they are valid
    // for the real (trimmed) layout too.
    size_t worst_packed = (size_t)n * sizeof(T);
    SIZE bitwidth_off, oc_off, packed_off;
    SIZE worst_size = ComputeLayout((Byte)mode, nblocks, worst_packed,
                                    bitwidth_off, oc_off, packed_off);
    compressed_data.resize({worst_size}, queue_idx);
    Byte *base = SubArray<1, Byte, DeviceType>(compressed_data).data();

    // Reset decoupled-look-back state.
    MemoryManager<DeviceType>::Memset1D(status_array.data(), (SIZE)nblocks, 0,
                                        queue_idx);
    MemoryManager<DeviceType>::Memset1D(counter_array.data(), 1, 0, queue_idx);

    // Computation only: the kernel writes the bit-widths into the workspace and
    // the packed bitstream into its final location. The metadata memory
    // movement (bit-widths + scalar header) is deferred to Serialize.
    bool use_delta = (mode != block_delta_mode_type::Fixed);
    auto stream = DeviceRuntime<DeviceType>::GetQueue(queue_idx);
    block_delta_fused::launch_encode<T>(
        original_data.data(), (SIZE)n, (SIZE)block_size, (SIZE)nblocks,
        use_delta, bitwidth_array.data(), base + packed_off,
        status_array.data(), counter_array.data(), total_array.data(), stream);

    unsigned long long total = 0;
    MemoryManager<DeviceType>::Copy1D(&total, total_array.data(), 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    packed_bytes = (size_t)total;

    // Trim to the real size (in-place shrink preserves the kernel's packed
    // writes). bitwidth_off is unused here -- Serialize recomputes the layout.
    (void)bitwidth_off;
    SIZE compressed_size = packed_off + (SIZE)packed_bytes;
    compressed_data.resize({compressed_size}, queue_idx);

    log::info(
        "BlockDelta(fused) compress ratio: " + std::to_string(n * sizeof(T)) +
        "/" + std::to_string(compressed_size) + " (" +
        std::to_string((double)n * sizeof(T) / compressed_size) + ")");
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("BlockDelta(fused) compress", n * sizeof(T));
      timer.clear();
    }
#else
    CompressPortable(original_data, compressed_data, queue_idx);
#endif
  }

  void DecompressFused(Array<1, Byte, DeviceType> &compressed_data,
                       Array<1, T, DeviceType> &decompressed_data,
                       int queue_idx) {
#if defined(MGARDX_COMPILE_CUDA) || defined(MGARDX_COMPILE_HIP)
    // ParseHeader (via Deserialize) has set n, nblocks, block_size, mode,
    // bitwidth_ptr and packed_ptr. Outlier mode uses the portable path.
    if (mode == block_delta_mode_type::Outlier) {
      DecompressPortable(compressed_data, decompressed_data, queue_idx);
      return;
    }
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    decompressed_data.resize({(SIZE)n}, queue_idx);
    MemoryManager<DeviceType>::Memset1D(status_array.data(), (SIZE)nblocks, 0,
                                        queue_idx);
    MemoryManager<DeviceType>::Memset1D(counter_array.data(), 1, 0, queue_idx);

    bool use_delta = (mode != block_delta_mode_type::Fixed);
    auto stream = DeviceRuntime<DeviceType>::GetQueue(queue_idx);
    block_delta_fused::launch_decode<T>(
        packed_ptr, (SIZE)n, (SIZE)block_size, (SIZE)nblocks, use_delta,
        bitwidth_ptr, decompressed_data.data(), status_array.data(),
        counter_array.data(), stream);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("BlockDelta(fused) decompress", n * sizeof(T));
      timer.clear();
    }
#else
    DecompressPortable(compressed_data, decompressed_data, queue_idx);
#endif
  }

public:
  bool initialized;
  SIZE max_size;
  int block_size;
  block_delta_mode_type mode;
  size_t n;
  size_t nblocks;
  size_t bitwidth_bytes;
  size_t packed_bytes;

private:
  static constexpr int kSignatureLen = 8;
  Byte signature[8] = {'M', 'G', 'X', 'B', 'L', 'K', 'D', '\0'};
  Byte signature_verify[8] = {0};
  // zero-copy pointers into the compressed buffer, set by ParseHeader
  Byte *bitwidth_ptr = nullptr;
  Byte *packed_ptr = nullptr;
  uint16_t *oc_ptr = nullptr;

  Array<1, Byte, DeviceType> bitwidth_array;
  Array<1, size_t, DeviceType> bytecount_array;
  Array<1, size_t, DeviceType> byte_offset_array;
  Array<1, uint16_t, DeviceType> oc_array;
  Array<1, Byte, DeviceType> scan_workspace;
  // Fused (GPU) decoupled-look-back state.
  Array<1, unsigned long long, DeviceType> status_array;
  Array<1, unsigned int, DeviceType> counter_array;
  Array<1, unsigned long long, DeviceType> total_array;
};

} // namespace mgard_x_160

#endif
