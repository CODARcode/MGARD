/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (jieyang@uoregon.edu)
 * Date: September 21, 2026
 */

#ifndef MGARD_X_HUFFMAN_TEMPLATE_HPP
#define MGARD_X_HUFFMAN_TEMPLATE_HPP

static bool debug_print_huffman = false;

#include "../LosslessCompressorInterface.hpp"

#include "../../RuntimeX/Utilities/Serializer.hpp"
#include "Condense.hpp"
#include "Decode.hpp"
#include "Deflate.hpp"
#include "DictionaryShift.hpp"
#include "EncodeFixedLen.hpp"
#include "GetCodebook.hpp"
#include "Histogram.hpp"
#include "HuffmanWorkspace.hpp"
#include "OutlierSeparator.hpp"
#include "ParallelDeflate.hpp"

#include <chrono>
using namespace std::chrono;

namespace mgard_x {

template <typename Q, typename S, typename H, typename DeviceType>
class Huffman {
public:
  Huffman() : initialized(false) {}

  Huffman(SIZE max_size, int dict_size, int chunk_size,
          double estimated_outlier_ratio)
      : initialized(true), max_size(max_size), dict_size(dict_size),
        chunk_size(chunk_size) {
    workspace = HuffmanWorkspace<Q, S, H, DeviceType>(
        max_size, dict_size, chunk_size, estimated_outlier_ratio);
  }

  void Resize(SIZE max_size, int dict_size, int chunk_size,
              double estimated_outlier_ratio, int queue_idx) {
    this->initialized = true;
    this->max_size = max_size;
    this->dict_size = dict_size;
    this->chunk_size = chunk_size;
    workspace.resize(max_size, dict_size, chunk_size, estimated_outlier_ratio,
                     queue_idx);
  }

  static size_t EstimateMemoryFootprint(SIZE primary_count, SIZE dict_size,
                                        SIZE chunk_size,
                                        double estimated_outlier_ratio = 1) {
    return HuffmanWorkspace<Q, S, H, DeviceType>::EstimateMemoryFootprint(
        primary_count, dict_size, chunk_size, estimated_outlier_ratio);
  }

  double EstimateCR(Array<1, Q, DeviceType> &primary_data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    SubArray primary_subarray(primary_data);
    workspace.reset(queue_idx);

    primary_count = primary_subarray.shape(0);

    Histogram<Q, unsigned int, DeviceType>(primary_subarray,
                                           workspace.freq_subarray,
                                           primary_count, dict_size, queue_idx);
    auto type_bw = sizeof(H) * 8;

    SubArray<1, H, DeviceType> _d_first_subarray(
        {(SIZE)type_bw}, (H *)workspace.decodebook_subarray((IDX)0));
    SubArray<1, H, DeviceType> _d_entry_subarray(
        {(SIZE)type_bw},
        (H *)workspace.decodebook_subarray(sizeof(H) * type_bw));
    SubArray<1, Q, DeviceType> _d_qcode_subarray(
        {(SIZE)dict_size},
        (Q *)workspace.decodebook_subarray(sizeof(H) * 2 * type_bw));

    // Sort Qcodes by frequency
    DeviceLauncher<DeviceType>::Execute(
        FillArraySequenceKernel(_d_qcode_subarray), queue_idx);

    MemoryManager<DeviceType>::Copy1D(workspace._d_freq_copy_subarray.data(),
                                      workspace.freq_subarray.data(), dict_size,
                                      queue_idx);
    MemoryManager<DeviceType>::Copy1D(workspace._d_qcode_copy_subarray.data(),
                                      _d_qcode_subarray.data(), dict_size,
                                      queue_idx);
    DeviceCollective<DeviceType>::SortByKey(
        (SIZE)dict_size, workspace._d_freq_copy_subarray,
        workspace._d_qcode_copy_subarray, workspace.freq_subarray,
        _d_qcode_subarray, workspace.sort_by_key_workspace, true, queue_idx);

    DeviceLauncher<DeviceType>::Execute(
        GetFirstNonzeroIndexKernel<unsigned int, DeviceType>(
            workspace.freq_subarray, workspace.first_nonzero_index_subarray),
        queue_idx);

    unsigned int first_nonzero_index;
    MemoryManager<DeviceType>().Copy1D(
        &first_nonzero_index, workspace.first_nonzero_index_subarray(IDX(0)), 1,
        queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    int nz_dict_size = dict_size - first_nonzero_index;

    SubArray<1, unsigned int, DeviceType> _nz_d_freq_subarray(
        {(SIZE)nz_dict_size}, workspace.freq_subarray(first_nonzero_index));
    SubArray<1, H, DeviceType> _nz_d_codebook_subarray(
        {(SIZE)nz_dict_size}, workspace.codebook_subarray(first_nonzero_index));

    DeviceLauncher<DeviceType>::Execute(
        GenerateCLKernel<unsigned int, DeviceType>(
            _nz_d_freq_subarray, workspace.CL_subarray, nz_dict_size,
            _nz_d_freq_subarray, workspace.lNodesLeader_subarray,
            workspace.iNodesFreq_subarray, workspace.iNodesLeader_subarray,
            workspace.tempFreq_subarray, workspace.tempIsLeaf_subarray,
            workspace.tempIndex_subarray, workspace.copyFreq_subarray,
            workspace.copyIsLeaf_subarray, workspace.copyIndex_subarray,
            workspace.diagonal_path_intersections_subarray,
            workspace.status_subarray),
        queue_idx);

    unsigned int *_freq = new unsigned int[dict_size];
    unsigned int *_cl = new unsigned int[dict_size];
    MemoryManager<DeviceType>::Copy1D(_freq, workspace.freq_subarray.data(),
                                      dict_size, queue_idx);
    MemoryManager<DeviceType>::Copy1D(_cl, workspace.CL_subarray.data(),
                                      dict_size, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    double LC = 0;
    for (SIZE i = 0; i < dict_size; i++) {
      LC += (double)_freq[i] * _cl[i];
    }
    delete[] _freq;
    delete[] _cl;

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman estimate CR", primary_count * sizeof(Q));
      timer.clear();
    }

    double CR = (double)(sizeof(Q) * primary_count) / (LC / 8 + 2000);
    return CR;
  }

  bool CompressPrimary(Array<1, Q, DeviceType> &primary_data,
                       Array<1, Byte, DeviceType> &compressed_data,
                       float target_cr, int queue_idx) {

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SubArray primary_subarray(primary_data);
    workspace.reset(queue_idx);

    primary_count = primary_subarray.shape(0);

    // Per-stage timing scaffolding (disabled; uncomment the block below and the
    // mark() calls to profile each stage under log::TIME). Each mark() syncs
    // the queue, prints the elapsed time of the stage just finished against the
    // primary input size, and restarts the stage timer. Note: this serializes
    // the pipeline (one sync per stage).
    // Timer timer_stage;
    // auto mark = [&](const char *name) {
    //   if (log::level & log::TIME) {
    //     DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    //     timer_stage.end();
    //     timer_stage.print(name, primary_count * sizeof(Q));
    //     timer_stage.clear();
    //     timer_stage.start();
    //   }
    // };
    // if (log::level & log::TIME) {
    //   DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    //   timer_stage.start();
    // }

    Histogram<Q, unsigned int, DeviceType>(primary_subarray,
                                           workspace.freq_subarray,
                                           primary_count, dict_size, queue_idx);
    // mark("Huffman stage: histogram");

    if (debug_print_huffman) {
      PrintSubarray("Histogram::freq_subarray", workspace.freq_subarray);
    }

    GetCodebook(dict_size, primary_count, workspace.freq_subarray,
                workspace.codebook_subarray, workspace.decodebook_subarray,
                workspace, queue_idx);
    // mark("Huffman stage: codebook");

    if (target_cr > 1.0) {
      // Encoded bits = sum over symbols of frequency x codeword length. After
      // GetCodebook, freq_subarray holds the frequencies sorted ascending and
      // CL_subarray has been reversed by GenerateCW, so they cannot be paired
      // by index. Use the unsorted histogram (_d_freq_copy_subarray) and the
      // symbol-indexed codebook, whose top byte is the codeword length (see
      // deflate_bitwidth).
      std::vector<unsigned int> freq(dict_size);
      std::vector<H> codebook(dict_size);
      MemoryManager<DeviceType>::Copy1D(freq.data(),
                                        workspace._d_freq_copy_subarray.data(),
                                        dict_size, queue_idx);
      MemoryManager<DeviceType>::Copy1D(codebook.data(),
                                        workspace.codebook_subarray.data(),
                                        dict_size, queue_idx);
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      double LC = 0;
      for (SIZE i = 0; i < dict_size; i++) {
        LC += (double)freq[i] * (double)(codebook[i] >> (sizeof(H) * 8 - 8));
      }
      double estimated_cr =
          (double)(sizeof(Q) * primary_count) / (LC / 8 + 2000);
      log::info("Huffman estimated CR: " + std::to_string(estimated_cr) +
                " (target: " + std::to_string(target_cr) + ")");
      if (estimated_cr < target_cr) {
        return false;
      }
    }

    if (debug_print_huffman) {
      PrintSubarray("GetCodebook::codebook_subarray",
                    workspace.codebook_subarray);
      PrintSubarray("GetCodebook::decodebook_subarray",
                    workspace.decodebook_subarray);
    }
    // Encoding is fused into the deflate kernels below: instead of first
    // materializing huff[i] = codebook[data[i]] into a primary_count-sized
    // array and reading it back twice, GroupBits and Pack look up
    // codebook[data[i]] on the fly. This removes a full pass and the huff_array
    // allocation.

    // Parallel deflate sizing. The actual bit-packing into the final buffer is
    // deferred to Serialize (once the output layout is known) so that each
    // chunk can be written directly to its final offset, eliminating a separate
    // condense/gather pass. Here we only compute the output geometry:
    //   per-group bit sums -> scan -> per-chunk bit lengths + word counts
    //   -> scan -> per-chunk word offsets (and the total ddata word count).
    SIZE groups_per_chunk =
        (chunk_size - 1) / GetDeflateGroupSize<DeviceType>(chunk_size) + 1;
    auto nchunk = (primary_count - 1) / chunk_size + 1;
    SIZE ngroups = (SIZE)(nchunk * groups_per_chunk);

    DeviceLauncher<DeviceType>::Execute(
        DeflateGroupBitsKernel<Q, H, DeviceType>(
            primary_subarray, workspace.codebook_subarray,
            workspace.deflate_group_bits_subarray, primary_count, chunk_size,
            groups_per_chunk, ngroups),
        queue_idx);
    // mark("Huffman stage: deflate group-bits");
    DeviceCollective<DeviceType>::ScanSumExtended(
        ngroups, workspace.deflate_group_bits_subarray,
        workspace.deflate_group_offsets_subarray,
        workspace.deflate_group_scan_workspace, true, queue_idx);
    DeviceLauncher<DeviceType>::Execute(
        DeflateChunkMetaKernel<H, DeviceType>(
            workspace.deflate_group_offsets_subarray,
            workspace.huff_bitwidths_subarray,
            workspace.deflate_chunk_words_subarray, (SIZE)nchunk,
            groups_per_chunk),
        queue_idx);
    DeviceCollective<DeviceType>::ScanSumExtended(
        (SIZE)nchunk, workspace.deflate_chunk_words_subarray,
        workspace.deflate_chunk_word_offsets_subarray,
        workspace.deflate_chunk_scan_workspace, true, queue_idx);
    // Total number of H-words in the densely packed stream.
    MemoryManager<DeviceType>::Copy1D(
        &ddata_size,
        workspace.deflate_chunk_word_offsets_subarray.data() + nchunk, 1,
        queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    // mark("Huffman stage: deflate sizing (scans+meta)");

    // Pack the densely-coded bitstream directly into its final location in the
    // output buffer (zero-copy): the byte offset where the packed stream lands
    // is fully determined by the sizes of the preceding sections, all known
    // here, so we can size the output now and have the pack kernel write
    // straight to it. Serialize then only fills in the surrounding metadata.
    // Each group writes MSB-first to its final intra-chunk offset; words fully
    // owned by a group use plain stores while the (<=2) words shared with
    // neighbouring groups use atomicOr, so the destination must be zeroed
    // first.
    SIZE packed_byte_offset;
    SIZE compressed_size = ComputeSerializedLayout(packed_byte_offset);
    compressed_data.resize({compressed_size}, queue_idx);
    SubArray<1, Byte, DeviceType> compressed_data_subarray(compressed_data);
    SubArray<1, H, DeviceType> packed_subarray(
        {(SIZE)ddata_size}, (H *)compressed_data_subarray(packed_byte_offset));
    MemoryManager<DeviceType>::Memset1D(packed_subarray.data(), ddata_size, 0,
                                        queue_idx);
    DeviceLauncher<DeviceType>::Execute(
        DeflatePackKernel<Q, H, DeviceType>(
            primary_subarray, workspace.codebook_subarray,
            workspace.deflate_group_offsets_subarray,
            workspace.deflate_chunk_word_offsets_subarray, packed_subarray,
            primary_count, chunk_size, groups_per_chunk, ngroups),
        queue_idx);
    // mark("Huffman stage: deflate pack");

    // Serialize(compressed_data, queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman compress", primary_count * sizeof(Q));
      timer.clear();
    }
    return true;
  }

  // Walks the serialized layout, returning the total compressed size and, via
  // packed_byte_offset, the byte offset at which the densely-packed Huffman
  // stream lands. Must stay in lockstep with the section order written by
  // Serialize. Depends only on values known after CompressPrimary's sizing
  // pass (primary_count, dict_size, chunk_size, ddata_size, outlier_count).
  SIZE ComputeSerializedLayout(SIZE &packed_byte_offset) {
    auto nchunk = (primary_count - 1) / chunk_size + 1;
    size_t type_bw = sizeof(H) * 8;
    size_t huffmeta_size = 2 * nchunk;
    SIZE byte_offset = 0;
    advance_with_align<Byte>(byte_offset, 7); // signature
    advance_with_align<size_t>(byte_offset, 1);
    advance_with_align<int>(byte_offset, 1);
    advance_with_align<int>(byte_offset, 1);
    advance_with_align<size_t>(byte_offset, 1);
    advance_with_align<size_t>(byte_offset, huffmeta_size);
    advance_with_align<size_t>(byte_offset, 1);
    advance_with_align<uint8_t>(
        byte_offset, (sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size));
    advance_with_align<size_t>(byte_offset, 1);
    align_byte_offset<H>(byte_offset);
    packed_byte_offset = byte_offset;
    advance_with_align<H>(byte_offset, ddata_size);
    // outlier
    advance_with_align<ATOMIC_IDX>(byte_offset, 1);
    advance_with_align<ATOMIC_IDX>(byte_offset, outlier_count);
    advance_with_align<S>(byte_offset, outlier_count);
    return byte_offset;
  }

  void Serialize(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    auto nchunk = (primary_count - 1) / chunk_size + 1;

    // out_meta: |signature|primary count|dict size|chunk size|huffmeta size|
    // huffmeta|decodebook size|decodebook|ddata size|
    // out_data: |huffman data|
    // huffmeta = |per-chunk bit lengths|per-chunk word offsets|, both already
    // computed on the device during CompressPrimary. ddata_size (total packed
    // word count) was also computed there.

    size_t type_bw = sizeof(H) * 8;
    size_t decodebook_size = workspace.decodebook_subarray.shape(0);
    size_t huffmeta_size = 2 * nchunk;

    // The output buffer was already sized and the packed Huffman stream already
    // written into it by CompressPrimary (zero-copy). Here we only fill in the
    // metadata sections around it; the packed region is skipped, not rewritten.
    SubArray compressed_data_subarray(compressed_data);

    SIZE byte_offset = 0;
    SerializeArray<Byte>(compressed_data_subarray, signature, 7, byte_offset,
                         queue_idx);
    SerializeArray<size_t>(compressed_data_subarray, &primary_count, 1,
                           byte_offset, queue_idx);
    SerializeArray<int>(compressed_data_subarray, &dict_size, 1, byte_offset,
                        queue_idx);
    SerializeArray<int>(compressed_data_subarray, &chunk_size, 1, byte_offset,
                        queue_idx);
    SerializeArray<size_t>(compressed_data_subarray, &huffmeta_size, 1,
                           byte_offset, queue_idx);
    // huffmeta first half: per-chunk bit lengths (device).
    SerializeArray<size_t>(compressed_data_subarray,
                           workspace.huff_bitwidths_subarray.data(), nchunk,
                           byte_offset, queue_idx);
    // huffmeta second half: per-chunk word offsets (device). The extended scan
    // holds nchunk+1 entries; the first nchunk are the chunk start offsets.
    SerializeArray<size_t>(compressed_data_subarray,
                           workspace.deflate_chunk_word_offsets_subarray.data(),
                           nchunk, byte_offset, queue_idx);
    SerializeArray<size_t>(compressed_data_subarray, &decodebook_size, 1,
                           byte_offset, queue_idx);
    SerializeArray<uint8_t>(compressed_data_subarray,
                            workspace.decodebook_subarray.data(),
                            (sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size),
                            byte_offset, queue_idx);
    SerializeArray<size_t>(compressed_data_subarray, &ddata_size, 1,
                           byte_offset, queue_idx);

    // The densely-packed Huffman stream is already in place (written directly
    // by CompressPrimary); just advance past it. Must mirror
    // ComputeSerializedLayout.
    advance_with_align<H>(byte_offset, ddata_size);

    // outlier
    SerializeArray<ATOMIC_IDX>(compressed_data_subarray, &outlier_count, 1,
                               byte_offset, queue_idx);
    SerializeArray<ATOMIC_IDX>(compressed_data_subarray,
                               workspace.outlier_idx_subarray.data(),
                               outlier_count, byte_offset, queue_idx);
    SerializeArray<S>(compressed_data_subarray,
                      workspace.outlier_subarray.data(), outlier_count,
                      byte_offset, queue_idx);

    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    log::dbg("Huffman block size: " + std::to_string(chunk_size));
    log::dbg("Huffman dictionary size: " + std::to_string(dict_size));
    log::dbg("Huffman compress ratio (primary): " +
             std::to_string(primary_count * sizeof(Q)) + "/" +
             std::to_string(ddata_size * sizeof(H)) + " (" +
             std::to_string((double)primary_count * sizeof(Q) /
                            (ddata_size * sizeof(H))) +
             ")");
    log::info(
        "Huffman compress ratio: " + std::to_string(primary_count * sizeof(Q)) +
        "/" + std::to_string(compressed_data.shape(0)) + " (" +
        std::to_string((double)primary_count * sizeof(Q) /
                       compressed_data.shape(0)) +
        ")");
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman serialize", compressed_data.shape(0));
      timer.clear();
    }
  }

  bool Verify(Array<1, Byte, DeviceType> &compressed_data, int queue_idx) {
    SubArray compressed_subarray(compressed_data);
    SIZE byte_offset = 0;
    // Deserialize the signature into the workspace's pinned host buffer; the
    // device buffer of the Array is unused (the comparison happens on host).
    Byte *signature_verify = workspace.signature_verify_array.dataHost();
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
    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }
    if (!Verify(compressed_data, queue_idx)) {
      throw std::runtime_error("Huffman signature mismatch.");
    }

    SubArray compressed_subarray(compressed_data);

    Byte *signature_ptr = nullptr;
    size_t *primary_count_ptr = &primary_count;
    int *dict_size_ptr = &dict_size;
    int *chunk_size_ptr = &chunk_size;
    size_t *huffmeta_size_ptr = &huffmeta_size;
    size_t *decodebook_size_ptr = &decodebook_size;
    size_t *ddata_size_ptr = &ddata_size;
    ATOMIC_IDX *outlier_count_ptr = &outlier_count;

    SIZE byte_offset = 0;
    DeserializeArray<Byte>(compressed_subarray, signature_ptr, 7, byte_offset,
                           true, queue_idx);
    DeserializeArray<size_t>(compressed_subarray, primary_count_ptr, 1,
                             byte_offset, false, queue_idx);
    DeserializeArray<int>(compressed_subarray, dict_size_ptr, 1, byte_offset,
                          false, queue_idx);
    DeserializeArray<int>(compressed_subarray, chunk_size_ptr, 1, byte_offset,
                          false, queue_idx);
    DeserializeArray<size_t>(compressed_subarray, huffmeta_size_ptr, 1,
                             byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    DeserializeArray<size_t>(compressed_subarray, huffmeta, huffmeta_size,
                             byte_offset, true, queue_idx);
    DeserializeArray<size_t>(compressed_subarray, decodebook_size_ptr, 1,
                             byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    DeserializeArray<uint8_t>(compressed_subarray, decodebook, decodebook_size,
                              byte_offset, true, queue_idx);
    DeserializeArray<size_t>(compressed_subarray, ddata_size_ptr, 1,
                             byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    DeserializeArray<H>(compressed_subarray, ddata, ddata_size, byte_offset,
                        true, queue_idx);

    // outlier
    DeserializeArray<ATOMIC_IDX>(compressed_subarray, outlier_count_ptr, 1,
                                 byte_offset, false, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    DeserializeArray<ATOMIC_IDX>(compressed_subarray, outlier_idx,
                                 outlier_count, byte_offset, true, queue_idx);
    DeserializeArray<S>(compressed_subarray, outlier, outlier_count,
                        byte_offset, true, queue_idx);
    workspace.outlier_idx_subarray =
        SubArray<1, ATOMIC_IDX, DeviceType>({(SIZE)outlier_count}, outlier_idx);
    workspace.outlier_subarray =
        SubArray<1, S, DeviceType>({(SIZE)outlier_count}, outlier);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman deserialize", compressed_data.shape(0));
      timer.clear();
    }
  }

  void DecompressPrimary(Array<1, Byte, DeviceType> &compressed_data,
                         Array<1, Q, DeviceType> &primary_data, int queue_idx) {

    // Deserialize(compressed_data, queue_idx);

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    SubArray<1, H, DeviceType> ddata_subarray({(SIZE)ddata_size}, ddata);
    SubArray<1, size_t, DeviceType> huffmeta_subarray({(SIZE)huffmeta_size},
                                                      huffmeta);
    primary_data.resize({(SIZE)primary_count});
    SubArray primary_subarray(primary_data);

    SubArray<1, uint8_t, DeviceType> decodebook_subarray(
        {(SIZE)decodebook_size}, decodebook);

    int nchunk = (primary_count - 1) / chunk_size + 1;
    Decode<Q, H, DeviceType>(
        ddata_subarray, huffmeta_subarray, primary_subarray, primary_count,
        chunk_size, nchunk, decodebook_subarray, decodebook_size, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman decompress", primary_count * sizeof(Q));
      timer.clear();
    }
  }

  bool Compress(Array<1, S, DeviceType> &original_data,
                Array<1, Byte, DeviceType> &compressed_data, float target_cr,
                int queue_idx) {

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    ATOMIC_IDX zero = 0;
    MemoryManager<DeviceType>::Copy1D(workspace.outlier_count_subarray.data(),
                                      &zero, 1, queue_idx);

    // The dictionary shift (+dict_size/2) is folded into the quantizer, so the
    // incoming data already lives in the non-negative dictionary range. A
    // single block-aggregated pass separates the out-of-range outliers.
    DeviceLauncher<DeviceType>::Execute(
        OutlierSeparatorKernel<S, MGARDX_SEPARATE_OUTLIER, DeviceType>(
            SubArray(original_data), dict_size,
            workspace.outlier_count_subarray, workspace.outlier_idx_subarray,
            workspace.outlier_subarray),
        queue_idx);
    MemoryManager<DeviceType>::Copy1D(
        &outlier_count, workspace.outlier_count_subarray.data(), 1, queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    log::info(
        "Outlier ratio: " + std::to_string(outlier_count) + "/" +
        std::to_string(original_data.shape(0)) + " (" +
        std::to_string((double)100 * outlier_count / original_data.shape(0)) +
        "%)");
    if (outlier_count > workspace.outlier_subarray.shape(0)) {
      throw std::runtime_error("Not enough workspace for outliers.");
    }

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman outlier", original_data.shape(0) * sizeof(S));
      timer.clear();
    }

    // Cast to unsigned type
    Array<1, Q, DeviceType> primary_data({original_data.shape(0)},
                                         (Q *)original_data.data());
    return CompressPrimary(primary_data, compressed_data, target_cr, queue_idx);
  }

  void Decompress(Array<1, Byte, DeviceType> &compressed_data,
                  Array<1, S, DeviceType> &decompressed_data, int queue_idx) {

    // Cast to unsigned type.
    // We must use the correct size to avoid resize to new allocation
    Array<1, Q, DeviceType> primary_data({decompressed_data.shape(0)},
                                         (Q *)decompressed_data.data());
    DecompressPrimary(compressed_data, primary_data, queue_idx);

    Timer timer;
    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.start();
    }

    // Restore the outliers; the dictionary shift is undone later in the
    // dequantizer (mirroring the quantizer), so no shift pass is needed here.
    DeviceLauncher<DeviceType>::Execute(
        OutlierSeparatorKernel<S, MGARDX_RESTORE_OUTLIER, DeviceType>(
            decompressed_data, dict_size, workspace.outlier_count_subarray,
            workspace.outlier_idx_subarray, workspace.outlier_subarray),
        queue_idx);
    DeviceRuntime<DeviceType>::SyncQueue(queue_idx);

    if (log::level & log::TIME) {
      DeviceRuntime<DeviceType>::SyncQueue(queue_idx);
      timer.end();
      timer.print("Huffman outlier", decompressed_data.shape(0) * sizeof(S));
      timer.clear();
    }
  }

  bool initialized;
  SIZE max_size;
  size_t primary_count;
  ATOMIC_IDX outlier_count = 0;
  int dict_size;
  int chunk_size;
  size_t huffmeta_size;
  size_t decodebook_size;
  size_t ddata_size;
  size_t *huffmeta;
  uint8_t *decodebook;
  ATOMIC_IDX *outlier_idx;
  S *outlier;
  H *ddata;
  Byte signature[7] = {'M', 'G', 'X', 'H', 'U', 'F', 'F'};
  HuffmanWorkspace<Q, S, H, DeviceType> workspace;
};

} // namespace mgard_x

#endif
