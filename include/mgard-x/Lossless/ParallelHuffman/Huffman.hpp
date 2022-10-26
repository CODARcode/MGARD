/*
 * Copyright 2022, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: March 17, 2022
 */

#ifndef MGARD_X_HUFFMAN_TEMPLATE_HPP
#define MGARD_X_HUFFMAN_TEMPLATE_HPP

static bool debug_print_huffman = false;

#include "Condense.hpp"
#include "Decode.hpp"
#include "Deflate.hpp"
#include "EncodeFixedLen.hpp"
#include "GetCodebook.hpp"
#include "Histogram.hpp"
#include "HuffmanWorkspace.hpp"
#include "OutlierSeparator.hpp"
#include "DictionaryShift.hpp"

#include <chrono>
using namespace std::chrono;

namespace mgard_x {

template <typename Q, typename S, typename H, typename DeviceType>
void HuffmanCompress(SubArray<1, Q, DeviceType> &dprimary_subarray,
                     int chunk_size, int dict_size, LENGTH outlier_count,
                     SubArray<1, LENGTH, DeviceType> outlier_idx_subarray,
                     SubArray<1, QUANTIZED_INT, DeviceType> outlier_subarray,
                     Array<1, Byte, DeviceType> &compressed_data,
                     HuffmanWorkspace<Q, S, H, DeviceType> &workspace) {

  Timer timer;
  if (log::level & log::TIME)
    timer.start();

  workspace.reset(0);

  size_t primary_count = dprimary_subarray.shape(0);

  Histogram<Q, unsigned int, DeviceType>(
      dprimary_subarray, workspace.freq_subarray, primary_count, dict_size, 0);
  // DeviceRuntime<DeviceType>::SyncQueue(0);

  if (debug_print_huffman) {
    PrintSubarray("Histogram::freq_subarray", workspace.freq_subarray);
  }

  GetCodebook(dict_size, workspace.freq_subarray,
                                workspace.codebook_subarray,
                                workspace.decodebook_subarray, workspace);
  if (debug_print_huffman) {
    PrintSubarray("GetCodebook::codebook_subarray",
                  workspace.codebook_subarray);
    PrintSubarray("GetCodebook::decodebook_subarray",
                  workspace.decodebook_subarray);
  }

  DeviceLauncher<DeviceType>::Execute(
      EncodeFixedLenKernel<unsigned int, H, DeviceType>(
          dprimary_subarray, workspace.huff_subarray,
          workspace.codebook_subarray),
      0);
  // DeviceRuntime<DeviceType>::SyncQueue(0);

  if (debug_print_huffman) {
    PrintSubarray("EncodeFixedLen::huff_subarray", workspace.huff_subarray);
  }

  // deflate
  DeviceLauncher<DeviceType>::Execute(
      DeflateKernel<H, DeviceType>(workspace.huff_subarray,
                                   workspace.huff_bitwidths_subarray,
                                   chunk_size),
      0);
  // DeviceRuntime<DeviceType>::SyncQueue(0);

  if (debug_print_huffman) {
    PrintSubarray("Deflate::huff_subarray", workspace.huff_subarray);
    PrintSubarray("Deflate::huff_bitwidths_subarray",
                  workspace.huff_bitwidths_subarray);
  }

  auto nchunk = (primary_count - 1) / chunk_size + 1;
  size_t *h_meta = new size_t[nchunk * 3]();
  size_t *dH_uInt_meta = h_meta;
  size_t *dH_bit_meta = h_meta + nchunk;
  size_t *dH_uInt_entry = h_meta + nchunk * 2;

  MemoryManager<DeviceType>().Copy1D(
      dH_bit_meta, workspace.huff_bitwidths_subarray.data(), nchunk, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  // transform in uInt
  memcpy(dH_uInt_meta, dH_bit_meta, nchunk * sizeof(size_t));
  std::for_each(dH_uInt_meta, dH_uInt_meta + nchunk,
                [&](size_t &i) { i = (i - 1) / (sizeof(H) * 8) + 1; });
  // make it entries
  memcpy(dH_uInt_entry + 1, dH_uInt_meta, (nchunk - 1) * sizeof(size_t));
  for (auto i = 1; i < nchunk; i++)
    dH_uInt_entry[i] += dH_uInt_entry[i - 1];

  // sum bits from each chunk
  auto total_bits =
      std::accumulate(dH_bit_meta, dH_bit_meta + nchunk, (size_t)0);
  auto total_uInts =
      std::accumulate(dH_uInt_meta, dH_uInt_meta + nchunk, (size_t)0);

  // printf("huffman encode time: %.6f s\n", time_span.count());

  // out_meta: |outlier count|outlier idx|outlier data|primary count|dict
  // size|chunk size|huffmeta size|huffmeta|decodebook size|decodebook|
  // out_data: |huffman data|

  size_t type_bw = sizeof(H) * 8;
  size_t decodebook_size = workspace.decodebook_subarray.shape(0);
  size_t huffmeta_size = 2 * nchunk;

  size_t ddata_size = total_uInts;

  SIZE byte_offset = 0;
  advance_with_align<size_t>(byte_offset, 1);
  advance_with_align<int>(byte_offset, 1);
  advance_with_align<int>(byte_offset, 1);
  advance_with_align<size_t>(byte_offset, 1);
  advance_with_align<size_t>(byte_offset, huffmeta_size);
  advance_with_align<size_t>(byte_offset, 1);
  advance_with_align<uint8_t>(
      byte_offset, (sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size));
  advance_with_align<size_t>(byte_offset, 1);
  advance_with_align<H>(byte_offset, ddata_size);
  // outliter
  advance_with_align<LENGTH>(byte_offset, 1);
  advance_with_align<LENGTH>(byte_offset, outlier_count);
  advance_with_align<QUANTIZED_INT>(byte_offset, outlier_count);

  compressed_data.resize({(SIZE)(byte_offset)});
  SubArray compressed_data_subarray(compressed_data);

  byte_offset = 0;
  SerializeArray<size_t>(compressed_data_subarray, &primary_count, 1,
                         byte_offset);
  SerializeArray<int>(compressed_data_subarray, &dict_size, 1, byte_offset);
  SerializeArray<int>(compressed_data_subarray, &chunk_size, 1, byte_offset);
  SerializeArray<size_t>(compressed_data_subarray, &huffmeta_size, 1,
                         byte_offset);
  SerializeArray<size_t>(compressed_data_subarray, dH_bit_meta, huffmeta_size,
                         byte_offset);
  SerializeArray<size_t>(compressed_data_subarray, &decodebook_size, 1,
                         byte_offset);
  SerializeArray<uint8_t>(
      compressed_data_subarray, workspace.decodebook_subarray.data(),
      (sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size), byte_offset);
  SerializeArray<size_t>(compressed_data_subarray, &ddata_size, 1, byte_offset);

  align_byte_offset<H>(byte_offset);

  MemoryManager<DeviceType>::Copy1D(
      workspace.condense_write_offsets_subarray.data(), dH_uInt_entry, nchunk,
      0);
  MemoryManager<DeviceType>::Copy1D(
      workspace.condense_actual_lengths_subarray.data(), dH_uInt_meta, nchunk,
      0);
  SubArray<1, H, DeviceType> compressed_data_cast_subarray(
      {(SIZE)ddata_size}, (H *)compressed_data_subarray(byte_offset));
  DeviceLauncher<DeviceType>::Execute(
      CondenseKernel<H, DeviceType>(workspace.huff_subarray,
                                    workspace.condense_write_offsets_subarray,
                                    workspace.condense_actual_lengths_subarray,
                                    compressed_data_cast_subarray, chunk_size),
      0);

  advance_with_align<H>(byte_offset, ddata_size);

  // outlier
  SerializeArray<LENGTH>(compressed_data_subarray, &outlier_count, 1,
                         byte_offset);
  SerializeArray<LENGTH>(compressed_data_subarray, outlier_idx_subarray.data(),
                         outlier_count, byte_offset);
  SerializeArray<QUANTIZED_INT>(compressed_data_subarray,
                                outlier_subarray.data(), outlier_count,
                                byte_offset);

  DeviceRuntime<DeviceType>::SyncQueue(0);

  delete[] h_meta;

  log::info("Huffman block size: " + std::to_string(chunk_size));
  log::info("Huffman dictionary size: " + std::to_string(dict_size));
  log::info(
      "Huffman compress ratio: " +
      std::to_string(primary_count * sizeof(QUANTIZED_UNSIGNED_INT)) + "/" +
      std::to_string(compressed_data.shape(0)) + " (" +
      std::to_string((double)primary_count * sizeof(QUANTIZED_UNSIGNED_INT) /
                     compressed_data.shape(0)) +
      ")");

  if (log::level & log::TIME) {
    timer.end();
    timer.print("Huffman compress");
    timer.clear();
  }
}

template <typename Q, typename S, typename H, typename DeviceType>
void HuffmanDecompress(SubArray<1, Byte, DeviceType> compressed_data,
                       Array<1, Q, DeviceType> &primary, LENGTH &outlier_count,
                       SubArray<1, LENGTH, DeviceType> &outlier_idx_subarray,
                       SubArray<1, QUANTIZED_INT, DeviceType> &outlier_subarray,
                       HuffmanWorkspace<Q, S, H, DeviceType> &workspace) {
  Timer timer;
  if (log::level & log::TIME)
    timer.start();
  size_t primary_count;
  int dict_size;
  int chunk_size;
  size_t huffmeta_size;
  size_t *huffmeta;
  size_t decodebook_size;
  uint8_t *decodebook;
  size_t ddata_size;
  // LENGTH outlier_count;
  LENGTH *outlier_idx;
  QUANTIZED_INT *outlier;

  size_t *primary_count_ptr = &primary_count;
  int *dict_size_ptr = &dict_size;
  int *chunk_size_ptr = &chunk_size;
  size_t *huffmeta_size_ptr = &huffmeta_size;
  size_t *decodebook_size_ptr = &decodebook_size;
  size_t *ddata_size_ptr = &ddata_size;
  LENGTH *outlier_count_ptr = &outlier_count;

  H *ddata;

  SIZE byte_offset = 0;
  DeserializeArray<size_t>(compressed_data, primary_count_ptr, 1, byte_offset,
                           false);
  DeserializeArray<int>(compressed_data, dict_size_ptr, 1, byte_offset, false);
  DeserializeArray<int>(compressed_data, chunk_size_ptr, 1, byte_offset, false);
  DeserializeArray<size_t>(compressed_data, huffmeta_size_ptr, 1, byte_offset,
                           false);
  DeserializeArray<size_t>(compressed_data, huffmeta, huffmeta_size,
                           byte_offset, true);
  DeserializeArray<size_t>(compressed_data, decodebook_size_ptr, 1, byte_offset,
                           false);
  DeserializeArray<uint8_t>(compressed_data, decodebook, decodebook_size,
                            byte_offset, true);
  DeserializeArray<size_t>(compressed_data, ddata_size_ptr, 1, byte_offset,
                           false);
  // align_byte_offset<H>(byte_offset);
  // SubArray<1, H, DeviceType> ddata_subarray({(SIZE)ddata_size},
  // (H*)compressed_data((IDX)byte_offset)); byte_offset += ddata_size *
  // sizeof(H);
  DeserializeArray<H>(compressed_data, ddata, ddata_size, byte_offset, true);

  // outlier
  DeserializeArray<LENGTH>(compressed_data, outlier_count_ptr, 1, byte_offset,
                           false);
  DeserializeArray<LENGTH>(compressed_data, outlier_idx, outlier_count,
                           byte_offset, true);
  DeserializeArray<QUANTIZED_INT>(compressed_data, outlier, outlier_count,
                                  byte_offset, true);
  outlier_idx_subarray =
      SubArray<1, LENGTH, DeviceType>({(SIZE)outlier_count}, outlier_idx);
  outlier_subarray =
      SubArray<1, QUANTIZED_INT, DeviceType>({(SIZE)outlier_count}, outlier);

  SubArray<1, H, DeviceType> ddata_subarray({(SIZE)ddata_size}, ddata);
  SubArray<1, size_t, DeviceType> huffmeta_subarray({(SIZE)huffmeta_size},
                                                    huffmeta);
  primary.resize({(SIZE)primary_count});
  SubArray primary_subarray(primary);

  SubArray<1, uint8_t, DeviceType> decodebook_subarray({(SIZE)decodebook_size},
                                                       decodebook);
  int nchunk = (primary_count - 1) / chunk_size + 1;
  Decode<Q, H, DeviceType>(ddata_subarray, huffmeta_subarray, primary_subarray,
                           primary_count, chunk_size, nchunk,
                           decodebook_subarray, decodebook_size, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  if (log::level & log::TIME) {
    timer.end();
    timer.print("Huffman decompress");
    timer.clear();
  }
}

template <typename Q, typename S, typename H, typename DeviceType>
void HuffmanCompress(SubArray<1, Q, DeviceType> &in_subarray,
                     int chunk_size, int dict_size,
                     Array<1, Byte, DeviceType> &compressed_data,
                     HuffmanWorkspace<Q, S, H, DeviceType> &workspace) {

}


} // namespace mgard_x

#endif