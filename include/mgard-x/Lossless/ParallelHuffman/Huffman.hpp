/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

static bool debug_print_huffman = false;

#include "Histogram.hpp"
#include "GetCodebook.hpp"
#include "EncodeFixedLen.hpp"
#include "Deflate.hpp"
#include "Decode.hpp"

#include <chrono>

using namespace std::chrono;

#ifndef MGARD_X_HUFFMAN_TEMPLATE_HPP
#define MGARD_X_HUFFMAN_TEMPLATE_HPP

namespace mgard_x {



template <typename T>
void align_byte_offset(SIZE &byte_offset) { 
  if (byte_offset % sizeof(T) != 0) {
    byte_offset = ((byte_offset - 1) / sizeof(T) + 1) * sizeof(T);
  }
}

template <typename T>
void advance_with_align(SIZE &byte_offset, SIZE count) { 
  align_byte_offset<T>(byte_offset);
  byte_offset += count * sizeof(T);
}

template <typename T, typename DeviceType>
void SerializeArray(SubArray<1, Byte, DeviceType> &array, T * data_ptr, SIZE count, SIZE &byte_offset) {
  using Mem = MemoryManager<DeviceType>;
  align_byte_offset<T>(byte_offset);
  Mem::Copy1D(array(byte_offset), (Byte *)data_ptr, count * sizeof(T), 0);
  byte_offset += count * sizeof(T);
  DeviceRuntime<DeviceType>::SyncQueue(0);
}

template <typename T, typename DeviceType>
void DeserializeArray(SubArray<1, Byte, DeviceType> &array, T * & data_ptr, SIZE count, SIZE &byte_offset, bool zero_copy) {
  using Mem = MemoryManager<DeviceType>;
  align_byte_offset<T>(byte_offset);
  if (zero_copy) {
    data_ptr = (T*)array(byte_offset);
  } else {
    Mem::Copy1D((Byte *)data_ptr, array(byte_offset), count * sizeof(T), 0);
  }
  byte_offset += count * sizeof(T);
  DeviceRuntime<DeviceType>::SyncQueue(0);
}

template <typename T, typename DeviceType>
void CompareSubArrays(SubArray<1, T, DeviceType> array1, SubArray<1, T, DeviceType> array2) {
  SIZE n = array1.shape[0];
  using Mem = MemoryManager<DeviceType>;
  T * q1 = new T[n];
  T * q2 = new T[n];
  Mem::Copy1D(q1, array1.data(), n, 0);
  Mem::Copy1D(q2, array2.data(), n, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  bool pass = true;
  for (int i = 0; i < n; i++) {
    if (q1[i] != q2[i] ) {
      pass = false;
      std::cout << "diff at " << i << "(" << q1[i] << ", " << q2[i] << ")\n";
    }
  }
  printf("pass: %d\n", pass);
  delete [] q1;
  delete [] q2;
}

template <typename Q, typename H, typename DeviceType>
Array<1, Byte, DeviceType>
HuffmanCompress(SubArray<1, Q, DeviceType>& dprimary_subarray,
                int chunk_size, int dict_size) {

  high_resolution_clock::time_point t1, t2, start, end;
  duration<double> time_span;

  size_t primary_count = dprimary_subarray.getShape(0);

  t1 = high_resolution_clock::now();

  int ht_state_num = 2 * dict_size;
  int ht_all_nodes = 2 * ht_state_num;

  Array<1, unsigned int, DeviceType> freq_array({(SIZE)ht_all_nodes});
  freq_array.memset(0);

  SubArray<1, unsigned int, DeviceType> freq_subarray(freq_array);
  Histogram<Q, unsigned int, DeviceType>().Execute(dprimary_subarray, freq_subarray, primary_count, dict_size, 0);
  DeviceRuntime<DeviceType>::SyncDevice();

  if (debug_print_huffman) {
    // PrintSubarray("Histogram::freq_subarray", freq_subarray);
  }

  auto type_bw = sizeof(H) * 8;
  size_t decodebook_size = sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size;
  Array<1, H, DeviceType> codebook_array({(SIZE)dict_size});
  codebook_array.memset(0);
  Array<1, uint8_t, DeviceType> decodebook_array({(SIZE)decodebook_size});
  codebook_array.memset(0xff);

  H * codebook = codebook_array.get_dv();
  uint8_t *decodebook = decodebook_array.get_dv();


  SubArray<1, H, DeviceType> codebook_subarray(codebook_array);
  SubArray<1, uint8_t, DeviceType> decodebook_subarray(decodebook_array);

  GetCodebook<Q, H, DeviceType>(dict_size, freq_subarray, codebook_subarray, decodebook_subarray);
  DeviceRuntime<DeviceType>::SyncDevice();

  if (debug_print_huffman) {
    // PrintSubarray("GetCodebook::codebook_subarray", codebook_subarray);
    // PrintSubarray("GetCodebook::decodebook_subarray", decodebook_subarray);
  }

  Array<1, H, DeviceType> huff_array({(SIZE)primary_count});
  huff_array.memset(0);
  H * huff = huff_array.get_dv();

  // gpuErrchk(DeviceTypeDeviceSynchronize());
  DeviceRuntime<DeviceType>::SyncDevice();

  SubArray<1, H, DeviceType> huff_subarray(huff_array);
  EncodeFixedLen<unsigned int, H, DeviceType>().Execute(dprimary_subarray,
                                                              huff_subarray,
                                                              primary_count,
                                                              codebook_subarray, 0);
  DeviceRuntime<DeviceType>::SyncDevice();
  if (debug_print_huffman) {
    // PrintSubarray("EncodeFixedLen::huff_subarray", huff_subarray);
  }

  // deflate
  auto nchunk = (primary_count - 1) / chunk_size + 1; 
  Array<1, size_t, DeviceType> huff_bitwidths_array({(SIZE)nchunk});
  huff_bitwidths_array.memset(0);
  size_t * huff_bitwidths = huff_bitwidths_array.get_dv();

  SubArray<1, size_t, DeviceType> huff_bitwidths_subarray({(SIZE)nchunk}, huff_bitwidths);
  Deflate<H, DeviceType>().Execute(huff_subarray, primary_count, huff_bitwidths_subarray, chunk_size, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  if (debug_print_huffman) {
    // PrintSubarray("Deflate::huff_subarray", huff_subarray);
    // PrintSubarray("Deflate::huff_bitwidths_subarray", huff_bitwidths_subarray);
  }

  size_t* h_meta = new size_t[nchunk * 3]();
  size_t* dH_uInt_meta = h_meta;
  size_t* dH_bit_meta = h_meta + nchunk;
  size_t* dH_uInt_entry = h_meta + nchunk * 2;

  MemoryManager<DeviceType>().Copy1D(dH_bit_meta, huff_bitwidths, nchunk, 0);
  // gpuErrchk(DeviceTypeDeviceSynchronize());
  DeviceRuntime<DeviceType>::SyncDevice();
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

  // gpuErrchk(DeviceTypeDeviceSynchronize());
  DeviceRuntime<DeviceType>::SyncDevice();
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("huffman encode time: %.6f s\n", time_span.count());

  // out_meta: |outlier count|outlier idx|outlier data|primary count|dict
  // size|chunk size|huffmeta size|huffmeta|decodebook size|decodebook|
  // out_data: |huffman data|

  size_t huffmeta_size = 2 * nchunk;

  t1 = high_resolution_clock::now();
  size_t ddata_size = total_uInts;

  SIZE byte_offset = 0;
  advance_with_align<size_t>(byte_offset, 1);
  advance_with_align<int>(byte_offset, 1);
  advance_with_align<int>(byte_offset, 1);
  advance_with_align<size_t>(byte_offset, 1);
  advance_with_align<size_t>(byte_offset, huffmeta_size);
  advance_with_align<size_t>(byte_offset, 1);
  advance_with_align<uint8_t>(byte_offset, (sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size));
  advance_with_align<size_t>(byte_offset, 1);
  advance_with_align<H>(byte_offset, ddata_size);
  Array<1, Byte, DeviceType> compressed_data({(SIZE)(byte_offset)});

  SubArray compressed_data_subarray(compressed_data);

  byte_offset = 0;
  SerializeArray<size_t>(compressed_data_subarray, &primary_count, 1, byte_offset);
  SerializeArray<int>(compressed_data_subarray, &dict_size, 1, byte_offset);
  SerializeArray<int>(compressed_data_subarray, &chunk_size, 1, byte_offset);
  SerializeArray<size_t>(compressed_data_subarray, &huffmeta_size, 1, byte_offset);
  SerializeArray<size_t>(compressed_data_subarray, h_meta + nchunk, huffmeta_size, byte_offset);
  SerializeArray<size_t>(compressed_data_subarray, &decodebook_size, 1, byte_offset);
  SerializeArray<uint8_t>(compressed_data_subarray, decodebook_subarray.data(), (sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size), byte_offset);
  SerializeArray<size_t>(compressed_data_subarray, &ddata_size, 1, byte_offset);
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("serilization time1: %.6f s\n", time_span.count());

  t1 = high_resolution_clock::now();

  align_byte_offset<H>(byte_offset);
  for (auto i = 0; i < nchunk; i++) {
    MemoryManager<DeviceType>::Copy1D((H*)compressed_data_subarray(byte_offset + dH_uInt_entry[i] * sizeof(H)),
                                      huff + i * chunk_size, dH_uInt_meta[i], 0);
  }

  DeviceRuntime<DeviceType>::SyncQueue(0);
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("serilization time2: %.6f s\n", time_span.count());
  delete[] h_meta;
  return compressed_data;
}

template <typename Q, typename H, typename DeviceType>
Array<1, Q, DeviceType>
HuffmanDecompress(SubArray<1, Byte, DeviceType> compressed_data) {

  size_t primary_count;
  int dict_size;
  int chunk_size;
  size_t huffmeta_size;
  size_t *huffmeta;
  size_t decodebook_size;
  uint8_t *decodebook;
  size_t ddata_size;

  size_t * primary_count_ptr = &primary_count;
  int * dict_size_ptr = &dict_size;
  int * chunk_size_ptr = &chunk_size;
  size_t * huffmeta_size_ptr = &huffmeta_size;
  size_t * decodebook_size_ptr = &decodebook_size;
  size_t * ddata_size_ptr = &ddata_size;

  H*ddata;
  
  SIZE byte_offset = 0;
  DeserializeArray<size_t>(compressed_data, primary_count_ptr, 1, byte_offset, false);
  DeserializeArray<int>(compressed_data, dict_size_ptr, 1, byte_offset, false);
  DeserializeArray<int>(compressed_data, chunk_size_ptr, 1, byte_offset, false);
  DeserializeArray<size_t>(compressed_data, huffmeta_size_ptr, 1, byte_offset, false);
  DeserializeArray<size_t>(compressed_data, huffmeta, huffmeta_size, byte_offset, true);
  DeserializeArray<size_t>(compressed_data, decodebook_size_ptr, 1, byte_offset, false);
  DeserializeArray<uint8_t>(compressed_data, decodebook, decodebook_size, byte_offset, true);
  DeserializeArray<size_t>(compressed_data, ddata_size_ptr, 1, byte_offset, false);

  Array<1, Q, DeviceType> dprimary({(SIZE)primary_count});
  align_byte_offset<H>(byte_offset);
  int nchunk = (primary_count - 1) / chunk_size + 1;
  SubArray<1, H, DeviceType> ddata_subarray({(SIZE)ddata_size}, (H*)compressed_data((IDX)byte_offset));
  SubArray<1, size_t, DeviceType> huffmeta_subarray({(SIZE)huffmeta_size}, huffmeta);
  SubArray<1, Q, DeviceType> dprimary_subarray(dprimary);
  SubArray<1, uint8_t, DeviceType> decodebook_subarray({(SIZE)decodebook_size}, decodebook);
  Decode<Q, H, DeviceType>().Execute(ddata_subarray,
                                                       huffmeta_subarray,
                                                       dprimary_subarray,
                                                       primary_count, chunk_size, nchunk,
                                                       decodebook_subarray, decodebook_size, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);
  return dprimary;
}




template <typename Q, typename H, typename DeviceType>
Array<1, Q, DeviceType>
HuffmanDecompress(Array<1, Byte, DeviceType> &compressed_data);


template <typename Q, typename H, typename DeviceType>
Array<1, Byte, DeviceType>
HuffmanCompress(Array<1, Q, DeviceType> &qv, int dict_size, int chunk_size) {
  using Mem = MemoryManager<DeviceType>;
  // high_resolution_clock::time_point t1, t2, start, end;
  // duration<double> time_span;
  size_t primary_count = qv.getShape()[0];

  // t1 = high_resolution_clock::now();
  // start huffman
  // histogram
  int ht_state_num = 2 * dict_size;
  int ht_all_nodes = 2 * ht_state_num;
  Array<1, unsigned int, DeviceType> freq_array({(SIZE)ht_all_nodes});
  freq_array.memset(0);

  SubArray dprimary_subarray(qv);
  SubArray freq_subarray(freq_array);

  Histogram<Q, unsigned int, DeviceType>().Execute(
      dprimary_subarray, freq_subarray, primary_count, dict_size, 0);
  
    // PrintSubarray("dprimary_subarray", dprimary_subarray);
  // PrintSubarray("freq_subarray", freq_subarray);

  auto type_bw = sizeof(H) * 8;
  size_t decodebook_size = sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size;
  Array<1, H, DeviceType> codebook_array({(SIZE)dict_size});
  codebook_array.memset(0);
  Array<1, uint8_t, DeviceType> decodebook_array({(SIZE)decodebook_size});
  codebook_array.memset(0xff);

  H * codebook = codebook_array.get_dv();
  uint8_t *decodebook = decodebook_array.get_dv();

  SubArray codebook_subarray(codebook_array);
  SubArray decodebook_subarray(decodebook_array);
  // Get codebooks
  // ParGetCodebook<Q, H, DeviceType>(dict_size, freq, codebook, decodebook);
  GetCodebook<Q, H, DeviceType>(dict_size, freq_subarray, codebook_subarray, decodebook_subarray);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  // Non-deflated output
  Array<1, H, DeviceType> huff_array({(SIZE)primary_count});
  huff_array.memset(0);
  H * huff = huff_array.get_dv();

  DeviceRuntime<DeviceType>::SyncQueue(0);

  SubArray<1, H, DeviceType> huff_subarray(huff_array);
  EncodeFixedLen<unsigned int, H, DeviceType>().Execute(dprimary_subarray,
                                                        huff_subarray,
                                                        primary_count,
                                                        codebook_subarray, 0);

  // deflate
  auto nchunk = (primary_count - 1) / chunk_size + 1; // |
  Array<1, size_t, DeviceType> huff_bitwidths_array({(SIZE)nchunk});
  huff_bitwidths_array.memset(0);

  SubArray huff_bitwidths_subarray(huff_bitwidths_array);
  Deflate<H, DeviceType>().Execute(huff_subarray, primary_count, huff_bitwidths_subarray, chunk_size, 0);

  DeviceRuntime<DeviceType>::SyncQueue(0);

  // dump TODO change to int
  size_t* h_meta = new size_t[nchunk * 3]();
  size_t* dH_uInt_meta = h_meta;
  size_t* dH_bit_meta = h_meta + nchunk;
  size_t* dH_uInt_entry = h_meta + nchunk * 2;
  // copy back densely Huffman code (dHcode)
  Mem::Copy1D(dH_bit_meta, huff_bitwidths_array.get_dv(), nchunk, 0);
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

  // gpuErrchk(DeviceTypeDeviceSynchronize());
  // t2 = high_resolution_clock::now();
  // time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("huffman encode time: %.6f s\n", time_span.count());

  // out_meta: |primary count|dict size|chunk size|
  //           |huffmeta size|huffmeta|
  //           |decodebook size|decodebook|
  // out_data: |huffman data|

  // t1 = high_resolution_clock::now();

  size_t huffmeta_size = 2 * nchunk;
  SIZE byte_offset = 0;
  advance_with_align<size_t>(byte_offset, 1); 
  advance_with_align<int>(byte_offset, 1);
  advance_with_align<int>(byte_offset, 1);
  advance_with_align<size_t>(byte_offset, 1);
  advance_with_align<size_t>(byte_offset, huffmeta_size);
  advance_with_align<size_t>(byte_offset, 1);
  advance_with_align<uint8_t>(byte_offset, decodebook_size);
  advance_with_align<size_t>(byte_offset, 1);
  advance_with_align<H>(byte_offset, total_uInts);

  // SIZE meta_size = 
  //     sizeof(size_t) + sizeof(int) + sizeof(int) + 
  //     sizeof(size_t) + 2 * nchunk * sizeof(size_t) + sizeof(size_t) +
  //     (sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size) * sizeof(uint8_t);
  // SIZE data_size = total_uInts * sizeof(H);

  Array<1, Byte, DeviceType> compressed_data({byte_offset});
  SubArray compressed_data_subarray(compressed_data);
  byte_offset = 0;
  SerializeArray<size_t>(compressed_data_subarray, &primary_count,             1,               byte_offset);
  SerializeArray<int>   (compressed_data_subarray, &dict_size,                 1,               byte_offset);
  SerializeArray<int>   (compressed_data_subarray, &chunk_size,                1,               byte_offset);
  SerializeArray<size_t>(compressed_data_subarray, &huffmeta_size,             1,               byte_offset);
  SerializeArray<size_t>(compressed_data_subarray, h_meta + nchunk,           huffmeta_size,   byte_offset);
  SerializeArray<size_t>(compressed_data_subarray, &decodebook_size,           1,               byte_offset);
  SerializeArray<Byte>  (compressed_data_subarray, decodebook_array.get_dv(), decodebook_size, byte_offset);
  SerializeArray<size_t>(compressed_data_subarray, &total_uInts,               1,             byte_offset);
  
  align_byte_offset<H>(byte_offset);
  for (auto i = 0; i < nchunk; i++) {
    Mem::Copy1D((H*)compressed_data_subarray(byte_offset + dH_uInt_entry[i]), 
                huff_subarray(i * chunk_size), dH_uInt_meta[i], 0);
  }

  SubArray<1, H, DeviceType> huff_subarray3({(SIZE)total_uInts}, (H*)compressed_data_subarray(byte_offset));

  size_t * meta;
  Mem::Malloc1D(meta, huffmeta_size, 0);
  Mem::Copy1D(meta, h_meta + nchunk, huffmeta_size, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  SubArray<1, size_t, DeviceType> huffmeta_subarray({(SIZE)huffmeta_size}, meta);
  
  DeviceRuntime<DeviceType>::SyncQueue(0);

  // PrintSubarray("huff_subarray", huff_subarray);

  std::cout << "primary_count: " << primary_count << "\n";
  std::cout << "dict_size: " << dict_size << "\n";
  std::cout << "chunk_size: " << chunk_size << "\n";
  std::cout << "huffmeta_size: " << huffmeta_size << "\n";
  std::cout << "decodebook_size: " << decodebook_size << "\n";
  std::cout << "total_uInts: " << total_uInts << "\n";

  {
    size_t primary_count;
    int dict_size;
    int chunk_size;
    size_t huffmeta_size;
    size_t decodebook_size;
    size_t total_uInts;

    size_t * primary_count_ptr = &primary_count;
    int * dict_size_ptr = &dict_size;
    int * chunk_size_ptr = &chunk_size;
    size_t * huffmeta_size_ptr = &huffmeta_size;
    size_t * decodebook_size_ptr = &decodebook_size;
    size_t * total_uInts_ptr = &total_uInts;

    size_t *huffmeta;
    Byte *decodebook;
    H * huff;

    SubArray compressed_data_subarray(compressed_data);

    SIZE byte_offset = 0;
    DeserializeArray<size_t>(compressed_data_subarray, primary_count_ptr,   1,               byte_offset, false);
    DeserializeArray<int>   (compressed_data_subarray, dict_size_ptr,       1,               byte_offset, false);
    DeserializeArray<int>   (compressed_data_subarray, chunk_size_ptr,      1,               byte_offset, false);
    DeserializeArray<size_t>(compressed_data_subarray, huffmeta_size_ptr,   1,               byte_offset, false);
    DeserializeArray<size_t>(compressed_data_subarray, huffmeta,            huffmeta_size,   byte_offset, true);
    DeserializeArray<size_t>(compressed_data_subarray, decodebook_size_ptr, 1,               byte_offset, false);
    DeserializeArray<Byte>  (compressed_data_subarray, decodebook,          decodebook_size, byte_offset, true);
    DeserializeArray<size_t>(compressed_data_subarray, total_uInts_ptr,     1,               byte_offset, false);
    DeserializeArray<H>     (compressed_data_subarray, huff,                total_uInts,     byte_offset, true);

    std::cout << "primary_count: " << primary_count << "\n";
    std::cout << "dict_size: " << dict_size << "\n";
    std::cout << "chunk_size: " << chunk_size << "\n";
    std::cout << "huffmeta_size: " << huffmeta_size << "\n";
    std::cout << "decodebook_size: " << decodebook_size << "\n";
    std::cout << "total_uInts: " << total_uInts << "\n";

    SubArray<1, size_t, DeviceType> huffmeta_subarray2({(SIZE)huffmeta_size}, huffmeta);
    SubArray<1, Byte, DeviceType> decodebook_subarray2({(SIZE)decodebook_size}, decodebook);
    SubArray<1, H, DeviceType> huff_subarray2({(SIZE)total_uInts}, huff);
    Array<1, Q, DeviceType> primary({(SIZE)primary_count});
    SubArray primary_subarray2(primary);

    // printf("huffmeta_subarray: ");
    // CompareSubArrays(huffmeta_subarray, huffmeta_subarray2);

    // printf("decodebook_subarray: ");
    // CompareSubArrays(decodebook_subarray, decodebook_subarray2);

    // printf("huff_subarray: ");
    // CompareSubArrays(huff_subarray3, huff_subarray2);

    DeviceRuntime<DeviceType>::SyncQueue(0);

    int nchunk = (primary_count - 1) / chunk_size + 1;
    Decode<Q, H, DeviceType>().Execute(huff_subarray2,
                                       huffmeta_subarray2,
                                       primary_subarray2,
                                       primary_count, chunk_size, nchunk,
                                       decodebook_subarray2, decodebook_size, 0);
    DeviceRuntime<DeviceType>::SyncQueue(0);

    printf("primary_subarray: ");
    CompareSubArrays(dprimary_subarray, primary_subarray2);
  }

  return compressed_data;
}

template <typename Q, typename H, typename DeviceType>
Array<1, Q, DeviceType>
HuffmanDecompress(Array<1, Byte, DeviceType> &compressed_data) {

  // printf("HuffmanDecompress\n");
  using Mem = MemoryManager<DeviceType>;

  size_t primary_count;
  int dict_size;
  int chunk_size;
  size_t huffmeta_size;
  size_t decodebook_size;
  size_t total_uInts;

  size_t * primary_count_ptr = &primary_count;
  int * dict_size_ptr = &dict_size;
  int * chunk_size_ptr = &chunk_size;
  size_t * huffmeta_size_ptr = &huffmeta_size;
  size_t * decodebook_size_ptr = &decodebook_size;
  size_t * total_uInts_ptr = &total_uInts;

  size_t *huffmeta;
  Byte *decodebook;
  H * huff;

  SubArray compressed_data_subarray(compressed_data);

  SIZE byte_offset = 0;
  DeserializeArray<size_t>(compressed_data_subarray, primary_count_ptr,   1,               byte_offset, false);
  DeserializeArray<int>   (compressed_data_subarray, dict_size_ptr,       1,               byte_offset, false);
  DeserializeArray<int>   (compressed_data_subarray, chunk_size_ptr,      1,               byte_offset, false);
  DeserializeArray<size_t>(compressed_data_subarray, huffmeta_size_ptr,   1,               byte_offset, false);
  DeserializeArray<size_t>(compressed_data_subarray, huffmeta,         huffmeta_size,   byte_offset, true);
  DeserializeArray<size_t>(compressed_data_subarray, decodebook_size_ptr, 1,               byte_offset, false);
  DeserializeArray<Byte>  (compressed_data_subarray, decodebook,       decodebook_size, byte_offset, true);
  DeserializeArray<size_t>(compressed_data_subarray, total_uInts_ptr,     1,               byte_offset, false);
  DeserializeArray<H>     (compressed_data_subarray, huff,             total_uInts,     byte_offset, true);

  // std::cout << "primary_count: " << primary_count << "\n";
  // std::cout << "dict_size: " << dict_size << "\n";
  // std::cout << "chunk_size: " << chunk_size << "\n";
  // std::cout << "huffmeta_size: " << huffmeta_size << "\n";
  // std::cout << "decodebook_size: " << decodebook_size << "\n";
  // std::cout << "total_uInts: " << total_uInts << "\n";

  SubArray<1, size_t, DeviceType> huffmeta_subarray({(SIZE)huffmeta_size}, huffmeta);
  SubArray<1, Byte, DeviceType> decodebook_subarray({(SIZE)decodebook_size}, decodebook);
  SubArray<1, H, DeviceType> huff_subarray({(SIZE)total_uInts}, huff);
  Array<1, Q, DeviceType> primary({(SIZE)primary_count});
  SubArray primary_subarray(primary);

  int nchunk = (primary_count - 1) / chunk_size + 1;
  Decode<Q, H, DeviceType>().Execute(huff_subarray,
                                     huffmeta_subarray,
                                     primary_subarray,
                                     primary_count, chunk_size, nchunk,
                                     decodebook_subarray, decodebook_size, 0);
  DeviceRuntime<DeviceType>::SyncQueue(0);

  // PrintSubarray("after huffman", SubArray<1, Q, DeviceType>({10}, primary.get_dv()));

  return primary;
}
}

#endif