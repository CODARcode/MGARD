#include <cuda_runtime.h>

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include <chrono>

using namespace std::chrono;

#include "cuda/Common.h"
#include "cuda/CommonInternal.h"

#include "cuda/ParallelHuffman/canonical.cuh"
#include "cuda/ParallelHuffman/cuda_error_handling.cuh"
#include "cuda/ParallelHuffman/cuda_mem.cuh"
#include "cuda/ParallelHuffman/dbg_gpu_printing.cuh"
#include "cuda/ParallelHuffman/format.hh"
#include "cuda/ParallelHuffman/histogram.cuh"
#include "cuda/ParallelHuffman/huffman.cuh"
#include "cuda/ParallelHuffman/huffman_codec.cuh"
#include "cuda/ParallelHuffman/huffman_workflow.cuh"
#include "cuda/ParallelHuffman/par_huffman.cuh"
#include "cuda/ParallelHuffman/types.hh"

#include "cuda/ParallelHuffman/Histogram.hpp"
#include "cuda/ParallelHuffman/GetCodebook.hpp"
#include "cuda/ParallelHuffman/EncodeFixedLen.hpp"
#include "cuda/ParallelHuffman/Deflate.hpp"
#include "cuda/ParallelHuffman/Decode.hpp"

int ht_state_num;
int ht_all_nodes;
using uint8__t = uint8_t;

template <typename Q>
void wrapper::GetFrequency(Q *d_bcode, size_t len, unsigned int *d_freq,
                           int dict_size) {
  // Parameters for thread and block count optimization

  // Initialize to device-specific values
  int deviceId;
  int maxbytes;
  int maxbytesOptIn;
  int numSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&maxbytes, cudaDevAttrMaxSharedMemoryPerBlock,
                         deviceId);
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId);

  // Account for opt-in extra shared memory on certain architectures
  cudaDeviceGetAttribute(&maxbytesOptIn,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);
  maxbytes = std::max(maxbytes, maxbytesOptIn);

  // Optimize launch
  int numBuckets = dict_size;
  int numValues = len;
  int itemsPerThread = 1;
  int RPerBlock = (maxbytes / (int)sizeof(int)) / (numBuckets + 1);
  int numBlocks = numSMs;
  cudaFuncSetAttribute(p2013Histogram<Q, unsigned int>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

  // printf("dict_size: %u, RPerBlock: %d\n", dict_size, RPerBlock);

  // fits to size
  int threadsPerBlock =
      ((((numValues / (numBlocks * itemsPerThread)) + 1) / 64) + 1) * 64;

  while (threadsPerBlock > 1024) {
    if (RPerBlock <= 1) {
      threadsPerBlock = 1024;
    } else {
      RPerBlock /= 2;
      numBlocks *= 2;
      threadsPerBlock =
          ((((numValues / (numBlocks * itemsPerThread)) + 1) / 64) + 1) * 64;
    }
  }

  // mgard_cuda::print_matrix_cuda(1, 10, (int *)d_bcode, 10);

  // printf("maxbytes: %d, p2013Histogram: %d\n", maxbytes,(numBuckets + 1) *
  // sizeof(int));

  // printf("numBlocks: %d, threadsPerBlock: %d, sm: %d\n", numBlocks, threadsPerBlock, ((numBuckets + 1) * RPerBlock) * sizeof(int));
  p2013Histogram //
      <<<numBlocks, threadsPerBlock,
         ((numBuckets + 1) * RPerBlock) * sizeof(int)>>> //
      (d_bcode, d_freq, numValues, numBuckets, RPerBlock);
  cudaDeviceSynchronize();

  // TODO make entropy optional
  // {
  //     auto   freq    = mem::CreateHostSpaceAndMemcpyFromDevice(d_freq,
  //     dict_size); double entropy = 0.0; for (auto i = 0; i < dict_size; i++)
  //         if (freq[i]) {
  //             auto possibility = freq[i] / (1.0 * len);
  //             entropy -= possibility * log(possibility);
  //             cout << i << ": " << freq[i] << "\n";
  //         }
  //     cout << log_info << "entropy:\t\t" << entropy << endl;
  //     delete[] freq;
  // }

  // #ifdef DEBUG_PRINT
  //     print_histogram<unsigned int><<<1, 32>>>(d_freq, dict_size, dict_size /
  //     2); cudaDeviceSynchronize();
  // #endif
}

template <typename H>
void PrintChunkHuffmanCoding(size_t *dH_bit_meta, //
                             size_t *dH_uInt_meta, size_t len, int chunk_size,
                             size_t total_bits, size_t total_uInts) {
  cout << "\n" << log_dbg << "Huffman coding detail start ------" << endl;
  printf("| %s\t%s\t%s\t%s\t%9s\n", "chunk", "bits", "bytes", "uInt",
         "chunkCR");
  for (size_t i = 0; i < 8; i++) {
    size_t n_byte = (dH_bit_meta[i] - 1) / 8 + 1;
    auto chunk_CR = ((double)chunk_size * sizeof(float) /
                     (1.0 * (double)dH_uInt_meta[i] * sizeof(H)));
    printf("| %lu\t%lu\t%lu\t%lu\t%9.6lf\n", i, dH_bit_meta[i], n_byte,
           dH_uInt_meta[i], chunk_CR);
  }
  cout << "| ..." << endl
       << "| Huff.total.bits:\t" << total_bits << endl
       << "| Huff.total.bytes:\t" << total_uInts * sizeof(H) << endl
       << "| Huff.CR (uInt):\t"
       << (double)len * sizeof(float) / (total_uInts * 1.0 * sizeof(H)) << endl;
  cout << log_dbg << "coding detail end ----------------" << endl;
  cout << endl;
}

// template <mgard_cuda::DIM D, typename T, typename S, typename Q, typename H, typename DeviceType>
// void HuffmanEncode(mgard_cuda::Handle<D, T> &handle, S *dqv, size_t n,
//                    std::vector<size_t> &outlier_idx, H *&dmeta,
//                    size_t &dmeta_size, H *&ddata, size_t &ddata_size,
//                    int chunk_size, int dict_size) {

template <mgard_cuda::DIM D, typename T, typename S, typename Q, typename H, typename DeviceType>
void HuffmanEncode(mgard_cuda::Handle<D, T> &handle, S *dqv, size_t n,
                   std::vector<size_t> &outlier_idx, H *&dmeta,
                   size_t &dmeta_size, H *&ddata, size_t &ddata_size,
                   int chunk_size, int dict_size) {

  high_resolution_clock::time_point t1, t2, start, end;
  duration<double> time_span;

  int queue_idx = 0;

  Q *dprimary = (Q *)dqv;
  size_t primary_count = n;

  t1 = high_resolution_clock::now();

  ht_state_num = 2 * dict_size;
  ht_all_nodes = 2 * ht_state_num;

  mgard_cuda::Array<1, unsigned int, DeviceType> freq_array({(mgard_cuda::SIZE)ht_all_nodes});
  freq_array.memset(0);

  mgard_cuda::SubArray<1, Q, DeviceType> dprimary_subarray({(mgard_cuda::SIZE)n}, dprimary);
  mgard_cuda::SubArray<1, unsigned int, DeviceType> freq_subarray(freq_array);
  mgard_cuda::Histogram<Q, unsigned int, DeviceType>().Execute(dprimary_subarray, freq_subarray, primary_count, dict_size, 0);
  gpuErrchk(cudaDeviceSynchronize());

  auto type_bw = sizeof(H) * 8;
  size_t decodebook_size = sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size;
  mgard_cuda::Array<1, H, DeviceType> codebook_array({(mgard_cuda::SIZE)dict_size});
  codebook_array.memset(0);
  mgard_cuda::Array<1, uint8_t, DeviceType> decodebook_array({(mgard_cuda::SIZE)decodebook_size});
  codebook_array.memset(0xff);

  H * codebook = codebook_array.get_dv();
  uint8_t *decodebook = decodebook_array.get_dv();


  mgard_cuda::SubArray<1, H, DeviceType> codebook_subarray(codebook_array);
  mgard_cuda::SubArray<1, uint8_t, DeviceType> decodebook_subarray(decodebook_array);

  mgard_cuda::GetCodebook<Q, H, DeviceType>(dict_size, freq_subarray, codebook_subarray, decodebook_subarray);
  cudaDeviceSynchronize();

  mgard_cuda::Array<1, H, DeviceType> huff_array({(mgard_cuda::SIZE)primary_count});
  huff_array.memset(0);
  H * huff = huff_array.get_dv();

  gpuErrchk(cudaDeviceSynchronize());

  mgard_cuda::SubArray<1, H, DeviceType> huff_subarray(huff_array);
  mgard_cuda::EncodeFixedLen<unsigned int, H, DeviceType>().Execute(dprimary_subarray,
                                                              huff_subarray,
                                                              primary_count,
                                                              codebook_subarray, 0);

  // deflate
  auto nchunk = (primary_count - 1) / chunk_size + 1; 
  mgard_cuda::Array<1, size_t, DeviceType> huff_bitwidths_array({(mgard_cuda::SIZE)nchunk});
  huff_bitwidths_array.memset(0);
  size_t * huff_bitwidths = huff_bitwidths_array.get_dv();

  mgard_cuda::SubArray<1, size_t, DeviceType> huff_bitwidths_subarray({(mgard_cuda::SIZE)nchunk}, huff_bitwidths);
  mgard_cuda::Deflate<H, DeviceType>().Execute(huff_subarray, primary_count, huff_bitwidths_subarray, chunk_size, 0);

  mgard_cuda::DeviceRuntime<DeviceType>::SyncQueue(0);


  // dump TODO change to int
  size_t* h_meta = new size_t[nchunk * 3]();
  size_t* dH_uInt_meta = h_meta;
  size_t* dH_bit_meta = h_meta + nchunk;
  size_t* dH_uInt_entry = h_meta + nchunk * 2;

  mgard_cuda::MemoryManager<DeviceType>().Copy1D(dH_bit_meta, huff_bitwidths, nchunk, 0);
  gpuErrchk(cudaDeviceSynchronize());
  // transform in uInt
  memcpy(dH_uInt_meta, dH_bit_meta, nchunk * sizeof(size_t));
  for_each(dH_uInt_meta, dH_uInt_meta + nchunk,
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

  gpuErrchk(cudaDeviceSynchronize());
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("huffman encode time: %.6f s\n", time_span.count());

  // out_meta: |outlier count|outlier idx|outlier data|primary count|dict
  // size|chunk size|huffmeta size|huffmeta|decodebook size|decodebook|
  // out_data: |huffman data|

  t1 = high_resolution_clock::now();
  dmeta_size = // sizeof(size_t) + outlier_count * sizeof(size_t) +
               // outlier_count * sizeof(S) + //outlier
      sizeof(size_t) + sizeof(int) + sizeof(int) + // primary
      sizeof(size_t) + 2 * nchunk * sizeof(size_t) + sizeof(size_t) +
      (sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size) * sizeof(uint8_t);


  mgard_cuda::cudaMallocHelper(handle, (void **)&dmeta, dmeta_size);
  ddata_size = total_uInts * sizeof(H);
  mgard_cuda::cudaMallocHelper(handle, (void **)&ddata, ddata_size);

  mgard_cuda::Array<1, mgard_cuda::Byte, DeviceType> meta_array({(mgard_cuda::SIZE)dmeta_size});
  mgard_cuda::Array<1, mgard_cuda::Byte, DeviceType> data_array({(mgard_cuda::SIZE)ddata_size});

  void *dmeta_p = (void *)dmeta;
  mgard_cuda::cudaMemcpyAsyncHelper(handle, dmeta_p, &primary_count,
                                    sizeof(size_t), mgard_cuda::H2D,
                                    (queue_idx++) % handle.num_of_queues);
  dmeta_p = dmeta_p + sizeof(size_t);
  mgard_cuda::cudaMemcpyAsyncHelper(handle, dmeta_p, &dict_size, sizeof(int),
                                    mgard_cuda::H2D,
                                    (queue_idx++) % handle.num_of_queues);
  dmeta_p = dmeta_p + sizeof(int);
  mgard_cuda::cudaMemcpyAsyncHelper(handle, dmeta_p, &chunk_size, sizeof(int),
                                    mgard_cuda::H2D,
                                    (queue_idx++) % handle.num_of_queues);
  dmeta_p = dmeta_p + sizeof(int);
  size_t huffmeta_size = 2 * nchunk * sizeof(size_t);
  // printf("compress huffmeta_size: %llu\n", huffmeta_size);
  mgard_cuda::cudaMemcpyAsyncHelper(handle, dmeta_p, &huffmeta_size,
                                    sizeof(size_t), mgard_cuda::H2D,
                                    (queue_idx++) % handle.num_of_queues);
  dmeta_p = dmeta_p + sizeof(size_t);
  mgard_cuda::cudaMemcpyAsyncHelper(handle, dmeta_p, h_meta + nchunk,
                                    huffmeta_size, mgard_cuda::H2D,
                                    (queue_idx++) % handle.num_of_queues);
  dmeta_p = dmeta_p + huffmeta_size;
  mgard_cuda::cudaMemcpyAsyncHelper(handle, dmeta_p, &decodebook_size,
                                    sizeof(size_t), mgard_cuda::H2D,
                                    (queue_idx++) % handle.num_of_queues);
  dmeta_p = dmeta_p + sizeof(size_t);
  // printf("compress decodebook_size: %llu\n", decodebook_size);
  mgard_cuda::cudaMemcpyAsyncHelper(handle, dmeta_p, decodebook,
                                    decodebook_size, mgard_cuda::H2D,
                                    (queue_idx++) % handle.num_of_queues);
  dmeta_p = dmeta_p + decodebook_size;

  gpuErrchk(cudaDeviceSynchronize());
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("serilization time1: %.6f s\n", time_span.count());

  t1 = high_resolution_clock::now();

  for (auto i = 0; i < nchunk; i++) {
    mgard_cuda::cudaMemcpyAsyncHelper(
        handle, ddata + dH_uInt_entry[i], (void *)(huff + i * chunk_size),
        dH_uInt_meta[i] * sizeof(H), mgard_cuda::D2D,
        (queue_idx++) % handle.num_of_queues);
  }

  gpuErrchk(cudaDeviceSynchronize());
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  // printf("serilization time2: %.6f s\n", time_span.count());

  //////// clean up
  // cudaFreeHost(flags);
  // cudaFree(doutlier);
  // cudaFree(dprimary);
  // cudaFree(freq);
  // cudaFree(codebook);
  // cudaFree(decodebook);
  // cudaFree(huff);
  // cudaFree(huff_bitwidths);
  delete[] h_meta;
}

template <mgard_cuda::DIM D, typename T, typename S, typename Q, typename H, typename DeviceType>
void HuffmanDecode(mgard_cuda::Handle<D, T> &handle, S *&dqv, size_t &n,
                   H *dmeta, size_t dmeta_size, H *ddata, size_t ddata_size) {

  Q *dprimary;
  S *doutlier;
  size_t primary_count;
  size_t outlier_count;
  size_t *outlier_idx;
  size_t huffmeta_size;

  int dict_size;
  int chunk_size;
  size_t *huffmeta;
  uint8_t *decodebook;
  size_t decodebook_size;

  void *dmeta_p = (void *)dmeta;

  // primary
  mgard_cuda::cudaMemcpyAsyncHelper(handle, &primary_count, dmeta_p,
                                    sizeof(size_t), mgard_cuda::D2H, 0);
  dmeta_p = dmeta_p + sizeof(size_t);
  // printf("decompress primary_count: %llu\n", primary_count);
  mgard_cuda::cudaMallocHelper(handle, (void **)&dprimary, primary_count * sizeof(Q));

  mgard_cuda::cudaMemcpyAsyncHelper(handle, &dict_size, dmeta_p, sizeof(int),
                                    mgard_cuda::D2H, 0);
  dmeta_p = dmeta_p + sizeof(int);
  mgard_cuda::cudaMemcpyAsyncHelper(handle, &chunk_size, dmeta_p, sizeof(int),
                                    mgard_cuda::D2H, 0);
  dmeta_p = dmeta_p + sizeof(int);
  mgard_cuda::cudaMemcpyAsyncHelper(handle, &huffmeta_size, dmeta_p,
                                    sizeof(size_t), mgard_cuda::D2H, 0);
  dmeta_p = dmeta_p + sizeof(size_t);
  // printf("decompress huffmeta_size: %llu\n", huffmeta_size);
  mgard_cuda::cudaMallocHelper(handle, (void **)&huffmeta, huffmeta_size);
  mgard_cuda::cudaMemcpyAsyncHelper(handle, huffmeta, dmeta_p, huffmeta_size,
                                    mgard_cuda::D2D, 0);
  // // huffmeta = (size_t *)dmeta_p;
  dmeta_p = dmeta_p + huffmeta_size;
  mgard_cuda::cudaMemcpyAsyncHelper(handle, &decodebook_size, dmeta_p,
                                    sizeof(size_t), mgard_cuda::D2H, 0);
  dmeta_p = dmeta_p + sizeof(size_t);
  // printf("decompress decodebook_size: %llu\n", decodebook_size);
  mgard_cuda::cudaMallocHelper(handle, (void **)&decodebook, decodebook_size);
  mgard_cuda::cudaMemcpyAsyncHelper(handle, decodebook, dmeta_p,
                                    decodebook_size, mgard_cuda::D2D, 0);
  // // decodebook = (uint8_t *)dmeta_p;
  dmeta_p = dmeta_p + decodebook_size;

  // printf("start decoding\n");
  int nchunk = (primary_count - 1) / chunk_size + 1;
  auto blockDim = tBLK_DEFLATE; // the same as deflating
  auto gridDim = (nchunk - 1) / blockDim + 1;

  // Decode<<<gridDim, blockDim, decodebook_size>>>( //
  //     ddata, huffmeta, dprimary, primary_count, chunk_size, nchunk,
      // (uint8_t *)decodebook, (size_t)decodebook_size);
  cudaDeviceSynchronize();

  mgard_cuda::SubArray<1, H, mgard_cuda::CUDA> ddata_subarray({(mgard_cuda::SIZE)ddata_size}, ddata);
  mgard_cuda::SubArray<1, size_t, mgard_cuda::CUDA> huffmeta_subarray({(mgard_cuda::SIZE)huffmeta_size/(mgard_cuda::SIZE)sizeof(size_t)}, huffmeta);
  mgard_cuda::SubArray<1, Q, mgard_cuda::CUDA> dprimary_subarray({(mgard_cuda::SIZE)primary_count}, dprimary);
  mgard_cuda::SubArray<1, uint8_t, mgard_cuda::CUDA> decodebook_subarray({(mgard_cuda::SIZE)decodebook_size}, decodebook);
  mgard_cuda::Decode<Q, H, mgard_cuda::CUDA>().Execute(ddata_subarray,
                                                       huffmeta_subarray,
                                                       dprimary_subarray,
                                                       primary_count, chunk_size, nchunk,
                                                       decodebook_subarray, decodebook_size, 0);
  cudaDeviceSynchronize();
  dqv = (S *)dprimary;
  n = primary_count;
}

template void wrapper::GetFrequency<uint8__t>(uint8__t *, size_t,
                                              unsigned int *, int);
template void wrapper::GetFrequency<uint16_t>(uint16_t *, size_t,
                                              unsigned int *, int);
template void wrapper::GetFrequency<uint32_t>(uint32_t *, size_t,
                                              unsigned int *, int);

template void PrintChunkHuffmanCoding<uint32_t>(size_t *, size_t *, size_t, int,
                                                size_t, size_t);
template void PrintChunkHuffmanCoding<uint64_t>(size_t *, size_t *, size_t, int,
                                                size_t, size_t);

// template tuple3ul HuffmanEncode<uint8__t, uint32_t, float>(Handle<uint8__t>
// &, string&, uint8__t*, size_t, void * &, size_t &, int, int); template
// tuple3ul HuffmanEncode<uint16_t, uint32_t, float>(Handle<uint16_t> &,
// string&, uint16_t*, size_t, void * &, size_t &, int, int); template tuple3ul
// HuffmanEncode<uint32_t, uint32_t, float>(Handle<uint32_t> &, string&,
// uint32_t*, size_t, void * &, size_t &, int, int); template tuple3ul
// HuffmanEncode<uint8__t, uint64_t, float>(Handle<uint8__t> &, string&,
// uint8__t*, size_t, void * &, size_t &, int, int); template tuple3ul
// HuffmanEncode<uint16_t, uint64_t, float>(Handle<uint16_t> &, string&,
// uint16_t*, size_t, void * &, size_t &, int, int);

// template uint8__t* HuffmanDecode<uint8__t, uint32_t, float>(std::string&,
// void * d_in, size_t, int, int, int); template uint16_t*
// HuffmanDecode<uint16_t, uint32_t, float>(std::string&, void * d_in, size_t,
// int, int, int); template uint32_t* HuffmanDecode<uint32_t, uint32_t,
// float>(std::string&, void * d_in, size_t, int, int, int); template uint8__t*
// HuffmanDecode<uint8__t, uint64_t, float>(std::string&, void * d_in, size_t,
// int, int, int); template uint16_t* HuffmanDecode<uint16_t, uint64_t,
// float>(std::string&, void * d_in, size_t, int, int, int); template uint32_t*
// HuffmanDecode<uint32_t, uint64_t, float>(std::string&, void * d_in, size_t,
// int, int, int);

// template void HuffmanEncode<double, int, uint32_t>(Handle<double> &handle,
//     int* dqv, size_t n, bool * dflags, uint32_t * &dmeta, size_t &dmeta_size,
//     uint32_t * &ddata, size_t &ddata_size, int chunk_size, int dict_size);
// template void HuffmanEncode<float, int, uint32_t>(Handle<float> &handle,
//     int* dqv, size_t n, bool * dflags, uint32_t * &dmeta, size_t &dmeta_size,
//     uint32_t * &ddata, size_t &ddata_size, int chunk_size, int dict_size);

// template void HuffmanDecode<float, int, uint32_t>(Handle<float> &handle,
//                           int* &dqv, size_t &n, uint32_t * dmeta, size_t
//                           dmeta_size, uint32_t * ddata, size_t ddata_size);
// template void HuffmanDecode<double, int, uint32_t>(Handle<double> &handle,
//                           int* &dqv, size_t &n, uint32_t * dmeta, size_t
//                           dmeta_size, uint32_t * ddata, size_t ddata_size);

#define KERNELS(D, T, S, Q, H)                                                 \
  template void HuffmanEncode<D, T, S, Q, H, mgard_cuda::CUDA>(                                  \
      mgard_cuda::Handle<D, T> & handle, S * dqv, size_t n,                    \
      std::vector<size_t> & outlier_idx, H * &dmeta, size_t & dmeta_size,      \
      H * &ddata, size_t & ddata_size, int chunk_size, int dict_size);         \
  template void HuffmanDecode<D, T, S, Q, H, mgard_cuda::CUDA>(                                  \
      mgard_cuda::Handle<D, T> & handle, S * &dqv, size_t & n, H * dmeta,      \
      size_t dmeta_size, H * ddata, size_t ddata_size);

KERNELS(1, double, int, uint32_t, uint32_t)
KERNELS(1, float, int, uint32_t, uint32_t)
KERNELS(2, double, int, uint32_t, uint32_t)
KERNELS(2, float, int, uint32_t, uint32_t)
KERNELS(3, double, int, uint32_t, uint32_t)
KERNELS(3, float, int, uint32_t, uint32_t)
KERNELS(4, double, int, uint32_t, uint32_t)
KERNELS(4, float, int, uint32_t, uint32_t)
KERNELS(5, double, int, uint32_t, uint32_t)
KERNELS(5, float, int, uint32_t, uint32_t)
KERNELS(1, double, int, uint32_t, uint64_t)
KERNELS(1, float, int, uint32_t, uint64_t)
KERNELS(2, double, int, uint32_t, uint64_t)
KERNELS(2, float, int, uint32_t, uint64_t)
KERNELS(3, double, int, uint32_t, uint64_t)
KERNELS(3, float, int, uint32_t, uint64_t)
KERNELS(4, double, int, uint32_t, uint64_t)
KERNELS(4, float, int, uint32_t, uint64_t)
KERNELS(5, double, int, uint32_t, uint64_t)
KERNELS(5, float, int, uint32_t, uint64_t)

// clang-format off
