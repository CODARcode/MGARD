#ifndef HUFFMAN_WORKFLOW
#define HUFFMAN_WORKFLOW

#include <cuda_runtime.h>
//#include <sys/stat.h>

#include <cstdint>
#include <string>
#include <tuple>

#include "cuda/mgard_cuda_common_internal.h"

using std::string;

// const int GB_unit = 1073741824;  // 1024^3

const int tBLK_ENCODE = 256;
const int tBLK_DEFLATE = 128;
const int tBLK_CANONICAL = 128;

// https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
// inline bool exists_test2(const std::string& name) {
//    return (access(name.c_str(), F_OK) != -1);
//}

namespace wrapper {

template <typename Q>
void GetFrequency(Q *d_bcode, size_t len, unsigned int *d_freq, int dict_size);

template <typename H>
void SetUpHuffmanTree(unsigned int *d_freq, H *d_codebook, int dict_size);

template <typename Q, typename H>
void MakeCanonical(H *d_plain_cb, uint8_t *d_singleton, size_t total_bytes,
                   int dict_size);

template <typename Q, typename H>
void EncodeByMemcpy(Q *d_bcode, size_t len, H *d_hcode, H *d_canonical_cb);

template <typename H>
void Deflate(H *d_hcode, size_t len, int chunk_size, int n_chunk,
             size_t *d_dH_bit_meta);

} // namespace wrapper

template <typename H>
void PrintChunkHuffmanCoding(size_t *dH_bit_meta, size_t *dH_uInt_meta,
                             size_t len, int chunk_size, size_t total_bits,
                             size_t total_uInts);

typedef std::tuple<size_t, size_t, size_t> tuple3ul;

template <typename T, int D, typename S, typename Q, typename H>
void HuffmanEncode(mgard_cuda_handle<T, D> &handle, S *dqv, size_t n,
                   std::vector<size_t> &outlier_idx, H *&dmeta,
                   size_t &dmeta_size, H *&ddata, size_t &ddata_size,
                   int chunk_size, int dict_size);

template <typename T, int D, typename S, typename Q, typename H>
void HuffmanDecode(mgard_cuda_handle<T, D> &handle, S *&dqv, size_t &n,
                   H *dmeta, size_t dmeta_size, H *ddata, size_t ddata_size);

#endif