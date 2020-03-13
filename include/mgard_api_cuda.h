#include "mgard_cuda.h"

#ifndef MGARD_API_CUDA_H
#define MGARD_API_CUDA_H

template <typename T>
unsigned char *mgard_compress_cuda(int itype_flag,  T  *v, int &out_size, int nrow, int ncol, int nfib, T tol_in);
template <typename T>
T *mgard_decompress_cuda(int itype_flag,  T& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib);

template <typename T>
unsigned char *mgard_compress_cuda(int itype_flag,  T  *v, int &out_size, int nrow, int ncol, int nfib, T tol_in, int opt, int B, bool profile);
template <typename T>
T *mgard_decompress_cuda(int itype_flag,  T& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib, int opt, int B, bool profile);

#endif