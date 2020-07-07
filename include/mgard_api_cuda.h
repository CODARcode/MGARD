#include "cuda/mgard_cuda.h"
#include "cuda/mgard_cuda_common.h"
#include "cuda/mgard_cuda_helper.h"

#ifndef MGARD_API_CUDA_H
#define MGARD_API_CUDA_H

template <typename T>
unsigned char *mgard_compress_cuda(int itype_flag,  T  *v, int &out_size, int nrow, int ncol, int nfib, T tol_in);

template <typename T>
T *mgard_decompress_cuda(int itype_flag,  T& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib);

template <typename T>
unsigned char *mgard_compress_cuda(mgard_cuda_handle<T> & handle, int itype_flag, T * v, int &out_size, T tol_in);

template <typename T>
T *mgard_decompress_cuda(mgard_cuda_handle<T> & handle, int itype_flag,  T& quantizer, unsigned char *data, int data_len);

#endif