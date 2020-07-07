#include "cuda/mgard_cuda.h"
#include "cuda/mgard_cuda_common.h"
#include "cuda/mgard_cuda_helper.h"

#ifndef MGARD_API_CUDA_H
#define MGARD_API_CUDA_H

template <typename T>
unsigned char *mgard_compress_cuda(T * data, int &out_size, int n1, int n2, int n3, T tol);

template <typename T>
T *mgard_decompress_cuda(unsigned char * data, int data_len, int n1, int n2, int n3);

template <typename T>
unsigned char *mgard_compress_cuda(T * data, int &out_size, int n1, int n2, int n3, 
                                   std::vector<T> &coords_x,
                                   std::vector<T> &coords_y,
                                   std::vector<T> &coords_z,
                                   T tol);

template <typename T>
T *mgard_decompress_cuda(unsigned char * data, int data_len, int n1, int n2, int n3,
                         std::vector<T> &coords_x,
                         std::vector<T> &coords_y,
                         std::vector<T> &coords_z);

template <typename T>
unsigned char *mgard_compress_cuda(mgard_cuda_handle<T> & handle, T * v, int &out_size, T tol);

template <typename T>
T *mgard_decompress_cuda(mgard_cuda_handle<T> & handle, unsigned char *data, int data_len);

#endif