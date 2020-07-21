#include "cuda/mgard_cuda.h"
#include "cuda/mgard_cuda_common.h"
#include "mgard.h"
#include "mgard_api.h"
#include <assert.h>
#include <iostream>

template <typename T>
unsigned char *mgard_compress_cuda(T *data, int &out_size, int n1, int n2,
                                   int n3, T tol)
// Perform compression preserving the tolerance in the L-infty norm
{
  assert(tol >= 1e-7);
  unsigned char *mgard_compressed_ptr = nullptr;
  if (n1 > 1 && n2 > 1 && n3 > 1) {
    assert(n1 > 3);
    assert(n2 > 3);
    assert(n3 > 3);

    mgard_cuda_handle<T> handle(n1, n2, n3);
    mgard_compressed_ptr =
        mgard_cuda::refactor_qz_cuda(handle, data, out_size, tol);
    return mgard_compressed_ptr;

  } else if (n1 > 1 && n2 > 1) {
    assert(n1 > 3);
    assert(n2 > 3);
    mgard_cuda_handle<T> handle(n1, n2, n3);
    mgard_compressed_ptr =
        mgard_cuda::refactor_qz_2D_cuda(handle, data, out_size, tol);
    return mgard_compressed_ptr;
  } else if (n1 > 1) {
    assert(n1 > 3);
    std::cerr << "MGARD: Not impemented!  Let us know if you need 1D "
                 "compression...\n";
    return nullptr;
  }

  return nullptr;
}

template unsigned char *mgard_compress_cuda<double>(double *data, int &out_size,
                                                    int n1, int n2, int n3,
                                                    double tol);
template unsigned char *mgard_compress_cuda<float>(float *data, int &out_size,
                                                   int n1, int n2, int n3,
                                                   float tol);

template <typename T>
T *mgard_decompress_cuda(unsigned char *data, int data_len, int n1, int n2,
                         int n3) {
  T *mgard_decompressed_ptr = nullptr;

  if (n1 > 1 && n2 > 1 && n3 > 1) {
    assert(n1 > 3);
    assert(n2 > 3);
    assert(n3 > 3);
    mgard_cuda_handle<T> handle(n1, n2, n3);
    mgard_decompressed_ptr =
        mgard_cuda::recompose_udq_cuda(handle, data, data_len);
    return mgard_decompressed_ptr;
  } else if (n1 > 1 && n2 > 1) {
    assert(n1 > 3);
    assert(n2 > 3);
    mgard_cuda_handle<T> handle(n1, n2, n3);
    mgard_decompressed_ptr =
        mgard_cuda::recompose_udq_2D_cuda(handle, data, data_len);
    return mgard_decompressed_ptr;
  } else if (n1 > 1) {
    assert(n1 > 3);
    std::cerr << "MGARD: Not impemented!  Let us know if you need 1D "
                 "compression...\n";
    // mgard_decompressed_ptr = mgard::recompose_udq_1D(n1,  data, data_len);
  }
  return nullptr;
}

template double *mgard_decompress_cuda<double>(unsigned char *data,
                                               int data_len, int n1, int n2,
                                               int n3);
template float *mgard_decompress_cuda<float>(unsigned char *data, int data_len,
                                             int n1, int n2, int n3);

template <typename T>
unsigned char *mgard_compress_cuda(T *data, int &out_size, int n1, int n2,
                                   int n3, std::vector<T> &coords_x,
                                   std::vector<T> &coords_y,
                                   std::vector<T> &coords_z, T tol)
// Perform compression preserving the tolerance in the L-infty norm
{
  assert(tol >= 1e-7);
  unsigned char *mgard_compressed_ptr = nullptr;
  if (n1 > 1 && n2 > 1 && n3 > 1) {
    assert(n1 > 3);
    assert(n2 > 3);
    assert(n3 > 3);

    mgard_cuda_handle<T> handle(n1, n2, n3, coords_x.data(), coords_y.data(),
                                coords_z.data());
    mgard_compressed_ptr =
        mgard_cuda::refactor_qz_cuda(handle, data, out_size, tol);
    return mgard_compressed_ptr;

  } else if (n1 > 1 && n2 > 1) {
    assert(n1 > 3);
    assert(n2 > 3);
    mgard_cuda_handle<T> handle(n1, n2, n3, coords_x.data(), coords_y.data(),
                                coords_z.data());
    mgard_compressed_ptr =
        mgard_cuda::refactor_qz_2D_cuda(handle, data, out_size, tol);
    return mgard_compressed_ptr;
  } else if (n1 > 1) {
    assert(n1 > 3);
    std::cerr << "MGARD: Not impemented!  Let us know if you need 1D "
                 "compression...\n";
    return nullptr;
  }

  return nullptr;
}

template unsigned char *
mgard_compress_cuda<double>(double *data, int &out_size, int n1, int n2, int n3,
                            std::vector<double> &coords_x,
                            std::vector<double> &coords_y,
                            std::vector<double> &coords_z, double tol);
template unsigned char *mgard_compress_cuda<float>(float *data, int &out_size,
                                                   int n1, int n2, int n3,
                                                   std::vector<float> &coords_x,
                                                   std::vector<float> &coords_y,
                                                   std::vector<float> &coords_z,
                                                   float tol);

template <typename T>
T *mgard_decompress_cuda(unsigned char *data, int data_len, int n1, int n2,
                         int n3, std::vector<T> &coords_x,
                         std::vector<T> &coords_y, std::vector<T> &coords_z) {
  T *mgard_decompressed_ptr = nullptr;

  if (n1 > 1 && n2 > 1 && n3 > 1) {
    assert(n1 > 3);
    assert(n2 > 3);
    assert(n3 > 3);
    mgard_cuda_handle<T> handle(n1, n2, n3, coords_x.data(), coords_y.data(),
                                coords_z.data());
    mgard_decompressed_ptr =
        mgard_cuda::recompose_udq_cuda(handle, data, data_len);
    return mgard_decompressed_ptr;
  } else if (n1 > 1 && n2 > 1) {
    assert(n1 > 3);
    assert(n2 > 3);
    mgard_cuda_handle<T> handle(n1, n2, n3, coords_x.data(), coords_y.data(),
                                coords_z.data());
    mgard_decompressed_ptr =
        mgard_cuda::recompose_udq_2D_cuda(handle, data, data_len);
    return mgard_decompressed_ptr;
  } else if (n1 > 1) {
    assert(n1 > 3);
    std::cerr << "MGARD: Not impemented!  Let us know if you need 1D "
                 "compression...\n";
    // mgard_decompressed_ptr = mgard::recompose_udq_1D(n1,  data, data_len);
  }
  return nullptr;
}

template double *mgard_decompress_cuda<double>(unsigned char *data,
                                               int data_len, int n1, int n2,
                                               int n3,
                                               std::vector<double> &coords_x,
                                               std::vector<double> &coords_y,
                                               std::vector<double> &coords_z);
template float *mgard_decompress_cuda<float>(unsigned char *data, int data_len,
                                             int n1, int n2, int n3,
                                             std::vector<float> &coords_x,
                                             std::vector<float> &coords_y,
                                             std::vector<float> &coords_z);

template <typename T>
unsigned char *mgard_compress_cuda(mgard_cuda_handle<T> &handle, T *v,
                                   int &out_size, T tol)
// Perform compression preserving the tolerance in the L-infty norm
{

  assert(tol >= 1e-7);
  unsigned char *mgard_compressed_ptr = nullptr;
  if (handle.nrow > 1 && handle.ncol > 1 && handle.nfib > 1) {
    assert(handle.nrow > 3);
    assert(handle.ncol > 3);
    assert(handle.n3 > 3);
    mgard_compressed_ptr =
        mgard_cuda::refactor_qz_cuda(handle, v, out_size, tol);
    return mgard_compressed_ptr;

  } else if (handle.nrow > 1 && handle.ncol > 1) {
    assert(handle.nrow > 3);
    assert(handle.ncol > 3);
    mgard_compressed_ptr =
        mgard_cuda::refactor_qz_2D_cuda(handle, v, out_size, tol);
    return mgard_compressed_ptr;
  } else if (handle.nrow > 1) {
    assert(handle.nrow > 3);
    std::cerr << "MGARD: Not impemented!  Let us know if you need 1D "
                 "compression...\n";
    return nullptr;
  }

  return nullptr;
}

template unsigned char *
mgard_compress_cuda<double>(mgard_cuda_handle<double> &handle, double *v,
                            int &out_size, double tol_in);
template unsigned char *
mgard_compress_cuda<float>(mgard_cuda_handle<float> &handle, float *v,
                           int &out_size, float tol_in);

template <typename T>
T *mgard_decompress_cuda(mgard_cuda_handle<T> &handle, unsigned char *data,
                         int data_len) {
  T *mgard_decompressed_ptr = nullptr;

  if (handle.nrow > 1 && handle.ncol > 1 && handle.nfib > 1) {
    assert(handle.nrow > 3);
    assert(handle.ncol > 3);
    assert(handle.nfib > 3);
    mgard_decompressed_ptr =
        mgard_cuda::recompose_udq_cuda(handle, data, data_len);
    return mgard_decompressed_ptr;
  } else if (handle.nrow > 1 && handle.ncol > 1) {
    assert(handle.nrow > 3);
    assert(handle.ncol > 3);
    mgard_decompressed_ptr =
        mgard_cuda::recompose_udq_2D_cuda(handle, data, data_len);
    return mgard_decompressed_ptr;
  } else if (handle.nrow > 1) {
    assert(handle.nrow > 3);
    std::cerr << "MGARD: Not impemented!  Let us know if you need 1D "
                 "compression...\n";
    // mgard_decompressed_ptr = mgard::recompose_udq_1D(nrow,  data, data_len);
  }
  return nullptr;
}

template double *mgard_decompress_cuda(mgard_cuda_handle<double> &handle,
                                       unsigned char *data, int data_len);
template float *mgard_decompress_cuda(mgard_cuda_handle<float> &handle,
                                      unsigned char *data, int data_len);
