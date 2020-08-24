// Copyright 2020, Oak Ridge National Laboratory.
// MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by NVIDIA GPUs
// Authors: Jieyang Chen
// Corresponding Author: Jieyang Chen
//
// version: 0.0.0.1
// See LICENSE for details.

#include "cuda/mgard_cuda.h"
#include "cuda/mgard_cuda_common.h"
#include "cuda/mgard_cuda_helper.h"

#ifndef MGARD_API_CUDA_H
#define MGARD_API_CUDA_H
//!\file
//!\brief Compression and decompression API.

//! Compress a function on an equispaced 3D tensor product grid while
//! controlling the error as measured in the \f$ L^{\infty} \f$ norm.
//!
//!\param[in] data Dataset to be compressed.
//!\param[out] out_size Size in bytes of the compressed dataset.
//!\param[in] n1 Size of the dataset in the first dimension.
//!\param[in] n2 Size of the dataset in the second dimension.
//!\param[in] n3 Size of the dataset in the third dimension.
//!\param[in] tol Relative error tolerance.
//!
//!\return Compressed dataset.
template <typename T>
unsigned char *mgard_compress_cuda(T *data, int &out_size, int n1, int n2,
                                   int n3, T tol);

//! Decompress a function on an equispaced 3D tensor product grid which was
//! compressed while controlling the error as measured in the \f$ L^{\infty} \f$
//! norm.
//!
//!\param[in] data Compressed dataset.
//!\param[in] data_len Size in bytes of the compressed dataset.
//!\param[in] n1 Size of the dataset in the first dimension.
//!\param[in] n2 Size of the dataset in the second dimension.
//!\param[in] n3 Size of the dataset in the third dimension.
//!
//!\return Decompressed dataset.
template <typename T>
T *mgard_decompress_cuda(unsigned char *data, int data_len, int n1, int n2,
                         int n3);

//! Compress a function on an equispaced 3D tensor product grid while
//! controlling the error as measured in the \f$ L^{\infty} \f$ norm.
//!
//!\param[in] data Dataset to be compressed.
//!\param[out] out_size Size in bytes of the compressed dataset.
//!\param[in] n1 Size of the dataset in the first dimension.
//!\param[in] n2 Size of the dataset in the second dimension.
//!\param[in] n3 Size of the dataset in the third dimension.
//!\param[in] coords_x First coordinates of the nodes of the grid.
//!\param[in] coords_y Second coordinates of the nodes of the grid.
//!\param[in] coords_z Third coordinates of the nodes of the grid.
//!\param[in] tol Relative error tolerance.
//!
//!\return Compressed dataset.
template <typename T>
unsigned char *mgard_compress_cuda(T *data, int &out_size, int n1, int n2,
                                   int n3, std::vector<T> &coords_x,
                                   std::vector<T> &coords_y,
                                   std::vector<T> &coords_z, T tol);

//! Decompress a function on an equispaced 3D tensor product grid which was
//! compressed while controlling the error as measured in the \f$ L^{\infty} \f$
//! norm.
//!
//!\param[in] data Compressed dataset.
//!\param[in] data_len Size in bytes of the compressed dataset.
//!\param[in] n1 Size of the dataset in the first dimension.
//!\param[in] n2 Size of the dataset in the second dimension.
//!\param[in] n3 Size of the dataset in the third dimension.
//!\param[in] coords_x First coordinates of the nodes of the grid.
//!\param[in] coords_y Second coordinates of the nodes of the grid.
//!\param[in] coords_z Third coordinates of the nodes of the grid.
//!
//!\return Decompressed dataset.
template <typename T>
T *mgard_decompress_cuda(unsigned char *data, int data_len, int n1, int n2,
                         int n3, std::vector<T> &coords_x,
                         std::vector<T> &coords_y, std::vector<T> &coords_z);

//!\file
//!\brief Compression and decompression API.

//! Compress a function on an equispaced 3D tensor product grid while
//! controlling the error as measured in the \f$ L^{\infty} \f$ norm.
//!
//!\param[in] handle mgard_cuda_handle type for storing precomputed variable to
//! help speedup compression. \param[in] data Dataset to be compressed.
//!\param[out] out_size Size in bytes of the compressed dataset.
//!\param[in] tol Relative error tolerance.
//!
//!\return Compressed dataset.
template <typename T>
unsigned char *mgard_compress_cuda(mgard_cuda_handle<T> &handle, T *v,
                                   int &out_size, T tol);

//! Decompress a function on an equispaced 3D tensor product grid which was
//! compressed while controlling the error as measured in the \f$ L^{\infty} \f$
//! norm.
//!
//!\param[in] handle mgard_cuda_handle type for storing precomputed variable to
//! help speedup decompression. \param[in] data Compressed dataset. \param[in]
//! data_len Size in bytes of the compressed dataset.
//!
//!\return Decompressed dataset.
template <typename T>
T *mgard_decompress_cuda(mgard_cuda_handle<T> &handle, unsigned char *data,
                         int data_len);

#endif