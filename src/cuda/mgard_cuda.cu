#include "cuda/mgard_cuda_common.h"
#include "cuda/mgard_cuda_common_internal.h"
#include "cuda/mgard_cuda.h"
#include "cuda/mgard_cuda_kernels.h"
#include "cuda/mgard_cuda_prep_2d.h"
#include "cuda/mgard_cuda_refactor_2d.h"
#include "cuda/mgard_cuda_recompose_2d.h"
#include "cuda/mgard_cuda_postp_2d.h"
#include "cuda/mgard_cuda_prep_3d.h"
#include "cuda/mgard_cuda_refactor_3d.h"
#include "cuda/mgard_cuda_recompose_3d.h"
#include "cuda/mgard_cuda_postp_3d.h"

#include "mgard_compress.hpp"
#include "mgard_nuni.h"
#include "mgard.h"
#include <iomanip> 
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>

namespace mgard_cuda
{

template <typename T>
unsigned char *
refactor_qz_cuda(mgard_cuda_handle<T> & handle, T *u, int &outsize, T tol) {
 
  tol /= handle.l_target + 2;
  int size_ratio = sizeof(T) / sizeof(int);
  T * dv;
  size_t dv_pitch;
  cudaMalloc3DHelper((void**)&dv, &dv_pitch, handle.nfib * sizeof(T), handle.ncol, handle.nrow);
  int lddv1 = dv_pitch / sizeof(T);
  int lddv2 = handle.ncol;
  int * dqv;
  cudaMallocHelper((void**)&dqv, (handle.nrow * handle.ncol * handle.nfib + size_ratio) * sizeof(int));

  std::vector<int> qv(handle.nrow * handle.ncol * handle.nfib + size_ratio);
  std::vector<unsigned char> out_data;

  cudaMemcpy3DAsyncHelper(handle, dv, lddv1 * sizeof(T), handle.nfib * sizeof(T), handle.ncol,
                   u, handle.nfib  * sizeof(T), handle.nfib * sizeof(T), handle.ncol,
                   handle.nfib * sizeof(T), handle.ncol, handle.nrow,
                   H2D, 0);

  if (!is_2kplus1_cuda (handle.nrow) || 
      !is_2kplus1_cuda (handle.ncol) || 
      !is_2kplus1_cuda (handle.nfib)) {
    prep_3D_cuda_cpt(handle, dv, lddv1, lddv2);
  }

    // print_matrix_cuda(handle.nrow, handle.ncol, handle.nfib, dv, lddv1, lddv2, handle.nfib);

  refactor_3D_cuda_cpt(handle, dv, lddv1, lddv2);

  // print_matrix_cuda(handle.nrow, handle.ncol, handle.nfib, dv, lddv1, lddv2, handle.nfib);

  T norm = max_norm_cuda(u, handle.nrow * handle.ncol * handle.nfib);
  
  linear_quantize(handle, handle.nrow, handle.ncol, handle.nfib, norm, tol, dv, lddv1, lddv2, dqv, handle.nfib, handle.ncol, 0);

  // print_matrix_cuda(handle.nrow, handle.ncol, handle.nfib, dqv, handle.nfib, handle.ncol, handle.nfib);

  cudaMemcpyAsyncHelper(handle, qv.data(), dqv, (handle.nrow * handle.ncol * handle.nfib + size_ratio) * sizeof(int), D2H,
                        0);
  handle.sync_all();

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);

  cudaFreeHelper(dv);
  
  return (unsigned char *)buffer;
}

template unsigned char *
refactor_qz_cuda<double>(mgard_cuda_handle<double> & handle, double *u, int &outsize, double tol);
template unsigned char *
refactor_qz_cuda<float>(mgard_cuda_handle<float> & handle, float *u, int &outsize, float tol);


template <typename T>
T *recompose_udq_cuda(mgard_cuda_handle<T> & handle, unsigned char *data, int data_len) {
  int size_ratio = sizeof(T) / sizeof(int);
  std::vector<int> out_data(handle.nrow * handle.ncol * handle.nfib + size_ratio);

  T * dv;
  size_t dv_pitch;
  cudaMalloc3DHelper((void**)&dv, &dv_pitch, handle.nfib * sizeof(T), handle.ncol, handle.nrow);
  int lddv1 = dv_pitch / sizeof(T);
  int lddv2 = handle.ncol;
  int * dqv;
  cudaMallocHelper((void**)&dqv, (handle.nrow * handle.ncol * handle.nfib + size_ratio) * sizeof(int));

  T * v = (T *)malloc(handle.nrow * handle.ncol * handle.nfib * sizeof(T));

  mgard::decompress_memory_z(data, data_len, out_data.data(),
                             out_data.size() * sizeof(int)); // decompress input buffer

  cudaMemcpyAsyncHelper(handle, dqv, out_data.data(), (handle.nrow * handle.ncol * handle.nfib + size_ratio) * sizeof(int), H2D,
                        0);

  linear_dequantize(handle, handle.nrow, handle.ncol, handle.nfib, dv, lddv1, lddv2, dqv, handle.nfib, handle.ncol, 0);

  recompose_3D_cuda_cpt(handle, dv, lddv1, lddv2);

  if (!is_2kplus1_cuda (handle.nrow) || 
      !is_2kplus1_cuda (handle.ncol) || 
      !is_2kplus1_cuda (handle.nfib)) {
    postp_3D_cuda_cpt(handle, dv, lddv1, lddv2);
  }

  cudaMemcpy3DAsyncHelper(handle, v, handle.nfib  * sizeof(T), handle.nfib * sizeof(T), handle.ncol,
                     dv, lddv1 * sizeof(T), handle.nfib * sizeof(T), handle.ncol,
                     handle.nfib * sizeof(T), handle.ncol, handle.nrow,
                     D2H, 0);
  handle.sync_all();
  cudaFreeHelper(dv);
  return v;
}

template double *recompose_udq_cuda<double>(mgard_cuda_handle<double> & handle, unsigned char *data, int data_len);
template float *recompose_udq_cuda<float>(mgard_cuda_handle<float> & handle, unsigned char *data, int data_len);


template <typename T>
unsigned char *
refactor_qz_2D_cuda (mgard_cuda_handle<T> & handle, T *u, int &outsize, T tol)
{
  tol /= handle.l_target + 2;
  int size_ratio = sizeof (T) / sizeof (int);
  std::vector<int> qv (handle.nrow * handle.ncol + size_ratio);
  std::vector<unsigned char> out_data;

  T norm = max_norm_cuda(u, handle.nrow * handle.ncol);
                                      
  T * dv;
  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, handle.ncol * sizeof(T), handle.nrow);
  int lddv = dv_pitch / sizeof(T);
                                                        
  int * dqv;
  cudaMallocHelper((void**)&dqv, (handle.nrow * handle.ncol + size_ratio) * sizeof(int));
  
  cudaMemcpy2DAsyncHelper(handle, dv, lddv * sizeof(T), 
                          u, handle.ncol * sizeof(T), 
                          handle.ncol * sizeof(T), handle.nrow, 
                          H2D, 0);
      
  if (!is_2kplus1_cuda (handle.nrow) || !is_2kplus1_cuda (handle.ncol)) {
    if (handle.opt == 0) {
      prep_2D_cuda(handle, dv, lddv);
    } else if (handle.opt == 1) {
      prep_2D_cuda_cpt(handle, dv, lddv);
    }
  }

  if (handle.opt == 0) {
    refactor_2D_cuda(handle, dv, lddv);
  } else if (handle.opt == 1) {
    refactor_2D_cuda_cpt(handle, dv, lddv);
  }

  linear_quantize(handle, handle.nrow, handle.ncol, norm, tol, dv, lddv, dqv, handle.ncol, 0);


  cudaMemcpyAsyncHelper(handle, qv.data(), dqv, (handle.nrow * handle.ncol + size_ratio) * sizeof(int), D2H,
                        0);

  mgard::compress_memory_z (qv.data (), sizeof (int) * qv.size (), out_data);
 
  outsize = out_data.size ();
  unsigned char *buffer = (unsigned char *)malloc (outsize);
  std::copy (out_data.begin (), out_data.end (), buffer);

  cudaFreeHelper(dv);
  return buffer;
}


template unsigned char *
refactor_qz_2D_cuda<double>(mgard_cuda_handle<double> & handle, double *u, int &outsize, double tol);
template unsigned char *
refactor_qz_2D_cuda<float>(mgard_cuda_handle<float> & handle, float *u, int &outsize, float tol);

template <typename T>
T* recompose_udq_2D_cuda(mgard_cuda_handle<T> & handle, unsigned char *data, int data_len)
{
  int size_ratio = sizeof (T) / sizeof (int);
      
  T * dv;
  size_t dv_pitch;
  cudaMallocPitchHelper((void**)&dv, &dv_pitch, handle.ncol * sizeof(T), handle.nrow);
  int lddv = dv_pitch / sizeof(T);
  
  int ldv = handle.ncol;
  int * dqv;
  cudaMallocHelper((void**)&dqv, (handle.nrow*handle.ncol + size_ratio) * sizeof(int));

  T * v = (T *)malloc (handle.nrow*handle.ncol*sizeof(T));

  int * qv;
  cudaMallocHostHelper((void**)&qv, (handle.nrow*handle.ncol + size_ratio) * sizeof(int));
  
  mgard::decompress_memory_z(data, data_len, qv, (handle.nrow*handle.ncol + size_ratio) * sizeof(int));

  cudaMemcpyAsyncHelper(handle, dqv, qv, (handle.nrow*handle.ncol + size_ratio) * sizeof(int), H2D,
                        0);

  linear_dequantize(handle, handle.nrow, handle.ncol, dv, lddv, dqv, handle.ncol, 0);


  if (handle.opt == 0) {
    recompose_2D_cuda(handle, dv, lddv);
  } else if (handle.opt == 1) {
    recompose_2D_cuda_cpt(handle, dv, lddv);
  }
  
  if (!is_2kplus1_cuda (handle.nrow) || !is_2kplus1_cuda (handle.ncol)) {
    if (handle.opt == 0) {
      postp_2D_cuda(handle, dv, lddv);
    } else if (handle.opt == 1) {
      postp_2D_cuda_cpt(handle, dv, lddv);
    }
  }

  cudaMemcpy2DAsyncHelper(handle, v, ldv  * sizeof(T), 
                     dv, lddv * sizeof(T), 
                     handle.ncol * sizeof(T), handle.nrow, 
                     D2H, 0);
  return v;
}

template double * recompose_udq_2D_cuda<double>(mgard_cuda_handle<double> & handle, unsigned char *data, int data_len);
template float * recompose_udq_2D_cuda<float>(mgard_cuda_handle<float> & handle, unsigned char *data, int data_len);

}





