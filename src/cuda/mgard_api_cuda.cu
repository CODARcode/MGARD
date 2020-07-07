#include "mgard.h"
#include "mgard_api.h"

#include "cuda/mgard_cuda.h"
#include "cuda/mgard_cuda_common.h"

template <typename T>
unsigned char *mgard_compress_cuda(int itype_flag,  T  * v, int &out_size, int nrow, int ncol, int nfib, T tol_in)

 //Perform compression preserving the tolerance in the L-infty norm
{ 
  
  T tol = tol_in;
  assert (tol >= 1e-7);
  unsigned char* mgard_compressed_ptr = nullptr;
  if(nrow > 1 && ncol > 1 && nfib > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      assert (nfib > 3);

      mgard_cuda_handle<T> handle(nrow, ncol, nfib);
      mgard_compressed_ptr = mgard_cuda::refactor_qz_cuda(handle, v, out_size, tol);
      return mgard_compressed_ptr;
      
    } else if (nrow > 1 && ncol > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      mgard_cuda_handle<T> handle(nrow, ncol, nfib);
      mgard_compressed_ptr = mgard_cuda::refactor_qz_2D_cuda(handle, v, out_size, tol);
      return mgard_compressed_ptr;
    } else if (nrow > 1 ) {
      assert (nrow > 3);
      std::cerr <<"MGARD: Not impemented!  Let us know if you need 1D compression...\n";
      return nullptr;
    }

  return nullptr;
}

template unsigned char *mgard_compress_cuda<double>(int itype_flag, double *v, int &out_size, int nrow, int ncol, int nfib, double tol_in);
template unsigned char *mgard_compress_cuda<float>(int itype_flag, float *v, int &out_size, int nrow, int ncol, int nfib, float tol_in);

template <typename T>
T *mgard_decompress_cuda(int itype_flag,  T& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib)
{
  T* mgard_decompressed_ptr = nullptr;
      
  if(nrow > 1 && ncol > 1 && nfib > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      assert (nfib > 3);
      mgard_cuda_handle<T> handle (nrow, ncol, nfib);
      mgard_decompressed_ptr = mgard_cuda::recompose_udq_cuda(handle, data, data_len);
      return mgard_decompressed_ptr;      
  } else if (nrow > 1 && ncol > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      mgard_cuda_handle<T> handle(nrow, ncol, nfib);
      mgard_decompressed_ptr = mgard_cuda::recompose_udq_2D_cuda(handle, data, data_len);
      return mgard_decompressed_ptr;
  } else if (nrow > 1 ) {
      assert (nrow > 3);
      std::cerr <<"MGARD: Not impemented!  Let us know if you need 1D compression...\n";
      //mgard_decompressed_ptr = mgard::recompose_udq_1D(nrow,  data, data_len);
  }
  return nullptr; 
}

template double *mgard_decompress_cuda<double>(int itype_flag,  double& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib);
template float *mgard_decompress_cuda<float>(int itype_flag,  float& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib);

template <typename T>
unsigned char *mgard_compress_cuda(mgard_cuda_handle<T> & handle, int itype_flag, T * v, int &out_size, T tol_in)

 //Perform compression preserving the tolerance in the L-infty norm
{ 
  
  T tol = tol_in;
  assert (tol >= 1e-7);
  unsigned char* mgard_compressed_ptr = nullptr;
  if(handle.nrow > 1 && handle.ncol > 1 && handle.nfib > 1) {
      assert (handle.nrow > 3);
      assert (handle.ncol > 3);
      assert (handle.nfib > 3);
      mgard_compressed_ptr = mgard_cuda::refactor_qz_cuda(handle, v, out_size, tol);
      return mgard_compressed_ptr;
      
    } else if (handle.nrow > 1 && handle.ncol > 1) {
      assert (handle.nrow > 3);
      assert (handle.ncol > 3);
      //mgard_cuda_handle * handle = new mgard_cuda_handle(1);
       
      //mgard_compressed_ptr = mgard::refactor_qz_2D(nrow, ncol, v, out_size, tol);
      mgard_compressed_ptr = mgard_cuda::refactor_qz_2D_cuda(handle, v, out_size, tol);
      return mgard_compressed_ptr;
    } else if (handle.nrow > 1 ) {
      assert (handle.nrow > 3);
      std::cerr <<"MGARD: Not impemented!  Let us know if you need 1D compression...\n";
      return nullptr;
    }

  return nullptr;
}

template unsigned char *mgard_compress_cuda<double>(mgard_cuda_handle<double> & handle, int itype_flag, double *v, int &out_size, double tol_in);
template unsigned char *mgard_compress_cuda<float>(mgard_cuda_handle<float> & handle, int itype_flag, float *v, int &out_size, float tol_in);


template <typename T>
T *mgard_decompress_cuda(mgard_cuda_handle<T> & handle, int itype_flag,  T& quantizer, unsigned char *data, int data_len)
{
  T* mgard_decompressed_ptr = nullptr;
      
  if(handle.nrow > 1 && handle.ncol > 1 && handle.nfib > 1) {
      assert (handle.nrow > 3);
      assert (handle.ncol > 3);
      assert (handle.nfib > 3);
      mgard_decompressed_ptr = mgard_cuda::recompose_udq_cuda(handle, data, data_len);
      return mgard_decompressed_ptr;      
  } else if (handle.nrow > 1 && handle.ncol > 1) {
      assert (handle.nrow > 3);
      assert (handle.ncol > 3);
      mgard_decompressed_ptr = mgard_cuda::recompose_udq_2D_cuda(handle, data, data_len);
      return mgard_decompressed_ptr;
  } else if (handle.nrow > 1 ) {
      assert (handle.nrow > 3);
      std::cerr <<"MGARD: Not impemented!  Let us know if you need 1D compression...\n";
      //mgard_decompressed_ptr = mgard::recompose_udq_1D(nrow,  data, data_len);
  }
  return nullptr; 
}

template double *mgard_decompress_cuda(mgard_cuda_handle<double> & handle, int itype_flag,  double& quantizer, unsigned char *data, int data_len);
template float *mgard_decompress_cuda(mgard_cuda_handle<float> & handle, int itype_flag,  float& quantizer, unsigned char *data, int data_len);

