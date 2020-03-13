#include "mgard.h"
#include "mgard_api.h"

#include "mgard_cuda.h"
#include "mgard_cuda_helper.h"
#include "mgard_api_cuda.h"

template <typename T>
unsigned char *mgard_compress_cuda(int itype_flag,  T  *v, int &out_size, int nrow, int ncol, int nfib, T tol_in)

 //Perform compression preserving the tolerance in the L-infty norm
{ 
  
  T tol = tol_in;
  assert (tol >= 1e-7);
  unsigned char* mgard_compressed_ptr = nullptr;
  if(nrow > 1 && ncol > 1 && nfib > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      assert (nfib > 3);

      mgard_cuda_handle * handle = new mgard_cuda_handle(1);
      int B = 16;
      bool profile = false;

      mgard_compressed_ptr = mgard::refactor_qz_cuda(nrow, ncol, nfib, v, out_size, tol, B, *handle, profile);
      return mgard_compressed_ptr;
      
    } else if (nrow > 1 && ncol > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      mgard_cuda_handle * handle = new mgard_cuda_handle(1);
      int B = 16;
      bool profile = false;

      //mgard_compressed_ptr = mgard::refactor_qz_2D(nrow, ncol, v, out_size, tol);
      mgard_compressed_ptr = mgard::refactor_qz_2D_cuda(nrow, ncol, v, out_size, tol, 0, B, *handle, profile);
      return mgard_compressed_ptr;
    } else if (nrow > 1 ) {
      assert (nrow > 3);
      std::cerr <<"MGARD: Not impemented!  Let us know if you need 1D compression...\n";
      return nullptr;
    }

  return nullptr;
}

template unsigned char *mgard_compress_cuda<double>(int itype_flag, double *v, int &out_size, int nrow, int ncol, int nfib, double tol_in);
//template unsigned char *mgard_compress_cuda<float>(int itype_flag, float *v, int &out_size, int nrow, int ncol, int nfib, float tol_in);

template <typename T>
T *mgard_decompress_cuda(int itype_flag,  T& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib)
{
  T* mgard_decompressed_ptr = nullptr;
      
  if(nrow > 1 && ncol > 1 && nfib > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      assert (nfib > 3);
      
      mgard_decompressed_ptr = mgard::recompose_udq(nrow, ncol, nfib, data, data_len);
      return mgard_decompressed_ptr;      
  } else if (nrow > 1 && ncol > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      //mgard_decompressed_ptr = mgard::recompose_udq_2D(nrow, ncol, data, data_len);
      mgard_cuda_handle * handle = new mgard_cuda_handle(1);
      int B = 16;
      bool profile = false;
      T dummy = 0;
      mgard_decompressed_ptr = mgard::recompose_udq_2D_cuda(nrow, ncol, data, data_len, 0, B, *handle, profile, dummy);
      //          mgard_decompressed_ptr = mgard::recompose_udq_2D(nrow, ncol, data, data_len);
      return mgard_decompressed_ptr;
  } else if (nrow > 1 ) {
      assert (nrow > 3);
      std::cerr <<"MGARD: Not impemented!  Let us know if you need 1D compression...\n";
      //mgard_decompressed_ptr = mgard::recompose_udq_1D(nrow,  data, data_len);
  }
  return nullptr; 
}

template double *mgard_decompress_cuda<double>(int itype_flag,  double& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib);
//template float *mgard_decompress_cuda<float>(int itype_flag,  float& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib);

template <typename T>
unsigned char *mgard_compress_cuda(int itype_flag,  T  *v, int &out_size, int nrow, int ncol, int nfib, T tol_in, int opt, 
                                    int B, bool profile)

 //Perform compression preserving the tolerance in the L-infty norm
{ 
  
  T tol = tol_in;
  assert (tol >= 1e-7);
  unsigned char* mgard_compressed_ptr = nullptr;
  if(nrow > 1 && ncol > 1 && nfib > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      assert (nfib > 3);
      mgard_cuda_handle * handle = new mgard_cuda_handle(1);
      mgard_compressed_ptr = mgard::refactor_qz_cuda(nrow, ncol, nfib, v, out_size, tol, B, *handle, profile);
      return mgard_compressed_ptr;
      
    } else if (nrow > 1 && ncol > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      mgard_cuda_handle * handle = new mgard_cuda_handle(1);
       
      //mgard_compressed_ptr = mgard::refactor_qz_2D(nrow, ncol, v, out_size, tol);
      mgard_compressed_ptr = mgard::refactor_qz_2D_cuda(nrow, ncol, v, out_size, tol, opt, B, *handle, profile);
      return mgard_compressed_ptr;
    } else if (nrow > 1 ) {
      assert (nrow > 3);
      std::cerr <<"MGARD: Not impemented!  Let us know if you need 1D compression...\n";
      return nullptr;
    }

  return nullptr;
}

template unsigned char *mgard_compress_cuda<double>(int itype_flag, double *v, int &out_size, int nrow, int ncol, int nfib, double tol_in, int opt, 
                                    int B, bool profile);
// template unsigned char *mgard_compress_cuda<float>(int itype_flag, float *v, int &out_size, int nrow, int ncol, int nfib, float tol_in, int opt, 
//                                     int B, bool profile);


template <typename T>
T *mgard_decompress_cuda(int itype_flag,  T& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib, int opt,
                              int B, bool profile)
{
  T* mgard_decompressed_ptr = nullptr;
      
  if(nrow > 1 && ncol > 1 && nfib > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      assert (nfib > 3);
      
      mgard_decompressed_ptr = mgard::recompose_udq(nrow, ncol, nfib, data, data_len);
      return mgard_decompressed_ptr;      
  } else if (nrow > 1 && ncol > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      mgard_cuda_handle * handle = new mgard_cuda_handle(1);
      T dummy;
      //mgard_decompressed_ptr = mgard::recompose_udq_2D(nrow, ncol, data, data_len);
      mgard_decompressed_ptr = mgard::recompose_udq_2D_cuda(nrow, ncol, data, data_len, opt, B, *handle, profile, dummy);
      //          mgard_decompressed_ptr = mgard::recompose_udq_2D(nrow, ncol, data, data_len);
      return mgard_decompressed_ptr;
  } else if (nrow > 1 ) {
      assert (nrow > 3);
      std::cerr <<"MGARD: Not impemented!  Let us know if you need 1D compression...\n";
      //mgard_decompressed_ptr = mgard::recompose_udq_1D(nrow,  data, data_len);
  }
  return nullptr; 
}

template double *mgard_decompress_cuda(int itype_flag,  double& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib, int opt,
                              int B, bool profile);
// template float *mgard_decompress_cuda(int itype_flag,  float& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib, int opt,
//                               int B, bool profile);
