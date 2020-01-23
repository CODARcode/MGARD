#include "mgard.h"
#include "mgard_api.h"

#include "mgard_cuda.h"
#include "mgard_api_cuda.h"


unsigned char *mgard_compress_cuda(int itype_flag,  double  *v, int &out_size, int nrow, int ncol, int nfib, double tol_in)

 //Perform compression preserving the tolerance in the L-infty norm
{ 
  
  double tol = tol_in;
  assert (tol >= 1e-7);
  unsigned char* mgard_compressed_ptr = nullptr;
  if(nrow > 1 && ncol > 1 && nfib > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      assert (nfib > 3);
      
      mgard_compressed_ptr = mgard::refactor_qz(nrow, ncol, nfib, v, out_size, tol);
      return mgard_compressed_ptr;
      
    } else if (nrow > 1 && ncol > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      //mgard_compressed_ptr = mgard::refactor_qz_2D(nrow, ncol, v, out_size, tol);
      mgard_compressed_ptr = mgard::refactor_qz_2D_cuda(nrow, ncol, v, out_size, tol, 0);
      return mgard_compressed_ptr;
    } else if (nrow > 1 ) {
      assert (nrow > 3);
      std::cerr <<"MGARD: Not impemented!  Let us know if you need 1D compression...\n";
      return nullptr;
    }

  return nullptr;
}

double *mgard_decompress_cuda(int itype_flag,  double& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib)
{
  double* mgard_decompressed_ptr = nullptr;
      
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
      mgard_decompressed_ptr = mgard::recompose_udq_2D_cuda(nrow, ncol, data, data_len, 0);
      //          mgard_decompressed_ptr = mgard::recompose_udq_2D(nrow, ncol, data, data_len);
      return mgard_decompressed_ptr;
  } else if (nrow > 1 ) {
      assert (nrow > 3);
      std::cerr <<"MGARD: Not impemented!  Let us know if you need 1D compression...\n";
      //mgard_decompressed_ptr = mgard::recompose_udq_1D(nrow,  data, data_len);
  }
  return nullptr; 
}



unsigned char *mgard_compress_cuda(int itype_flag,  double  *v, int &out_size, int nrow, int ncol, int nfib, double tol_in, int opt)

 //Perform compression preserving the tolerance in the L-infty norm
{ 
  
  double tol = tol_in;
  assert (tol >= 1e-7);
  unsigned char* mgard_compressed_ptr = nullptr;
  if(nrow > 1 && ncol > 1 && nfib > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      assert (nfib > 3);
      
      mgard_compressed_ptr = mgard::refactor_qz(nrow, ncol, nfib, v, out_size, tol);
      return mgard_compressed_ptr;
      
    } else if (nrow > 1 && ncol > 1) {
      assert (nrow > 3);
      assert (ncol > 3);
      //mgard_compressed_ptr = mgard::refactor_qz_2D(nrow, ncol, v, out_size, tol);
      mgard_compressed_ptr = mgard::refactor_qz_2D_cuda(nrow, ncol, v, out_size, tol, opt);
      return mgard_compressed_ptr;
    } else if (nrow > 1 ) {
      assert (nrow > 3);
      std::cerr <<"MGARD: Not impemented!  Let us know if you need 1D compression...\n";
      return nullptr;
    }

  return nullptr;
}

double *mgard_decompress_cuda(int itype_flag,  double& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib, int opt)
{
  double* mgard_decompressed_ptr = nullptr;
      
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
      mgard_decompressed_ptr = mgard::recompose_udq_2D_cuda(nrow, ncol, data, data_len, opt);
      //          mgard_decompressed_ptr = mgard::recompose_udq_2D(nrow, ncol, data, data_len);
      return mgard_decompressed_ptr;
  } else if (nrow > 1 ) {
      assert (nrow > 3);
      std::cerr <<"MGARD: Not impemented!  Let us know if you need 1D compression...\n";
      //mgard_decompressed_ptr = mgard::recompose_udq_1D(nrow,  data, data_len);
  }
  return nullptr; 
}