// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk

// version: 0.0.0.1

// This file is part of MGARD.

// MGARD is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Foobar is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License
// along with MGARD.  If not, see <http://www.gnu.org/licenses/>.


#include "mgard.h"

extern "C" unsigned char *mgard_compress(int itype_flag, void *data, int &out_size, int nrow, int ncol, void* tol_in)
{ 

  assert (ncol > 3);
  assert (nrow >= 1);

  if(itype_flag == 0)
    {
      float *v   = static_cast<float*>(data);
      float *tol = static_cast<float*>(tol_in);
      assert (*tol >= 1e-8);
      
      unsigned char* mgard_compressed_ptr;

      //mgard_compressed_ptr = mgard::refactor_qz_float(nrow, ncol, v, out_size, *tol);

      return mgard_compressed_ptr;
    }
  else if(itype_flag == 1)
    {
      double *v   = static_cast<double*>(data);
      double *tol = static_cast<double*>(tol_in);
      assert (*tol >= 1e-8);
      
      unsigned char* mgard_compressed_ptr;

      mgard_compressed_ptr = mgard::refactor_qz(nrow, ncol, v, out_size, *tol);

      return mgard_compressed_ptr;
    }
  else
    {
      std::cerr <<"MGARD: Unknown data type, assuming 32-bit floats...\n";
      float *v = static_cast<float*>(data);
      float *tol = static_cast<float*>(tol_in);
      assert (*tol >= 1e-8);
      
      unsigned char* mgard_compressed_ptr;

      //mgard_compressed_ptr = mgard::refactor_qz_float(nrow, ncol, v, out_size, *tol);

      return mgard_compressed_ptr;
    }
  
}

extern "C" void *mgard_decompress(int itype_flag, unsigned char *data, int data_len, int nrow, int ncol)
{
  assert (ncol > 3);
  assert (nrow >= 1);

  if(itype_flag == 0)
    {
      
      float* mgard_decompressed_ptr;

      //      mgard_decompressed_ptr = mgard::recompose_udq_float(nrow, ncol, v, out_size, *tol);

      return mgard_decompressed_ptr;
    }
  else if(itype_flag == 1)
    {
      
      double* mgard_decompressed_ptr;

      mgard_decompressed_ptr = static_cast<double*> (mgard::recompose_udq(nrow, ncol, data, data_len));

      return mgard_decompressed_ptr;
    }
  else
    {
      std::cerr <<"MGARD: Unknown data type, assuming 32-bit floats...\n";

      float* mgard_decompressed_ptr;

      //mgard_decompressed_ptr = mgard::recompose_udq_float(nrow, ncol, v, out_size, *tol);

    }
  
}


