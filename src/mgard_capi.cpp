// Copyright 2017, Brown University, Providence, RI.
//
//                         All Rights Reserved
//
// Permission to use, copy, modify, and distribute this software and
// its documentation for any purpose other than its incorporation into a
// commercial product or service is hereby granted without fee, provided
// that the above copyright notice appear in all copies and that both
// that copyright notice and this permission notice appear in supporting
// documentation, and that the name of Brown University not be used in
// advertising or publicity pertaining to distribution of the software
// without specific, written prior permission.
//
// BROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
// INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
// PARTICULAR PURPOSE.  IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR
// ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//
//
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.1
//
// This file is part of MGARD.
//
// MGARD is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// MGARD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with MGARD.  If not, see <http://www.gnu.org/licenses/>.

#include "mgard.h"

extern "C" unsigned char *mgard_compress(int itype_flag,  void  *data, int &out_size, int nrow, int ncol, void* tol_in)
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
      const float *v = static_cast<const float*>(data);
      float *tol = static_cast<float*>(tol_in);
      assert (*tol >= 1e-8);
      
      unsigned char* mgard_compressed_ptr;

      //mgard_compressed_ptr = mgard::refactor_qz_float(nrow, ncol, v, out_size, *tol);

      return mgard_compressed_ptr;
    }
  
}




extern "C" void mgard_sirius_compress(int itype_flag,  double  *data, unsigned char *compressed_coarse, unsigned char *compressed_fine, int &out_size_1, int &out_size_2, int nrow, int ncol, double *tol_coarse, double *tol_fine)
{ 

  assert (ncol > 3);
  assert (nrow >= 1);

  if(itype_flag == 0)
    {
      // float *v   = static_cast<float*>(data);
      // float *tolc = static_cast<float*>(tol_coarse);
      // float *tolf = static_cast<float*>(tol_fine);
      assert (*tol_coarse >= 1e-8);
      assert (*tol_fine >= 1e-8);
      
      //mgard_compressed_ptr = mgard::refactor_qz_float(nrow, ncol, v, out_size, *tol);

      //return mgard_compressed_ptr;
    }
  else if(itype_flag == 1)
    {
      // double *v   = static_cast<double*>(data);
      // double *tolc = static_cast<double*>(tol_coarse);
      // double *tolf = static_cast<double*>(tol_fine);

      assert (*tol_coarse >= 1e-8);
      assert (*tol_fine >= 1e-8);

      
      compressed_coarse = mgard::refactor_qz(nrow, ncol, data, out_size_1, *tol_coarse);
      compressed_fine   = mgard::refactor_qz(nrow, ncol, data, out_size_2, *tol_fine);

      // std::string out_file = "test.dat";
      // std::ofstream outfile(out_file, std::ios::out | std::ios::binary);
      // outfile.write( reinterpret_cast<char*>( compressed_fine ),  out_size_2);
    }
  else
    {
      std::cerr <<"MGARD: Unknown data type, assuming 32-bit floats...\n";

    }
  
}


extern "C" void *mgard_decompress(int itype_flag,  unsigned char *data, int data_len, int nrow, int ncol)
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

      //      mgard_decompressed_ptr = static_cast<double*> (mgard::recompose_udq(nrow, ncol, data, data_len));

      mgard_decompressed_ptr = mgard::recompose_udq(nrow, ncol, data, data_len);
      return mgard_decompressed_ptr;
    }
  else
    {
      std::cerr <<"MGARD: Unknown data type, assuming 32-bit floats...\n";

      float* mgard_decompressed_ptr;

      //mgard_decompressed_ptr = mgard::recompose_udq_float(nrow, ncol, v, out_size, *tol);

    }
  
}


extern "C" void mgard_sirius_decompress(int itype_flag,  unsigned char *data_coarse, unsigned char *data_fine, double *v_coarse, double *v_fine, int data_len_coarse, int data_len_fine, int nrow, int ncol)
{
  assert (ncol > 3);
  assert (nrow >= 1);

  if(itype_flag == 0)
    {
      std::cerr <<"MGARD: Unknown data type, assuming 32-bit floats...\n";
    }
  else if(itype_flag == 1)
    {
      
      v_coarse = mgard::recompose_udq(nrow, ncol, data_coarse, data_len_coarse);
      v_fine   = mgard::recompose_udq(nrow, ncol, data_fine  , data_len_fine);

    }
  else
    {
      std::cerr <<"MGARD: Unknown data type, assuming 32-bit floats...\n";

      //      float* mgard_decompressed_ptr;

      //mgard_decompressed_ptr = mgard::recompose_udq_float(nrow, ncol, v, out_size, *tol);

    }
  
}


