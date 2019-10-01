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
// version: 0.0.0.2
//
// This file is part of MGARD.
//
// MGARD is distributed under the OSI-approved Apache License, Version 2.0.
// See accompanying file Copyright.txt for details.
//


#include "mgard_api.h"
#include "mgard.h"

unsigned char *mgard_compress(int itype_flag,  double  *v, int &out_size, int nrow, int ncol, int nfib, double tol_in)

 //Perform compression preserving the tolerance in the L-infty norm
{ 
  if(itype_flag == 0)
    {

      double tol = tol_in;
      assert (tol >= 1e-8);
      
      unsigned char* mgard_compressed_ptr = nullptr;

      return mgard_compressed_ptr;
    }
  else if(itype_flag == 1)
    {
      double tol = tol_in;
      assert (tol >= 1e-8);
      unsigned char* mgard_compressed_ptr = nullptr;
      if(nrow > 1 && ncol > 1 && nfib > 1)
        {
          assert (nrow > 3);
          assert (ncol > 3);
          assert (nfib > 3);
          
	  mgard_compressed_ptr = mgard::refactor_qz(nrow, ncol, nfib, v, out_size, tol);
          return mgard_compressed_ptr;
                
        }
      else if (nrow > 1 && ncol > 1)
        {
          assert (nrow > 3);
          assert (ncol > 3);
          mgard_compressed_ptr = mgard::refactor_qz_2D(nrow, ncol, v, out_size, tol);
          return mgard_compressed_ptr;
        }
      else if (nrow > 1 )
        {
          assert (nrow > 3);
	  std::cerr <<"MGARD: Let us know if you need 1D compression...\n";
	  return nullptr;
        }

      
    }
  else
    {
      std::cerr <<"MGARD: Unknown data type...\n";
      unsigned char* mgard_compressed_ptr = nullptr;

      return mgard_compressed_ptr;
    }
  
}



unsigned char *mgard_compress(int itype_flag, double  *v, int &out_size, int nrow, int ncol, int nfib, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z , double tol)
 //Perform compression preserving the tolerance in the L-infty norm, arbitrary tensor grids
{ 
  if(itype_flag == 0)
    {

      assert (tol >= 1e-8);
      
      unsigned char* mgard_compressed_ptr = nullptr;


      return mgard_compressed_ptr;
    }
  else if(itype_flag == 1)
    {
      assert (tol >= 1e-8);
      unsigned char* mgard_compressed_ptr = nullptr;
      if(nrow > 1 && ncol > 1 && nfib > 1)
        {
          assert (nrow > 3);
          assert (ncol > 3);
          assert (nfib > 3);
          
	  mgard_compressed_ptr = mgard::refactor_qz(nrow, ncol, nfib, coords_x, coords_y, coords_z, v, out_size, tol);
          return mgard_compressed_ptr;
                
        }
      else if (nrow > 1 && ncol > 1)
        {
          assert (nrow > 3);
          assert (ncol > 3);
          mgard_compressed_ptr = mgard::refactor_qz_2D(nrow, ncol, coords_x, coords_y, v, out_size, tol);
          return mgard_compressed_ptr;
        }
      else if (nrow > 1 )
        {
          assert (nrow > 3);
	  std::cerr <<"MGARD: Let us know if you need 1D compression...\n";
	  return nullptr;
        }

      
    }
  else
    {
      std::cerr <<"MGARD: Unknown data type exiting...\n";
      return nullptr;
    }
  
}




unsigned char *mgard_compress(int itype_flag,  double  *v, int &out_size, int nrow, int ncol, int nfib, double tol_in, double s )
{
  //Perform compression preserving the tolerance in s norm by defaulting to the s-norm
  if(itype_flag == 0)
    {


      double tol = tol_in;
      assert (tol >= 1e-8);
      
      unsigned char* mgard_compressed_ptr = nullptr;

      return mgard_compressed_ptr;
    }
  else if(itype_flag == 1)
    {
      double tol = tol_in;
      assert (tol >= 1e-8);

      unsigned char* mgard_compressed_ptr = nullptr;
      if(nrow > 1 && ncol > 1 && nfib > 1)
        {
          assert (nrow > 3);
          assert (ncol > 3);
          assert (nfib > 3);
          
          mgard_compressed_ptr = mgard::refactor_qz(nrow, ncol, nfib, v, out_size, tol, s);
          return mgard_compressed_ptr;
                
        }
      else if (nrow > 1 && ncol > 1)
        {
          assert (nrow > 3);
          assert (ncol > 3);
          mgard_compressed_ptr = mgard::refactor_qz_2D(nrow, ncol, v, out_size, tol, s);
	  
          return mgard_compressed_ptr;
        }
      else if (nrow > 1 )
        {
          assert (nrow > 3);
	  std::cerr <<"MGARD: Let us know if you need 1D compression...\n";
	  return nullptr;
        }

      
    }
  else
    {
      std::cerr <<"MGARD: Unknown data type...\n";

      return nullptr;
    }
  
}



//unsigned char *mgard_compress(int itype_flag,  double  *v, int &out_size, int nrow, int ncol, int nfib, double tol_in, double (*qoi) (int, int, int, std::vector<double>), double s)

unsigned char *mgard_compress(int itype_flag,  double  *v, int &out_size, int nrow, int ncol, int nfib, double tol_in, double (*qoi) (int, int, int, double*), double s)
{
  //Perform compression preserving the tolerance in s norm by defaulting to the L-2 norm
  if(itype_flag == 0)
    {
 
      double tol = tol_in;
      assert (tol >= 1e-8);
      
      unsigned char* mgard_compressed_ptr = nullptr;

      //mgard_compressed_ptr = mgard::refactor_qz_float(nrow, ncol, v, out_size, *tol);

      return mgard_compressed_ptr;
    }
  else if(itype_flag == 1)
    {
 
      double tol = tol_in;
      assert (tol >= 1e-8);
      unsigned char* mgard_compressed_ptr = nullptr;
      if(nrow > 1 && ncol > 1 && nfib > 1)
        {
          assert (nrow > 3);
          assert (ncol > 3);
          assert (nfib > 3);

	  std::vector<double> coords_x(ncol), coords_y(nrow), coords_z(nfib); // coordinate arrays
	    //dummy equispaced coordinates
	  std::iota(std::begin(coords_x), std::end(coords_x), 0);
	  std::iota(std::begin(coords_y), std::end(coords_y), 0);
	  std::iota(std::begin(coords_z), std::end(coords_z), 0);

	  double xi_norm =  mgard_gen::qoi_norm(nrow,  ncol,  nfib, coords_x,  coords_y, coords_z, qoi, s);
	  tol *= xi_norm;
          mgard_compressed_ptr = mgard::refactor_qz(nrow, ncol, nfib, v, out_size, tol, -s);
          return mgard_compressed_ptr;
                
        }
      else if (nrow > 1 && ncol > 1)
        {
          assert (nrow > 3);
          assert (ncol > 3);

	  std::vector<double> coords_x(ncol), coords_y(nrow), coords_z(nfib); // coordinate arrays
	  //dummy equispaced coordinates
	  std::iota(std::begin(coords_x), std::end(coords_x), 0);
	  std::iota(std::begin(coords_y), std::end(coords_y), 0);
	  std::iota(std::begin(coords_z), std::end(coords_z), 0);
	  

	  double xi_norm =  mgard_gen::qoi_norm(nrow,  ncol,  nfib, coords_x,  coords_y, coords_z, qoi, s);
	  tol *= xi_norm;

          mgard_compressed_ptr = mgard::refactor_qz_2D(nrow, ncol, v, out_size, tol, -s);
          return mgard_compressed_ptr;
        }
      else if (nrow > 1 )
        {
          assert (nrow > 3);
          //mgard_compressed_ptr = mgard::refactor_qz_1D(nrow, v, out_size, *tol);
        }

      
    }
  else
    {
      std::cerr <<"MGARD: Unknown data type, assuming 32-bit floats...\n";
      // const float *v = static_cast<const float*>(data);
      // float *tol = static_cast<float*>(tol_in);
      // assert (*tol >= 1e-8);
      
      unsigned char* mgard_compressed_ptr = nullptr;

      //mgard_compressed_ptr = mgard::refactor_qz_float(nrow, ncol, v, out_size, *tol);

      return mgard_compressed_ptr;
    }
  
}

double *mgard_decompress(int itype_flag,  double& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib)
{
  assert (ncol > 3);
  assert (nrow >= 1);

  if(itype_flag == 0)
    {
      
      float* mgard_decompressed_ptr;

      //      mgard_decompressed_ptr = mgard::recompose_udq_float(nrow, ncol, v, out_size, *tol);

      //      return mgard_decompressed_ptr;
    }
  else if(itype_flag == 1)
    {
      
      double* mgard_decompressed_ptr = nullptr;
      
      //      mgard_decompressed_ptr = static_cast<double*> (mgard::recompose_udq(nrow, ncol, data, data_len));



      if(nrow > 1 && ncol > 1 && nfib > 1)
        {
          assert (nrow > 3);
          assert (ncol > 3);
          assert (nfib > 3);
	  
          mgard_decompressed_ptr = mgard::recompose_udq(nrow, ncol, nfib, data, data_len);
          return mgard_decompressed_ptr;      
        }
      else if (nrow > 1 && ncol > 1)
        {
          assert (nrow > 3);
          assert (ncol > 3);
          mgard_decompressed_ptr = mgard::recompose_udq_2D(nrow, ncol, data, data_len);
          //          mgard_decompressed_ptr = mgard::recompose_udq_2D(nrow, ncol, data, data_len);
          return mgard_decompressed_ptr;
        }
      else if (nrow > 1 )
        {
          assert (nrow > 3);
          //mgard_decompressed_ptr = mgard::recompose_udq_1D(nrow,  data, data_len);
        }
      

    }
  else
    {
      std::cerr <<"MGARD: Unknown data type, assuming 32-bit floats...\n";

      float* mgard_decompressed_ptr = nullptr;

      //mgard_decompressed_ptr = mgard::recompose_udq_float(nrow, ncol, v, out_size, *tol);

    }
  
}


double  *mgard_decompress(int itype_flag,  double& quantizer, unsigned char *data, int data_len, int nrow, int ncol, int nfib, double s)
{
  assert (ncol > 3);
  assert (nrow >= 1);

  if(itype_flag == 0)
    {
      
      float* mgard_decompressed_ptr;

      //      mgard_decompressed_ptr = mgard::recompose_udq_float(nrow, ncol, v, out_size, *tol);

      //      return mgard_decompressed_ptr;
    }
  else if(itype_flag == 1)
    {
      
      double* mgard_decompressed_ptr = nullptr;

      //      mgard_decompressed_ptr = static_cast<double*> (mgard::recompose_udq(nrow, ncol, data, data_len));



      if(nrow > 1 && ncol > 1 && nfib > 1)
        {
          assert (nrow > 3);
          assert (ncol > 3);
          assert (nfib > 3);

          mgard_decompressed_ptr = mgard::recompose_udq(nrow, ncol, nfib, data, data_len, s);
          return mgard_decompressed_ptr;      
        }
      else if (nrow > 1 && ncol > 1)
        {
          assert (nrow > 3);
          assert (ncol > 3);
          mgard_decompressed_ptr = mgard::recompose_udq_2D(nrow, ncol, data, data_len, s);
          //          mgard_decompressed_ptr = mgard::recompose_udq_2D(nrow, ncol, data, data_len);
          return mgard_decompressed_ptr;
        }
      else if (nrow > 1 )
        {
          assert (nrow > 3);
          //mgard_decompressed_ptr = mgard::recompose_udq_1D(nrow,  data, data_len);
        }
      

    }
  else
    {
      std::cerr <<"MGARD: Unknown data type, assuming 32-bit floats...\n";

      float* mgard_decompressed_ptr = nullptr;

      //mgard_decompressed_ptr = mgard::recompose_udq_float(nrow, ncol, v, out_size, *tol);

    }
  
}


double mgard_compress(int nrow, int ncol, int nfib, double (*qoi) (int, int, int, std::vector<double>), double s)
{
  std::vector<double> coords_x(ncol), coords_y(nrow), coords_z(nfib); // coordinate arrays
  //dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  double xi_norm =  mgard_gen::qoi_norm(nrow,  ncol,  nfib, coords_x,  coords_y, coords_z, qoi, s);

  return xi_norm;
}



double mgard_compress(int nrow, int ncol, int nfib, double (*qoi) (int, int, int, double*), double s)
{
  std::vector<double> coords_x(ncol), coords_y(nrow), coords_z(nfib); // coordinate arrays
  //dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  double xi_norm =  mgard_gen::qoi_norm(nrow,  ncol,  nfib, coords_x,  coords_y, coords_z, qoi, s);

  return xi_norm;
}


unsigned char *mgard_compress(int itype_flag,  double  *v, int &out_size, int nrow, int ncol, int nfib, double tol_in, double norm_of_qoi, double s)
{
  tol_in *= norm_of_qoi;
  return mgard_compress(itype_flag, v, out_size,  nrow,  ncol,  nfib,  tol_in, s);

}


