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
// #include "mgard.h"
// #include "mgard_nuni.h"


inline int get_index3(const int ncol, const int nfib, const int i, const int j, const int k)
{
  return (ncol*i + j)*nfib + k;
}


double qoi(const int nrow, const int ncol, const int nfib, std::vector<double> u)
{
  int i = 100;
  return u[i];
}



double qoi_x(const int nrow, const int ncol, const int nfib, std::vector<double> u)
{

  int type_indicator = 0 ;
  // int nrow = 111;
  // int ncol = 160;
  // int nfib = 15;

  for(int irow = 0; irow < nrow; ++irow )
    {
      for(int jcol = 0; jcol < ncol; ++jcol )
        {
          for(int kfib = 0; kfib < nfib; ++kfib)
            {
	      if  (u[get_index3(ncol,nfib,irow,jcol,kfib)] != 0)
		return jcol;
            }
          
        }
    }
}


double qoi_ave(const int nrow, const int ncol, const int nfib, std::vector<double> u)
{
  double  sum = 0;

  for ( double x : u ) sum += x;
  
  return sum/u.size();
}

double qoi_one(const int nrow, const int ncol, const int nfib, std::vector<double> u)
{
  double qov = 1.0;

  int type_indicator = 0 ;
  // int nrow = 111;
  // int ncol = 160;
  // int nfib = 15;
  double h;
  
  for(int irow = 0; irow < nrow; ++irow )
    {
      for(int jcol = 0; jcol < ncol; ++jcol )
        {
          for(int kfib = 0; kfib < nfib; ++kfib)
            {
	      if((irow == 0 || irow  == nrow -1) && (u[get_index3(ncol,nfib,irow,jcol,kfib)] != 0))
	       	++type_indicator;

	      if((jcol == 0 || jcol  == ncol -1) && (u[get_index3(ncol,nfib,irow,jcol,kfib)] != 0))
	       	++type_indicator;
	      
	      if((kfib == 0 || kfib  == nfib -1) && (u[get_index3(ncol,nfib,irow,jcol,kfib)] != 0))
	       	++type_indicator;
            }
          
        }
    }

  switch (type_indicator)
    {
    case 0:
      return 1.0;
    case 1: 
      return 0.5;
    case 2: 
      return 0.25;
    case 3: 
      return 0.125;
    default:
      return 1.0;
    }
}


int  parse_cmdl(int argc, char**argv, int& nrow, int& ncol, int& nfib, double& tol, double& s, std::string& in_file, std::string& coord_file)
  {
    if ( argc >= 6 )
      {
        in_file    = argv[1];
        coord_file = argv[2];
        nrow =  strtol ((argv[3]), NULL, 0) ; //number of rows
        ncol =  strtol ((argv[4]), NULL, 0) ; // number of columns
        nfib =  strtol ((argv[5]), NULL, 0) ; // number of columns
        tol  =  strtod ((argv[6]), 0) ; // error tolerance
        tol  =  strtod ((argv[7]), 0) ; // error tolerance
	
        assert( in_file.size() != 0 );
        assert( ncol > 3  );
        assert( nrow >= 1 );
        assert( tol  >= 1e-8);

        struct stat file_stats;
        int flag = stat(in_file.c_str(), &file_stats);

        if( flag != 0 ) // can't stat file somehow
          {
            throw std::runtime_error("Cannot stat input file! Nothing to be done, exiting...");
          }

        return 1;
      }
    else
      {
        std::cerr << "Usage: " << argv[0] << " inputfile nrow ncol tol" <<"\n";
        throw std::runtime_error("Too few arguments, exiting...");
      }

  }


void set_level(const int nrow, const int ncol, const int nfib, int& nlevel)
  {
    int nlevel_x = std::log2(ncol-1);
    int nc = std::pow(2, nlevel_x ) + 1; //ncol new
  
    int nlevel_y = std::log2(nrow-1);
    int nr = std::pow(2, nlevel_y ) + 1 ; //nrow new
    
    int nlevel_z = std::log2(nfib-1);
    int nf = std::pow(2, nlevel_z ) + 1; //nfib new
    
    nlevel = std::min(nlevel_x, nlevel_y);
    nlevel = std::min(nlevel, nlevel_z);
    
    // std::cout << "Got: " << nlevel << " \t "<< tol <<"\n";
    // std::cout << "n?: " << nr << " \t "<< nc << "\t" << nf<<"\n";
  }


int main(int argc, char**argv)
{

  double tol, s, norm;
  int nrow, ncol, nfib, nlevel;
  std::string in_file, coord_file, out_file, zip_file;

  // -- get commandline params --//
  parse_cmdl(argc, argv, nrow, ncol, nfib, tol, s, in_file, coord_file);

  std::vector<double> v(nrow*ncol*nfib), coords_x(ncol), coords_y(nrow), coords_z(nfib);
  
  //-- set level hierarchy --//
  set_level( nrow,  ncol,  nfib,  nlevel);
  
  
  //-- read input file and set dummy coordinates --//
  std::ifstream infile(in_file, std::ios::in | std::ios::binary);
  std::ifstream cordfile(coord_file, std::ios::in | std::ios::binary);
  
  //  infile.read( reinterpret_cast<char*>( v.data() ), nrow*ncol*nfib*sizeof(double) );
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);
  

  //-- set and creat output files -- //
  out_file = in_file +   std::to_string(tol) + "_y.dat";
  zip_file = in_file +   std::to_string(tol) + ".gz";
  
  std::ofstream outfile(out_file, std::ios::out | std::ios::binary);
  std::ofstream zipfile(zip_file, std::ios::in | std::ios::binary);


  //-- call the compressor --//
  std::vector<unsigned char> out_data;
  int out_size;
      
  
  int l_target = nlevel-1;

  auto funp = &qoi_x ;  
  
  std::vector<double> norm_vec(nlevel+1);

  unsigned char* test;

  //  test = mgard_compress(1, v.data(), out_size,  nrow,  ncol,  nfib,  tol);
  //  test = mgard::refactor_qz_2D(nrow, ncol, v.data(), out_size, tol);

  //  free(test);


  double xnorm = mgard_compress(nrow,  ncol,  nfib,  funp, 0);


  std::cout << xnorm << "\n";
  return 0;
				     
  

 
}
