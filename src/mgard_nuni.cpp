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


#include "mgard.h"
#include "mgard_nuni.h"

namespace mgard_common
{

  int  parse_cmdl(int argc, char**argv, int& nrow, int& ncol, int& nfib, double& tol, double &s, std::string& in_file, std::string& coord_file)
  {
    if ( argc >= 7 )
      {
        in_file    = argv[1];
        coord_file = argv[2];
        nrow =  strtol ((argv[3]), NULL, 0) ; //number of rows
        ncol =  strtol ((argv[4]), NULL, 0) ; // number of columns
        nfib =  strtol ((argv[5]), NULL, 0) ; // number of columns
        tol  =  strtod ((argv[6]), 0) ; // error tolerance
        s  =  strtod ((argv[7]), 0) ; // norm to compress in

        assert( in_file.size() != 0 );
        assert( ncol > 3  );
        assert( nrow >= 1 );
	//        assert( tol  >= 1e-8);

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


  bool is_2kplus1(double num)
  {
    float frac_part, f_level, int_part;
  
    f_level = std::log2(num-1);
    frac_part = modff(f_level, &int_part);
  
    if( frac_part == 0)
      {
        return 1;
      }
    else
      {
        return 0;
      }
  }

  inline int get_index(const int ncol, const int i, const int j)
  {
    return ncol*i + j;
  }

  inline int get_index3(const int ncol, const int nfib, const int i, const int j, const int k)
  {
    return (ncol*i + j)*nfib + k;
  }


  double max_norm(const std::vector<double>& v)
  {
    double norm = 0;
    
    for (int i = 0; i < v.size(); ++i)
      {
        double ntest = std::abs(v[i]);
        if (ntest > norm) norm = ntest ;
      }
    return norm;
  }

  inline  double interp_1d(double x, double x1, double x2, double q00, double q01) {
  return ((x2 - x) / (x2 - x1)) * q00 + ((x - x1) / (x2 - x1)) * q01;
}

    
  inline double
  interp_2d(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y)
  {
    double x2x1, y2y1, x2x, y2y, yy1, xx1;
    x2x1 = x2 - x1;
    y2y1 = y2 - y1;
    x2x = x2 - x;
    y2y = y2 - y;
    yy1 = y - y1;
    xx1 = x - x1;
    return 1.0 / (x2x1 * y2y1) * (
                                  q11 * x2x * y2y +
                                  q21 * xx1 * y2y +
                                  q12 * x2x * yy1 +
                                  q22 * xx1 * yy1
                                  );
  }


  inline double
  interp_3d(double q000, double q100, double q110, double q010, double q001, double q101, double q111, double q011, double x1, double x2, double y1, double y2, double z1, double z2, double x, double y, double z)
  {
    
    double x00 = interp_1d(x, x1, x2, q000, q100);
    double x10 = interp_1d(x, x1, x2, q010, q110);
    double x01 = interp_1d(x, x1, x2, q001, q101);
    double x11 = interp_1d(x, x1, x2, q011, q111);
    double r0 =  interp_1d(y, y1, y2, x00, x01);
    double r1 =  interp_1d(y, y1, y2, x10, x11);
    
    return interp_1d(z, z1, z2, r0, r1);
    

  }
  
  
  inline double get_h(const std::vector<double>& coords, int i, int stride)
  {
    return (coords[i+stride] - coords[i]);
  }

  inline double get_dist(const std::vector<double>& coords, int i, int j)
  {
    return (coords[j] - coords[i]);
  }

  void qread_2D_interleave( const int nrow, const int ncol, const int nlevel, double* v, std::string infile)
  {
    int buff_size = 128*1024;
    unsigned char unzip_buffer[buff_size];
    int  int_buffer[buff_size/sizeof(int)];
    unsigned int  unzipped_bytes, total_bytes = 0;
    double coeff;

    gzFile in_file_z = gzopen(infile.c_str(), "r");
    std::cout << "File to oppen:" << in_file_z <<"\n";

    unzipped_bytes = gzread(in_file_z, unzip_buffer, sizeof(double)); //read the quantization constant
    std::memcpy(&coeff, &unzip_buffer, unzipped_bytes);
    std::cout << "Qunatar"<<coeff <<"\n";
    int last = 0;
    while (true) {
      unzipped_bytes = gzread(in_file_z, unzip_buffer, buff_size);
      //      std::cout << unzipped_bytes <<"\n";
      if (unzipped_bytes > 0) {
        total_bytes += unzipped_bytes;
        int num_int = unzipped_bytes/sizeof(int);

        std::memcpy(&int_buffer, &unzip_buffer, unzipped_bytes);
        for(int i = 0; i < num_int ; ++i)
          {
            v[last] =  double(int_buffer[i])*coeff;
            ++last;
          }

      } else {
        break;
      }
    }

    gzclose(in_file_z);
  }





  inline short encode(double x)
  {
    return  short (x * 32768   + (x >= 0 ? 0.0 : -1.0))  ;
  }


  inline double decode(short x)
  {
    return double (double (2*x + 1.0) / 65535.0);
  }



    int ma_quant(const int N, const double eps, const double u) {
	if (u < N*eps)
		return int (u/eps);
	else 
		return N + int (log(u/eps/N)/log((1.0+eps)/(1.0-eps))); 
}

  double ma_dequant(const int N, const double eps, const int k) {
	if (k<N)
		return (k+0.5)*eps; 
	else 
		return N*eps/(1.0-eps)*pow((1.0+eps)/(1.0-eps),k-N); 
}

  void qread_2D_bin( const int nrow, const int ncol, const int nlevel, double* v, std::string infile)
  {
    int buff_size = 128*1024;
    unsigned char unzip_buffer[buff_size];
    int  int_buffer[buff_size/sizeof(int)];
    unsigned int  unzipped_bytes, total_bytes = 0;
    double coeff;

    gzFile in_file_z = gzopen(infile.c_str(), "r");
    std::cout << infile <<"\n";

    unzipped_bytes = gzread(in_file_z, unzip_buffer, sizeof(double)); //read the quantization constant
    std::memcpy(&coeff, &unzip_buffer, unzipped_bytes);
    int N = (0.5/coeff + 0.5);     
    int last = 0;
    while (true) {
      unzipped_bytes = gzread(in_file_z, unzip_buffer, buff_size);
      //      std::cout << unzipped_bytes <<"\n";
      if (unzipped_bytes > 0) {
        total_bytes += unzipped_bytes;
        int num_int = unzipped_bytes/sizeof(int);

        std::memcpy(&int_buffer, &unzip_buffer, unzipped_bytes);
        for(int i = 0; i < num_int ; ++i)
          {
            int k = int_buffer[i];
            if(k < 0)
              {
                v[last] = -ma_dequant(N, coeff, -k);
              }
            else
              {
                v[last] = ma_dequant(N, coeff, k);
              }

            ++last;
          }

      } else {
        break;
      }
    }

    gzclose(in_file_z);
  }





  void qwrite_2D_bin( const int nrow, const int ncol, const int nlevel,  const int  l,   double* v, double tol, double norm, const std::string outfile)
  {

    
    double coeff = 2*norm*tol/(nlevel + 1);


    gzFile out_file = gzopen(outfile.c_str(), "w6b");
    int prune_count = 0;

    int quantum;
    
    gzwrite(out_file, &coeff, sizeof(double));
    
    int N = (0.5/coeff + 0.5);     

    std::cout << "Write::Quantization factor: "<<coeff << "\t"<< N <<"\n";

    for (auto index = 0; index < ncol*nrow; ++index )
      {
        if(v[index] < 0)
          {
            quantum = -ma_quant(N, coeff, -v[index]);
          }
        else
          {
            quantum = ma_quant(N, coeff, v[index]);          
          }

        
        if (quantum == 0) ++prune_count;
        gzwrite(out_file, &quantum, sizeof(int));
      }

    std::cout << "Pruned : "<< prune_count << " Reduction : " << (double) nrow*ncol /(nrow*ncol - prune_count) << "\n";
    gzclose(out_file);


  }
  
  void qwrite_2D_interleave( const int nrow, const int ncol, const int nlevel,  const int  l,   double* v, double tol, double norm, const std::string outfile)
  {

    int stride = std::pow(2,l);//current stride

    tol /=  (double) (nlevel + 1);

    double coeff = 2.0*norm*tol;
    std::cout << "Quantization factor: "<<coeff <<"\n";

    gzFile out_file = gzopen(outfile.c_str(), "w6b");
    int prune_count = 0;
    gzwrite(out_file, &coeff, sizeof(double));


    for (auto index = 0; index < ncol*nrow; ++index )
      {
        int quantum =  (int)(v[index]/coeff);
        if (quantum == 0) ++prune_count;
        gzwrite(out_file, &quantum, sizeof(int));
      }

    std::cout << "Pruned : "<< prune_count << " Reduction : " << (double) nrow*ncol /(nrow*ncol - prune_count) << "\n";
    gzclose(out_file);


  }


void qwrite_3D_interleave( const int nrow, const int ncol, const int nfib, const int nlevel,  const int  l,   double* v, double tol, double norm, const std::string outfile)
  {

    int stride = std::pow(2,l);//current stride

    tol /=  (double) (nlevel + 1);

    double coeff = 2.0*norm*tol;
    std::cout << "Quantization factor: "<<coeff <<"\n";

    gzFile out_file = gzopen(outfile.c_str(), "w6b");
    int prune_count = 0;
    gzwrite(out_file, &coeff, sizeof(double));


    for (auto index = 0; index < ncol*nrow*nfib; ++index )
      {
        int quantum =  (int)(v[index]/coeff);
        if (quantum == 0) ++prune_count;
        gzwrite(out_file, &quantum, sizeof(int));
      }

    std::cout << "Pruned : "<< prune_count << " Reduction : " << (double) nrow*ncol*nfib /(nrow*ncol*nfib - prune_count) << "\n";
    gzclose(out_file);


  }






  void qwrite_3D_interleave2( const int nrow, const int ncol, const int nfib, const int nlevel,  const int  l,   double* v, double tol, double norm, const std::string outfile)
  {

    int stride = std::pow(2,l);//current stride

    //    tol /=  (double) (nlevel + 1);

    double coeff = 1.0*norm*tol;
    std::cout << "Quantization factor: "<<coeff <<"\n";

    gzFile out_file = gzopen(outfile.c_str(), "w6b");
    int prune_count = 0;
    gzwrite(out_file, &coeff, sizeof(double));


    for (auto index = 0; index < ncol*nrow*nfib; ++index )
      {
        int quantum =  (int)(v[index]/coeff);
        if (quantum == 0) ++prune_count;
        gzwrite(out_file, &quantum, sizeof(int));
      }

    std::cout << "Pruned : "<< prune_count << " Reduction : " << (double) nrow*ncol*nfib /(nrow*ncol*nfib - prune_count) << "\n";
    gzclose(out_file);


  }
  

  void copy_slice(double* work, std::vector<double>&work2d, int nrow, int ncol, int nfib, int is)
  {
    for(int i = 0; i < nrow; ++i)
      {
        for(int j = 0; j < ncol; ++j)
          {
            work2d[get_index(ncol,i,j)] = work[get_index3(ncol, nfib, i, j, is)];
          }
      }
  }


  void copy_from_slice(double* work, std::vector<double>&work2d, int nrow, int ncol, int nfib, int is)
  {
    for(int i = 0; i < nrow; ++i)
      {
        for(int j = 0; j < ncol; ++j)
          {
             work[get_index3(ncol, nfib, i, j, is)] = work2d[get_index(ncol,i,j)] ;
          }
      }
  }
}


namespace mgard_cannon
{

  void assign_num_level(const int nrow, const int ncol,    const int  l, double* v, double num)
  {
    // set the value of nodal values at level l to number num


    int stride = std::pow(2,l);//current stride

    for(int irow = 0;  irow < nrow; irow += stride)
      {
        for(int jcol = 0;  jcol < ncol; jcol += stride)
          {
            v[mgard_common::get_index(ncol,irow, jcol)]  = num ;
          }
      }
  }

  
  void subtract_level(const int nrow, const int ncol,  const int  l, double* v, double* work)
  {
    // v += work at level l
    int stride = std::pow(2,l);//current stride

    for(int irow = 0;  irow < nrow; irow += stride)
      {
        for(int jcol = 0;  jcol < ncol; jcol += stride)
          {
            v[mgard_common::get_index(ncol,irow, jcol)] -= work[mgard_common::get_index(ncol,irow, jcol)];
          }
      }

  }

    void pi_lminus1(const int  l, std::vector<double>& v,  const std::vector<double>& coords)
  {
    int n = v.size();
    int nlevel = static_cast<int> (std::log2(v.size() - 1));
    int my_level = nlevel - l;
    int stride = std::pow(2,l);//current stride
    //  int Pstride = stride/2; //finer stride
    int Cstride = stride*2; // coarser stride
  
  
    if ( my_level != 0)
      {
        for( int i = Cstride; i < n ; i += Cstride)
          {
            double h1 = mgard_common::get_h(coords, i-Cstride, stride);
            double h2 = mgard_common::get_h(coords, i-stride , stride);
            double hsum = h1+h2;
            v[i - stride] -=  ( h1*v[i] + h2*v[i - Cstride])/hsum ;
          }
      }
  }

  void restrict(const int  l, std::vector<double>& v, const std::vector<double>& coords)
  {
    int stride = std::pow(2,l);
    int Pstride = stride/2;//finer stride
    int n = v.size();
    
    // calculate the result of restriction

    double h1 = mgard_common::get_h(coords, 0, Pstride);
    double h2 = mgard_common::get_h(coords, Pstride , Pstride);
    double hsum = h1+h2;                            
    
    v.front() += h2*v[Pstride]/hsum; //first element
    
    for(int i =  stride; i <= n - stride; i += stride)
      {
        v[i] += h1*v[i - Pstride]/hsum;
        h1 = mgard_common::get_h(coords, i, Pstride);
        h2 = mgard_common::get_h(coords, i+Pstride, Pstride);
        hsum = h1+h2;
        v[i] += h2*v[i + Pstride]/hsum;
      }
    v.back() +=  h1*v[n - Pstride - 1]/hsum; //last element
  }


  void prolongate(const int  l, std::vector<double>& v, const std::vector<double>& coords)
  {
  
    int stride = std::pow(2,l);
    int Pstride = stride/2;
    int n = v.size();
        
    for(  int i =  stride; i < n ; i += stride)
      {
        double h1 = mgard_common::get_h(coords, i-stride,  Pstride);
        double h2 = mgard_common::get_h(coords, i-Pstride, Pstride);
        
        v[i - Pstride] = (h2*v[i - stride] +  h1*v[i])/(h1+h2);
      }
  
  }



  void solve_tridiag_M(const int  l, std::vector<double>& v, const std::vector<double>& coords)
  {

    //  int my_level = nlevel - l;
    int stride = std::pow(2,l);//current stride

    double am, bm, h1, h2;
    int n = v.size();
    
    am = 2.0*mgard_common::get_h(coords, 0, stride) ; // first element of upper diagonal U.

    //    bm = mgard_common::get_h(coords, 0, stride) / am;
    bm = mgard_common::get_h(coords, 0, stride)/am;
    int nlevel = static_cast<int> (std::log2(v.size() - 1));
    //    std::cout << nlevel;
    int nc = std::pow(2,nlevel - l) + 1 ;
    std::vector<double> coeff(nc);
    int counter = 1;
    coeff.front() = am;

    //std::cout <<  am<< "\t"<< bm<<"\n";
    // forward sweep
    for(int i = stride; i < n - 1 ; i += stride)
      {
        h1 = mgard_common::get_h(coords, i-stride, stride);
        h2 = mgard_common::get_h(coords, i, stride);
        //        std::cout << i<< "\t"<< v[i-stride] << "\t" << h1<< "\t"<< h2<<"\n";
        v[i]  -= v[i-stride]*bm;


        am = 2.0*(h1+h2) - bm*h1;
        bm = h2 / am;
        //        std::cout <<  am<< "\t"<< bm<<"\n";

        coeff.at(counter) = am;
        ++counter;
      }


    h2 = mgard_common::get_h(coords, n-1-stride, stride);  
    am = 2.0*h2 - bm*h2;// a_n = 2 - b_(n-1)
    //    std::cout << h1 << "\t"<< h2<<"\n";
    v[n-1]  -= v[n-1-stride]*bm;

    coeff.at(counter) = am;

    //backward sweep

    v[n-1] /= am ;
    --counter;

    for(int i = n-1-stride; i >= 0 ; i -= stride)
      {
        //h1 = mgard_common::get_h(coords, i-stride, stride);
        h2 = mgard_common::get_h(coords, i, stride); 
        v[i] = (v[i] - h2*v[i+stride])/coeff.at(counter)  ;
        --counter;
        //        bm = (2.0*(h1+h2) - am) / h1 ;
        //am = 1.0 / bm;
      }
    //h1 = mgard_common::get_h(coords, 0, stride);
    //    std::cout << h1 << "\n";
    //    v[0] = (v[0] - h1*v[1])/coeff[0];
  }


    void mass_matrix_multiply(const int  l, std::vector<double>& v, const std::vector<double>& coords)
  {

    int stride = std::pow(2,l);
    int n = v.size();
    double temp1, temp2;

    // Mass matrix times nodal value-vec
    temp1 = v.front(); //save u(0) for later use
    v.front() = 2.0*mgard_common::get_h(coords, 0, stride)*temp1 +  mgard_common::get_h(coords, 0, stride)*v[stride];
    for(int i = stride;  i <= n - 1 - stride; i += stride )
      {
        temp2 = v[i];
        v[i] = mgard_common::get_h(coords, i-stride, stride)*temp1 + 2*(mgard_common::get_h(coords, i-stride, stride) + mgard_common::get_h(coords, i, stride))*temp2 + mgard_common::get_h(coords, i, stride)*v[i+stride];
        temp1 = temp2; // save u(n) for later use
      }
    v[n-1] = mgard_common::get_h(coords, n-stride-1, stride) * temp1 + 2*mgard_common::get_h(coords, n-stride-1, stride)*v[n-1] ;
  }

   void write_level_2D( const int nrow, const int ncol, const int  l,   double* v, std::ofstream& outfile)
  {
    int stride = std::pow(2,l);
    //  int nrow = std::pow(2, nlevel_row) + 1;
    // int ncol = std::pow(2, nlevel_col) + 1;


    for (int irow = 0; irow < nrow; irow += stride)
      {
        for(int jcol = 0;  jcol < ncol; jcol += stride)
          {
            outfile.write(reinterpret_cast<char*>( &v[mgard_common::get_index(ncol, irow, jcol)]), sizeof(double) );
          }
      }

  }

  void copy_level(const int nrow, const int ncol,  const int  l, double* v, std::vector<double>& work)
  {

    int stride = std::pow(2,l);//current stride

    for(int irow = 0;  irow < nrow; irow += stride)
      {
        for(int jcol = 0;  jcol < ncol; jcol += stride)
          {
            work[mgard_common::get_index(ncol,irow, jcol)]  =  v[mgard_common::get_index(ncol,irow, jcol)] ;
          }
      }

  }


  void copy_level3(const int nrow, const int ncol,  const int nfib, const int  l, double* v, std::vector<double>& work)
  {

    int stride = std::pow(2,l);//current stride

    for(int irow = 0;  irow < nrow; irow += stride)
      {
        for(int jcol = 0;  jcol < ncol; jcol += stride)
          {
            for(int kfib = 0;  kfib < nfib; kfib += stride)
              {
                work[mgard_common::get_index3(ncol,nfib, irow, jcol, kfib)]  =  v[mgard_common::get_index3(ncol,nfib, irow, jcol, kfib)] ;
              }
          }
        
      }
  }
}

namespace mgard_gen
{
  inline  double* get_ref(std::vector<double>& v,  const int n, const int no, const int i) //return reference to logical element
  {
    // no: original number of points
    // n : number of points at next coarser level (L-1) with  2^k+1 nodes
    // may not work for the last element!
    if(i != n-1)
      {
        return &v[floor(( (double) no-2.0)/( (double) n-2.0)*i)];        
      }
    else if( i == n-1 )
      {
        return &v[no-1];
      }

    //    return &v[floor(((no-2)/(n-2))*i ) ];
  }
  
  
  inline  int get_lindex(const int n, const int no, const int i)
  {
    // no: original number of points
    // n : number of points at next coarser level (L-1) with  2^k+1 nodes
    
    //    return floor((no-2)/(n-2)*i);
    if(i != n-1)
      {
        return  floor(( (double) no-2.0)/( (double) n-2.0)*i);
      }
    else if ( i == n-1)
      {
        return no-1;
      }
  }
  
  inline  double get_h_l(const std::vector<double>& coords, const int n, const int no, int i, int stride)
  {
    
    //    return (*get_ref(coords, n, no, i+stride) - *get_ref(coords, n, no, i));
    return (coords[get_lindex(n, no, i+stride)] - coords[get_lindex(n, no, i)]);
  }


  double l2_norm(const int  l,  const int n, const int no,  std::vector<double>& v, const std::vector<double>& x)
  {
    int stride = std::pow(2,l);


    double norm = 0;
    for(int i = 0; i < n-stride ; i += stride )
      {
        int ir  =   get_lindex(n,  no,  i);
        int irP =   get_lindex(n,  no,  i+stride);
        double h = x[irP] - x[ir];
        double temp = 0.5*(v[ir] +  v[irP]);
        norm += h*(v[ir]*v[ir] + v[irP]*v[irP] + 4.0*temp*temp);
        
      }
  
    norm /= 6.0;
    //    std::cout << norm << "\t";
    return norm;
  }

  double l2_norm2(const int  l, int nr, int nc, int nrow, int ncol,  std::vector<double>& v, const std::vector<double>& coords_x, const std::vector<double>& coords_y)
  {
    std::vector<double> row_vec(ncol), col_vec(nrow);
    double result;
    int stride = std::pow(2,l);
    for(int irow = 0;  irow < nr; irow += stride)
          {
            int ir = get_lindex(nr, nrow, irow);
            for(int jcol = 0; jcol < ncol; ++jcol)
              {
                row_vec[jcol] = v[mgard_common::get_index(ncol, ir, jcol)];
              }

            double temp =  l2_norm(l, nc, ncol, row_vec, coords_x);
            //std::cout << temp << "\t";

            col_vec[ir] = temp;
          }

    result =  l2_norm(l, nr, nrow, col_vec, coords_y);

    return result;
  }

  double l2_norm3(const int  l, int nr, int nc, int nf, int nrow, int ncol, int nfib,  std::vector<double>& v, const std::vector<double>& coords_x, const std::vector<double>& coords_y, const std::vector<double>& coords_z)
  {
    std::vector<double> work2d(nrow*ncol), fib_vec(nfib);
    double result;
    int stride = std::pow(2,l);
    for(int  kfib = 0;  kfib < nf; kfib += stride)
      {
        int kf = get_lindex(nf, nfib, kfib);
        mgard_common::copy_slice(v.data(), work2d, nrow, ncol, nfib, kf);
        double temp = l2_norm2(l,  nr,  nc,  nrow,  ncol, work2d, coords_x, coords_y);  
        fib_vec[kf] = temp;
      }

    result =  l2_norm(l, nf, nfib, fib_vec, coords_z);
    std::cout << result << "\t";

    return result;
  }


  
  void write_level_2D_l(const int  l,   double* v, std::ofstream& outfile, int nr, int nc, int nrow, int ncol)
  {
    int stride = std::pow(2,l);
    //  int nrow = std::pow(2, nlevel_row) + 1;
    // int ncol = std::pow(2, nlevel_col) + 1;


    for (int irow = 0; irow < nr; irow += stride)
      {
        int ir  = get_lindex(nr,  nrow,  irow);
        for(int jcol = 0;  jcol < nc; jcol += stride)
          {
            int jr  = get_lindex(nc,  ncol,  jcol);
            outfile.write(reinterpret_cast<char*>( &v[mgard_common::get_index(ncol, ir, jr)]), sizeof(double) );
          }
      }

  }
  // void assign_num_level_l(const int  l, double* v, double num, int nr, int nc, int nrow, int ncol)
  // {
  //   // logical assign number to level
  //   int stride = std::pow(2,l);//current stride

  //   for(int irow = 0;  irow < nr; irow += stride)
  //     {
  //       int ir  = get_lindex(nr,  nrow,  irow);
  //       for(int jcol = 0;  jcol < nc; jcol += stride)
  //         {
  //           int jr  = get_lindex(nc,  ncol,  jcol);
  //           v[mgard_common::get_index(ncol,ir, jr)]  = num ;
  //         }
  //     }

  // }

  void qwrite_3D(const int nr, const int nc, const int nf, const int nrow, const int ncol, const int nfib, const int nlevel,  const int  l,   double* v, const std::vector<double>& coords_x, const std::vector<double>& coords_y, const std::vector<double>& coords_z, double tol, double s, double norm, const std::string outfile)
  {

    
    //    double coeff = 2.0*norm*tol;
    //    double coeff = norm*tol/216;
    double coeff = tol; ///4.322;// 
    std::cout << "Quantization factor: "<<coeff <<"\n";

    gzFile out_file = gzopen(outfile.c_str(), "w6b");
    int prune_count = 0;
    gzwrite(out_file, &coeff, sizeof(double));

    //    double s = 0.0;
    int count = 0;

     //level -1, first level for non 2^k+1

    double dx = mgard_gen::get_h_l(coords_x, ncol, ncol, 0, 1);
    double dy = mgard_gen::get_h_l(coords_y, nrow, nrow, 0, 1);
    double dz = mgard_gen::get_h_l(coords_z, nfib, nfib, 0, 1);
    
    double vol =  std::sqrt(dx*dy*dz);
    vol *= std::pow(2, s*(nlevel+1)); //2^-2sl with l=0, s = 0.5
    //    vol = 1;
    //    std::cout << "Volume -1: " << vol << std::endl;
    
    for(int kfib = 0; kfib < nf - 1; ++kfib)
      {
        int kf  = mgard_gen::get_lindex(nf, nfib, kfib);
        int kfp = mgard_gen::get_lindex(nf, nfib, kfib + 1);
        
        if( kfp != kf + 1)//skipped a plane
          {
            for(int irow = 0;  irow < nrow; ++irow )
              {
                for(int jcol = 0;  jcol < ncol; ++jcol )
                  {
                    double val = v[mgard_common::get_index3(ncol, nfib, irow, jcol, kf + 1)];
                    int quantum =  (int)(val/(coeff/vol));
                    gzwrite(out_file, &quantum, sizeof(int));
                    ++count;
                  }
              }
          }
      }

    
    int count_row = 0;
    int count_col = 0;
    int count_sol = 0;
    

    for (int kfib = 0; kfib < nf ; ++kfib)
      {
        int kf  = mgard_gen::get_lindex(nf, nfib, kfib);
        for(int irow = 0;  irow < nr - 1 ; ++irow )
          {
            int ir = mgard_gen::get_lindex(nr, nrow, irow);
            int irP = mgard_gen::get_lindex(nr, nrow, irow + 1);
            if( irP != ir + 1) //skipped a row
              {
                //  std::cout <<"Skipped row: "  << ir + 1 << "\n";
                for(int jcol = 0;  jcol < ncol; ++jcol )
                  {
                    double val = v[mgard_common::get_index3(ncol, nfib, ir + 1, jcol, kf)];
                    int quantum =  (int)(val/(coeff/vol));
                    gzwrite(out_file, &quantum, sizeof(int));
                    ++count_row;
                    ++count;
                  }
              }
          }
        
        for(int irow = 0;  irow < nr  ; ++irow )
          {
            int ir = mgard_gen::get_lindex(nr, nrow, irow);
            
            //      std::cout <<"Non skipped row: "  << ir  << "\n";
            for(int jcol = 0;  jcol < nc - 1 ; ++jcol)
              {
                int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                int jcP  = mgard_gen::get_lindex(nc,  ncol,  jcol+1);
                if(jcP != jc + 1)//skipped a column
                  {
                    double val = v[mgard_common::get_index3(ncol, nfib, ir, jc + 1, kf)];
                    int quantum =  (int)(val/(coeff/vol));
                    gzwrite(out_file, &quantum, sizeof(int));
                    ++count_col;
                    ++count;
                    //                    std::cout <<"Skipped col: " << ir << "\t" << jc + 1 << "\n";
                  }
              }
          }
      }

    //std::cout << "Wrote : "<< count_row <<"\t" << count_col << "\t" << count_sol << "\n";
    

    
    // // 2^k+1 part //
    
    for(int ilevel = 0; ilevel < nlevel ; ++ilevel)
      {
        int stride = std::pow(2,ilevel);
        int Cstride = 2*stride;

        int fib_counter = 0;

        double dx = get_h_l(coords_x, nc, ncol, 0, stride);
        double dy = get_h_l(coords_y, nr, nrow, 0, stride);
        double dz = get_h_l(coords_z, nf, nfib, 0, stride);
        
        double vol =  std::sqrt(dx*dy*dz);
	vol *= std::pow(2, s*(nlevel-ilevel)); //2^-2sl with l=0, s = 0.5
        std::cout << "Volume : " << ilevel << "\t"<< vol << std::endl;
        // std::cout << "Stride : " << stride << "\t"<< vol << std::endl;
        
        for(int kfib = 0; kfib < nf; kfib += stride)
          {
            int kf = mgard_gen::get_lindex(nf, nfib, kfib);
            int row_counter = 0;

            if(fib_counter % 2 == 0)
              {
              for(int irow = 0;  irow < nr; irow += stride)
                {
                  int ir = mgard_gen::get_lindex(nr, nrow, irow);
                  if( row_counter % 2 == 0)
                    {
                      for(int jcol = Cstride;  jcol < nc; jcol += Cstride)
                        {
                          int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                          double val = v[mgard_common::get_index3(ncol, nfib, ir,jc - stride, kf)];
                          int quantum =  (int)(val/(coeff/vol));
                          gzwrite(out_file, &quantum, sizeof(int));
                          ++count;
                          //                          outfile.write(reinterpret_cast<char*>( &v[mgard_common::get_index3(ncol, nfib, ir,jc - stride, kf)] ), sizeof(double) ); 
                        //                  std::cout <<  v[irow][icol - stride] << "\t";
                        }
                    
                    }else
                    {
                      for(int jcol = 0;  jcol < nc; jcol += stride)
                        {
                          int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                          double val = v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)];
                          int quantum =  (int)(val/(coeff/vol));
                          gzwrite(out_file, &quantum, sizeof(int));
                          ++count;
                          //         outfile.write(reinterpret_cast<char*>( &v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)] ), sizeof(double) ); 
                    //                  std::cout <<  v[irow][icol] << "\t";
                        }
                
                    }
                  ++row_counter;
                }
              }
            else{
              for(int irow = 0;  irow < nr; irow += stride)
                {
                  int ir = mgard_gen::get_lindex(nr, nrow, irow);
                  for(int jcol = 0;  jcol < nc; jcol += stride) 
                    {
                      int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                      double val = v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)];
                      int quantum =  (int)(val/(coeff/vol));
                      gzwrite(out_file, &quantum, sizeof(int));
                      ++count;
                      //                      outfile.write(reinterpret_cast<char*>( &v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)] ), sizeof(double) );
                    }
                }
            }
            ++fib_counter;
          }
      }
        


    // last level -> L=0
    int stride = std::pow(2,nlevel);
    dx = get_h_l(coords_x, nc, ncol, 0, stride);
    dy = get_h_l(coords_y, nr, nrow, 0, stride);
    dz = get_h_l(coords_z, nf, nfib, 0, stride);
    
    vol =  std::sqrt(dx*dy*dz);
    //    vol *= std::pow(2, 0);
    // std::cout << "Volume : " << nlevel << "\t"<< vol << std::endl;
    // std::cout << "Stride : " << stride << "\t"<< vol << std::endl;
    
    for(int irow = 0;  irow < nr; irow += stride)
      {
        int ir = mgard_gen::get_lindex(nr, nrow, irow);
        for(int jcol = 0;  jcol < nc; jcol += stride) 
          {
            int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
            for(int kfib = 0; kfib < nf; kfib += stride)
              {
                int kf = mgard_gen::get_lindex(nf, nfib, kfib);
                double val = v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)];
                int quantum =  (int)(val/(coeff/vol));
                gzwrite(out_file, &quantum, sizeof(int));
                ++count;
              }
          }
      }
    


    // std::cout << "Pruned : "<< prune_count << " Reduction : " << (double) nrow*ncol*nfib /(nrow*ncol*nfib - prune_count) << "\n";
    std::cout << "Wrote : "<< count << "\n";
    gzclose(out_file);


  }


  void dequant_3D(const int nr, const int nc, const int nf, const int nrow, const int ncol, const int nfib, const int nlevel,  const int  l,   double* v, double* work , const std::vector<double>& coords_x, const std::vector<double>& coords_y, const std::vector<double>& coords_z, double s)
  {

    //    double s = 0.0;

     //level -1, first level for non 2^k+1

    double dx = mgard_gen::get_h_l(coords_x, ncol, ncol, 0, 1);
    double dy = mgard_gen::get_h_l(coords_y, nrow, nrow, 0, 1);
    double dz = mgard_gen::get_h_l(coords_z, nfib, nfib, 0, 1);
    
    double vol =  std::sqrt(dx*dy*dz); 
    vol *= std::pow(2, s*(nlevel+1)); //2^-2sl with l=0, s = 0.5
    //    std::cout << "Volume -1: " << vol << std::endl;

    int imeg = 0 ; //mega-counter
    
    for(int kfib = 0; kfib < nf - 1; ++kfib)
      {
        int kf  = mgard_gen::get_lindex(nf, nfib, kfib);
        int kfp = mgard_gen::get_lindex(nf, nfib, kfib + 1);
        
        if( kfp != kf + 1)//skipped a plane
          {
            for(int irow = 0;  irow < nrow; ++irow )
              {
                for(int jcol = 0;  jcol < ncol; ++jcol )
                  {
                    v[mgard_common::get_index3(ncol, nfib, irow, jcol, kf + 1)] = work[imeg]/vol ;
                    ++imeg;
                   
                  }
              }
          }
      }

    
    int count_row = 0;
    int count_col = 0;
    int count_sol = 0;
    

    for (int kfib = 0; kfib < nf ; ++kfib)
      {
        int kf  = mgard_gen::get_lindex(nf, nfib, kfib);
        for(int irow = 0;  irow < nr - 1 ; ++irow )
          {
            int ir = mgard_gen::get_lindex(nr, nrow, irow);
            int irP = mgard_gen::get_lindex(nr, nrow, irow + 1);
            if( irP != ir + 1) //skipped a row
              {
                //  std::cout <<"Skipped row: "  << ir + 1 << "\n";
                for(int jcol = 0;  jcol < ncol; ++jcol )
                  {
                    v[mgard_common::get_index3(ncol, nfib, ir + 1, jcol, kf)] = work[imeg]/vol ;
                    ++imeg; 
                  }
              }
          }
        
        for(int irow = 0;  irow < nr  ; ++irow )
          {
            int ir = mgard_gen::get_lindex(nr, nrow, irow);
            
            //      std::cout <<"Non skipped row: "  << ir  << "\n";
            for(int jcol = 0;  jcol < nc - 1 ; ++jcol)
              {
                int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                int jcP  = mgard_gen::get_lindex(nc,  ncol,  jcol+1);
                if(jcP != jc + 1)//skipped a column
                  {
                    v[mgard_common::get_index3(ncol, nfib, ir, jc + 1, kf)] = work[imeg]/vol ;
                    ++imeg;

                  }
              }
          }
      }

    //std::cout << "Wrote : "<< count_row <<"\t" << count_col << "\t" << count_sol << "\n";
    

    
    // // 2^k+1 part //
    
    for(int ilevel = 0; ilevel < nlevel ; ++ilevel)
      {
        int stride = std::pow(2,ilevel);
        int Cstride = 2*stride;

        int fib_counter = 0;

        double dx = get_h_l(coords_x, nc, ncol, 0, stride);
        double dy = get_h_l(coords_y, nr, nrow, 0, stride);
        double dz = get_h_l(coords_z, nf, nfib, 0, stride);
        
        double vol =  std::sqrt(dx*dy*dz);
	vol *= std::pow(2, s*(nlevel-ilevel)); //2^-2sl with l=0, s = 0.5
        // std::cout << "Volume : " << ilevel << "\t"<< vol << std::endl;
        // std::cout << "Stride : " << stride << "\t"<< vol << std::endl;
        
        for(int kfib = 0; kfib < nf; kfib += stride)
          {
            int kf = mgard_gen::get_lindex(nf, nfib, kfib);
            int row_counter = 0;

            if(fib_counter % 2 == 0)
              {
              for(int irow = 0;  irow < nr; irow += stride)
                {
                  int ir = mgard_gen::get_lindex(nr, nrow, irow);
                  if( row_counter % 2 == 0)
                    {
                      for(int jcol = Cstride;  jcol < nc; jcol += Cstride)
                        {
                          int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                          v[mgard_common::get_index3(ncol, nfib, ir,jc - stride, kf)] = work[imeg]/vol ;
                          ++imeg; ;
                        }
                    
                    }else
                    {
                      for(int jcol = 0;  jcol < nc; jcol += stride)
                        {
                          int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                          v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)] = work[imeg]/vol ;
                          ++imeg; ;;
                        
                        }
                
                    }
                  ++row_counter;
                }
              }
            else{
              for(int irow = 0;  irow < nr; irow += stride)
                {
                  int ir = mgard_gen::get_lindex(nr, nrow, irow);
                  for(int jcol = 0;  jcol < nc; jcol += stride) 
                    {
                      int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                      v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)] = work[imeg]/vol ;
                          ++imeg; ;;
                    }
                }
            }
            ++fib_counter;
          }
      }
        


    // last level
    int stride = std::pow(2,nlevel);
    dx = get_h_l(coords_x, nc, ncol, 0, stride);
    dy = get_h_l(coords_y, nr, nrow, 0, stride);
    dz = get_h_l(coords_z, nf, nfib, 0, stride);
    
    vol =  std::sqrt(dx*dy*dz);
    std::cout << "Volume : " << nlevel << "\t"<< vol << std::endl;
    // std::cout << "Stride : " << stride << "\t"<< vol << std::endl;
    
    for(int irow = 0;  irow < nr; irow += stride)
      {
        int ir = mgard_gen::get_lindex(nr, nrow, irow);
        for(int jcol = 0;  jcol < nc; jcol += stride) 
          {
            int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
            for(int kfib = 0; kfib < nf; kfib += stride)
              {
                int kf = mgard_gen::get_lindex(nf, nfib, kfib);
                v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)] = work[imeg]/vol ;
                ++imeg; ;;
              }
          }
      }
    


    // std::cout << "Pruned : "<< prune_count << " Reduction : " << (double) nrow*ncol*nfib /(nrow*ncol*nfib - prune_count) << "\n";
    std::cout << "Mega count : "<< imeg << "\n";
  }


  
  void copy_level_l(const int  l, double* v, double* work, int nr, int nc, int nrow, int ncol)
  {
    // work_l = v_l
    int stride = std::pow(2,l);//current stride

    for(int irow = 0;  irow < nr; irow += stride)
      {
        int ir  = get_lindex(nr,  nrow,  irow);
        for(int jcol = 0;  jcol < nc; jcol += stride)
          {
            int jr  = get_lindex(nc,  ncol,  jcol);
            work[mgard_common::get_index(ncol,ir, jr)]  =  v[mgard_common::get_index(ncol,ir, jr)] ;
          }
      }

  }

    void subtract_level_l(const int  l, double* v, double* work, int nr, int nc, int nrow, int ncol)
  {
    // v -= work at level l


    int stride = std::pow(2,l);//current stride


    for(int irow = 0;  irow < nr; irow += stride)
      {
        int ir  = get_lindex(nr,  nrow,  irow);
        for(int jcol = 0;  jcol < nc; jcol += stride)
          {
            int jr  = get_lindex(nc,  ncol,  jcol);
            v[mgard_common::get_index(ncol,ir, jr)]  -= work[mgard_common::get_index(ncol,ir, jr)];
          }
      }

  }
  
  // void pi_lminus1_l(const int  l, std::vector<double>& v, const  std::vector<double>& coords, int n, int no)
  // {
  //   int nlevel = static_cast<int> (std::log2(n - 1));
  //   int my_level = nlevel - l;
  //   int stride = std::pow(2,l);//current stride
  //   //  int Pstride = stride/2; //finer stride
  //   int Cstride = stride*2; // coarser stride
  
  
  //   if ( my_level != 0)
  //     {
  //       for( int i = Cstride; i < n ; i += Cstride)
  //         {
  //           double h1 = get_h_l(coords, n, no, i-Cstride, stride);
  //           double h2 = get_h_l(coords, n, no, i-stride , stride);
  //           double hsum = h1+h2;
  //           v[i - stride] -=  ( h1*v[i] + h2*v[i - Cstride])/hsum ;
  //         }

  //     }
  // }

  void pi_lminus1_l(const int  l, std::vector<double>& v, const std::vector<double>& coords, int n, int no)
  {
  int nlevel = static_cast<int> (std::log2(n - 1));
  int my_level = nlevel - l;
  int stride = std::pow(2,l);//current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride*2; // coarser stride
  
  
  if ( my_level != 0)
    {
      for( int i = Cstride; i < n-1 ; i += Cstride)
        {
          double h1 = get_h_l(coords, n, no, i-Cstride, stride);
          double h2 = get_h_l(coords, n, no, i-stride , stride);
          double hsum = h1+h2;
          *get_ref(v, n, no, i-stride) -= (h1 *(*get_ref(v, n, no, i)) + h2*(*get_ref(v, n, no, i-Cstride)) )/hsum;
        }

      double h1 = get_h_l(coords, n, no, n-1-Cstride, stride);
      double h2 = get_h_l(coords, n, no, n-1-stride , stride);
      double hsum = h1+h2;
      *get_ref(v, n, no, n-1-stride) -= (h1 *(v.back()) + h2*(*get_ref(v, n, no, n-1-Cstride)) )/hsum;
    }
}



  
  void pi_lminus1_first( std::vector<double>& v,  const std::vector<double>& coords, int n, int no)
  {

    for( int i = 0; i < n-1 ; ++i)
      {
        int i_logic  = get_lindex(n,  no,  i);
        int i_logicP = get_lindex(n,  no, i+1);

        if (i_logicP != i_logic+1)
          {
            //          std::cout << i_logic +1 << "\t" << i_logicP<<"\n";
            double h1 = mgard_common::get_dist(coords, i_logic,  i_logic+1);
            double h2 = mgard_common::get_dist(coords, i_logic + 1, i_logicP);
            double hsum = h1+h2;           
            v[i_logic + 1] -= (h2*v[i_logic] + h1*v[i_logicP] )/hsum;          
          }


      }
  }
  
  void pi_Ql_first(const int nr, const int nc, const int nrow, const int ncol,  const int  l, double* v,  const std::vector<double>& coords_x, const std::vector<double>& coords_y, std::vector<double>& row_vec, std::vector<double>& col_vec)
  {
    // Restrict data to coarser level

    int stride = 1;//current stride
    //  int Pstride = stride/2; //finer stride
    int Cstride = 2; // coarser stride

    
    for(int irow = 0;  irow < nr; irow += stride) // Do the rows existing  in the coarser level
      {
        int irow_r = get_lindex(nr, nrow, irow); // get the real location of logical index irow

        for (int jcol = 0; jcol < ncol; ++jcol)
          {
            // int jcol_r = get_lindex(nc, ncol, jcol);
            // std::cerr << irow_r << "\t"<< jcol_r << "\n";

            row_vec[jcol] = v[mgard_common::get_index(ncol, irow_r, jcol)];
          }

        pi_lminus1_first(row_vec, coords_x, nc, ncol);

        for (int jcol = 0; jcol < ncol; ++jcol)
          {
            //            int jcol_r = get_lindex(nc, ncol, jcol);
            v[mgard_common::get_index(ncol, irow_r, jcol)] = row_vec[jcol] ;
          }

        // if( irP != ir +1) //are we skipping the next row?
        //   {
        //     ++irow;
        //   }

      }

    if (nrow > 1)
      {
        for(int jcol = 0;  jcol < nc; jcol += stride) // Do the columns existing  in the coarser level
          {
            int jcol_r = get_lindex(nc, ncol, jcol);
            int jr  = get_lindex(nc, ncol, jcol);
            int jrP = get_lindex(nc, ncol, jcol+1); 

            for(int irow = 0;  irow < nrow; ++irow)
              {
                int irow_r = get_lindex(nr, nrow, irow);
                col_vec[irow] = v[mgard_common::get_index(ncol, irow, jcol_r)];
              }

            pi_lminus1_first(col_vec, coords_y, nr, nrow);

            for(int irow = 0;  irow < nrow; ++irow)
              {
                int irow_r = get_lindex(nr, nrow, irow);
                v[mgard_common::get_index(ncol, irow, jcol_r)] = col_vec[irow];
              }

            // if( jrP != jr +1) //are we skipping the next row?
            //   {
            //     ++jcol;
            //   }
          }
      }

    //        Now the new-new stuff
    for(int irow = 0;  irow < nr-1; ++irow) 
      {
        int ir  = get_lindex(nr,  nrow,  irow);
        int irP = get_lindex(nr,  nrow,  irow+1);
        
        for(int jcol = 0;  jcol < nc-1; ++jcol) 
          {
            int jr  = get_lindex(nc, ncol, jcol);
            int jrP = get_lindex(nc, ncol, jcol+1); 
            
            if ((irP != ir+1) && (jrP != jr+1)) //we skipped both a row and a column
              {
                
                double q11 =  v[mgard_common::get_index(ncol, ir, jr)];
                double q12 =  v[mgard_common::get_index(ncol, irP, jr)];
                double q21 =  v[mgard_common::get_index(ncol, ir, jrP)];
                double q22 =  v[mgard_common::get_index(ncol, irP, jrP)];
                
                
                double x1 = 0.0;
                double y1 = 0.0;
                
                double x2 =  mgard_common::get_dist(coords_x,  jr,  jrP);
                double y2 =  mgard_common::get_dist(coords_y,  ir,  irP);
                
                double x  =  mgard_common::get_dist(coords_x,  jr,  jr+1);
                double y  =  mgard_common::get_dist(coords_y,  ir,  ir+1);
                
                
                double temp = mgard_common::interp_2d( q11,  q12,  q21,  q22,  x1,  x2,  y1,  y2,  x,  y) ;
                
                
                v[mgard_common::get_index(ncol, ir+1, jr+1)] -= temp;
              }

           }
      }
  }
  void pi_Ql(const int nr, const int nc, const int nrow, const int ncol,  const int  l, double* v,  const std::vector<double>& coords_x, const std::vector<double>& coords_y, std::vector<double>& row_vec, std::vector<double>& col_vec)
  {
    // Restrict data to coarser level

    int stride = std::pow(2,l);//current stride
    //  int Pstride = stride/2; //finer stride
    int Cstride = stride*2; // coarser stride

    //  std::vector<double> row_vec(ncol), col_vec(nrow)   ;

    for(int irow = 0;  irow < nr; irow += Cstride) // Do the rows existing  in the coarser level
      {
        int ir = get_lindex(nr, nrow, irow); // get the real location of logical index irow
        for (int jcol = 0; jcol < ncol; ++jcol)
          {
            //            int jcol_r = get_lindex(nc, ncol, jcol);  
            row_vec[jcol] = v[mgard_common::get_index(ncol, ir, jcol)];
          }

        //        mgard_cannon::pi_lminus1(l, row_vec, coords_x);
        pi_lminus1_l(l, row_vec, coords_x, nc,  ncol);
        for (int jcol = 0; jcol < ncol; ++jcol)
          {
            v[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol] ;
          }

      }

    if (nrow > 1)
      {
        for(int jcol = 0;  jcol < nc; jcol += Cstride) // Do the columns existing  in the coarser level
          {
            int jr = get_lindex(nc, ncol, jcol);  
            for(int irow = 0;  irow < nrow; ++irow)
              {
                //                int irow_r = get_lindex(nr, nrow, irow);
                col_vec[irow] = v[mgard_common::get_index(ncol, irow, jr)];
              }

            pi_lminus1_l(l, col_vec, coords_y, nr,  nrow);
            for(int irow = 0;  irow < nrow; ++irow)
              {
                //                int irow_r = get_lindex(nr, nrow, irow);
                v[mgard_common::get_index(ncol, irow, jr)] = col_vec[irow];
              }

          }


        // Now the new-new stuff
        for(int irow = stride;  irow < nr; irow += Cstride) 
          {
            int ir1 = get_lindex(nr, nrow, irow-stride);
            int ir  = get_lindex(nr, nrow, irow);
            int ir2 = get_lindex(nr, nrow, irow+stride); 

            for(int jcol = stride;  jcol < nc; jcol += Cstride) 
              {

                int jr1 = get_lindex(nc, ncol, jcol-stride);
                int jr = get_lindex(nc, ncol, jcol); 
                int jr2 = get_lindex(nc, ncol, jcol+stride); 

                
                double q11 =  v[mgard_common::get_index(ncol, ir1, jr1)];
                double q12 =  v[mgard_common::get_index(ncol, ir2, jr1)];
                double q21 =  v[mgard_common::get_index(ncol, ir1, jr2)];
                double q22 =  v[mgard_common::get_index(ncol, ir2, jr2)];
              
                double x1 =  0.0; //relative coordinate axis centered at irow - Cstride, jcol - Cstride
                double y1 =  0.0;
                double x2 =  mgard_common::get_dist(coords_x,  jr1,  jr2);
                double y2 =  mgard_common::get_dist(coords_y,  ir1,  ir2); 
              
                double x = mgard_common::get_dist(coords_x,  jr1,  jr); 
                double y = mgard_common::get_dist(coords_y,  ir1,  ir); 
                double temp = mgard_common::interp_2d( q11,  q12,  q21,  q22,  x1,  x2,  y1,  y2,  x,  y) ;
                //              std::cout << temp <<"\n";
                v[mgard_common::get_index(ncol, ir, jr)] -= temp; 
              }
          }
      }

  }

void pi_Ql3D(const int nr, const int nc, const int nf, const int nrow, const int ncol, const int nfib, const int  l, double* v,  const std::vector<double>& coords_x, const std::vector<double>& coords_y, const std::vector<double>& coords_z, std::vector<double>& row_vec, std::vector<double>& col_vec, std::vector<double>& fib_vec)
  {

    int stride = std::pow(2,l);//current stride
    //  int Pstride = stride/2; //finer stride
    int Cstride = stride*2; // coarser stride

    //  std::vector<double> row_vec(ncol), col_vec(nrow)   ;

    for(int kfib = 0; kfib < nf; kfib += Cstride)
      {
        int kf = get_lindex(nf, nfib, kfib); // get the real location of logical index irow
        for(int irow = 0;  irow < nr; irow += Cstride) // Do the rows existing  in the coarser level
          {
            int ir = get_lindex(nr, nrow, irow); // get the real location of logical index irow
            for (int jcol = 0; jcol < ncol; ++jcol)
              {
                row_vec[jcol] = v[mgard_common::get_index3(ncol, nfib, ir, jcol, kf)];
              }

            pi_lminus1_l(l, row_vec, coords_x, nc,  ncol);

            for (int jcol = 0; jcol < ncol; ++jcol)
            {
                v[mgard_common::get_index3(ncol, nfib, ir, jcol, kf)] = row_vec[jcol] ;
            }
            
          }
      }



    if (nrow > 1)
      {
        for(int kfib = 0; kfib < nf; kfib += Cstride)
          {
            int kf = get_lindex(nf, nfib, kfib); 
            for(int jcol = 0;  jcol < nc; jcol += Cstride) // Do the columns existing  in the coarser level
              {
                int jr = get_lindex(nc, ncol, jcol);  
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    //                int irow_r = get_lindex(nr, nrow, irow);
                    col_vec[irow] = v[mgard_common::get_index3(ncol, nfib, irow, jr, kf)];
                  }
                pi_lminus1_l(l, col_vec, coords_y, nr,  nrow);
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    v[mgard_common::get_index3(ncol, nfib, irow, jr, kf)] = col_vec[irow];
                  }

              }
          }
      }
    
    if (nfib > 1)
      {
        for(int irow = 0;  irow < nr; irow += Cstride) // Do the columns existing  in the coarser level
              {
                int ir = get_lindex(nr, nrow, irow); // get the real location of logical index irow
                for(int jcol = 0;  jcol < nc; jcol += Cstride)
                  {
                    int jr = get_lindex(nc, ncol, jcol);  
                    for (int kfib = 0; kfib < nfib; ++kfib)
                      {
                        fib_vec[kfib] = v[mgard_common::get_index3(ncol, nfib, ir, jr, kfib)];
                      }
                    pi_lminus1_l(l, fib_vec, coords_z, nf,  nfib);
                    for (int kfib = 0; kfib < nfib; ++kfib)
                      {
                         v[mgard_common::get_index3(ncol, nfib, ir, jr, kfib)] = fib_vec[kfib] ;
                      }
                  }
              }
      }
        

    //        Now the new-new stuff, xy-plane
    for(int kfib = 0 ; kfib < nf; kfib += Cstride)
      {
        int kf = get_lindex(nf, nfib, kfib); 
        for(int irow = stride;  irow < nr; irow += Cstride) 
          {
            int ir1 = get_lindex(nr, nrow, irow-stride);
            int ir  = get_lindex(nr, nrow, irow);
            int ir2 = get_lindex(nr, nrow, irow+stride); 

            for(int jcol = stride;  jcol < nc; jcol += Cstride) 
              {

                int jr1 = get_lindex(nc, ncol, jcol-stride);
                int jr = get_lindex(nc, ncol, jcol); 
                int jr2 = get_lindex(nc, ncol, jcol+stride); 

                
                double q11 =  v[mgard_common::get_index3(ncol, nfib, ir1, jr1, kf)];
                double q12 =  v[mgard_common::get_index3(ncol, nfib, ir2, jr1, kf)];
                double q21 =  v[mgard_common::get_index3(ncol, nfib, ir1, jr2, kf)];
                double q22 =  v[mgard_common::get_index3(ncol, nfib, ir2, jr2, kf)];
              
                double x1 =  0.0; //relative coordinate axis centered at irow - Cstride, jcol - Cstride
                double y1 =  0.0;
                double x2 =  mgard_common::get_dist(coords_x,  jr1,  jr2);
                double y2 =  mgard_common::get_dist(coords_y,  ir1,  ir2); 
              
                double x = mgard_common::get_dist(coords_x,  jr1,  jr); 
                double y = mgard_common::get_dist(coords_y,  ir1,  ir); 
                double temp = mgard_common::interp_2d( q11,  q12,  q21,  q22,  x1,  x2,  y1,  y2,  x,  y) ;
                //              std::cout << temp <<"\n";
                v[mgard_common::get_index3(ncol, nfib, ir, jr, kf)]  -= temp; 
              }
          }
      }


    // // //        Now the new-new stuff, xz-plane
    for(int irow = 0;  irow < nr; irow += Cstride) 
      {
        int irr = get_lindex(nr, nrow, irow); 
        for(int jcol = stride;  jcol < nc; jcol += Cstride) 
          {
            int ir1 = get_lindex(nc, ncol, jcol-stride);
            int ir  = get_lindex(nc, ncol, jcol);
            int ir2 = get_lindex(nc, ncol, jcol+stride); 

            for(int kfib = stride ; kfib < nf; kfib += Cstride)
              {
                int jr1 = get_lindex(nf, nfib, kfib-stride);
                int jr =  get_lindex(nf, nfib, kfib); 
                int jr2 = get_lindex(nf, nfib, kfib+stride); 

                
                double q11 =  v[mgard_common::get_index3(ncol, nfib, irr, ir1, jr1)];
                double q12 =  v[mgard_common::get_index3(ncol, nfib, irr, ir2, jr1)];
                double q21 =  v[mgard_common::get_index3(ncol, nfib, irr, ir1, jr2)];
                double q22 =  v[mgard_common::get_index3(ncol, nfib, irr, ir2, jr2)];
              
                double x1 =  0.0; //relative coordinate axis centered at irow - Cstride, jcol - Cstride
                double y1 =  0.0;
                double x2 =  mgard_common::get_dist(coords_z,  jr1,  jr2);
                double y2 =  mgard_common::get_dist(coords_x,  ir1,  ir2); 
              
                double x = mgard_common::get_dist(coords_z,  jr1,  jr); 
                double y = mgard_common::get_dist(coords_x,  ir1,  ir); 
                double temp = mgard_common::interp_2d( q11,  q12,  q21,  q22,  x1,  x2,  y1,  y2,  x,  y) ;
                //              std::cout << temp <<"\n";
                v[mgard_common::get_index3(ncol, nfib, irr, ir, jr)]  -= temp; 
              }
          }
      }

    
    //     //        Now the new-new stuff, yz-plane
    for(int jcol = 0;  jcol < nc; jcol += Cstride)
      {
        int jrr = get_lindex(nc, ncol, jcol);  
          for(int irow = stride;  irow < nr; irow += Cstride) 
          {
            int ir1 = get_lindex(nr, nrow, irow-stride);
            int ir  = get_lindex(nr, nrow, irow);
            int ir2 = get_lindex(nr, nrow, irow+stride); 

            for(int kfib = stride ; kfib < nf; kfib += Cstride)
              {
                int jr1 = get_lindex(nf, nfib, kfib-stride);
                int jr = get_lindex(nf, nfib, kfib); 
                int jr2 = get_lindex(nf, nfib, kfib+stride); 

                
                double q11 =  v[mgard_common::get_index3(ncol, nfib, ir1, jrr, jr1)];
                double q12 =  v[mgard_common::get_index3(ncol, nfib, ir2, jrr, jr1)];
                double q21 =  v[mgard_common::get_index3(ncol, nfib, ir1, jrr, jr2)];
                double q22 =  v[mgard_common::get_index3(ncol, nfib, ir2, jrr, jr2)];
              
                double x1 =  0.0; //relative coordinate axis centered at irow - Cstride, jcol - Cstride
                double y1 =  0.0;
                double x2 =  mgard_common::get_dist(coords_z,  jr1,  jr2);
                double y2 =  mgard_common::get_dist(coords_y,  ir1,  ir2); 
              
                double x = mgard_common::get_dist(coords_z,  jr1,  jr); 
                double y = mgard_common::get_dist(coords_y,  ir1,  ir); 
                double temp = mgard_common::interp_2d( q11,  q12,  q21,  q22,  x1,  x2,  y1,  y2,  x,  y) ;
                //              std::cout << temp <<"\n";
                v[mgard_common::get_index3(ncol, nfib, ir, jrr, jr)]  -= temp; 
              }
          }
      }



    // ///    new-new-new stuff

     for(int irow = stride;  irow < nr; irow += Cstride)
      {
        int ir1 = get_lindex(nr, nrow, irow-stride);
        int ir  = get_lindex(nr, nrow, irow);
        int ir2 = get_lindex(nr, nrow, irow+stride); 
        
        for(int jcol = stride;  jcol < nc; jcol += Cstride)
          {
            int jr1 = get_lindex(nc, ncol, jcol-stride);
            int jr =  get_lindex(nc, ncol, jcol); 
            int jr2 = get_lindex(nc, ncol, jcol+stride); 
            
            for(int kfib = stride;  kfib < nf; kfib += Cstride)
              {

                int kr1 = get_lindex(nf, nfib, kfib-stride);
                int kr = get_lindex(nf, nfib, kfib); 
                int kr2 = get_lindex(nf, nfib, kfib+stride); 

                double x1 = 0.0;
                double y1 = 0.0;
                double z1 = 0.0;

                double x2 =  mgard_common::get_dist(coords_x,  jr1,  jr2);
                double y2 =  mgard_common::get_dist(coords_y,  ir1,  ir2);
                double z2 =  mgard_common::get_dist(coords_z,  kr1,  kr2); 
              
                double x = mgard_common::get_dist(coords_x,  jr1,  jr); 
                double y = mgard_common::get_dist(coords_y,  ir1,  ir);
                double z = mgard_common::get_dist(coords_z,  kr1,  kr); 

                double q000 = v[mgard_common::get_index3(ncol, nfib, ir1, jr1, kr1)];
                double q100 = v[mgard_common::get_index3(ncol, nfib, ir1, jr2, kr1)];
                double q110 = v[mgard_common::get_index3(ncol, nfib, ir1, jr2, kr2)];

                double q010 = v[mgard_common::get_index3(ncol, nfib, ir1, jr1, kr2)];
                
                double q001 = v[mgard_common::get_index3(ncol, nfib, ir2, jr1, kr1)];
                double q101 = v[mgard_common::get_index3(ncol, nfib, ir2, jr2, kr1)];
                double q111 = v[mgard_common::get_index3(ncol, nfib, ir2, jr2, kr2)];

                double q011 = v[mgard_common::get_index3(ncol, nfib, ir2, jr1, kr2)];

                
                                
                double temp =  mgard_common::interp_3d( q000,  q100,  q110,  q010,  q001,  q101,  q111,  q011,  x1,  x2,  y1,  y2,  z1,  z2,  x,  y,  z);

                v[mgard_common::get_index3(ncol, nfib, ir, jr, kr)]  -= temp;
              }
          }
      }
        
  }



  void pi_Ql3D_first(const int nr, const int nc, const int nf, const int nrow, const int ncol, const int nfib, const int  l, double* v,  const std::vector<double>& coords_x, const std::vector<double>& coords_y, const std::vector<double>& coords_z, std::vector<double>& row_vec, std::vector<double>& col_vec, std::vector<double>& fib_vec)
  {
    
    int stride = 1;//current stride
    //  int Pstride = stride/2; //finer stride
    int Cstride = stride*2; // coarser stride

    //  std::vector<double> row_vec(ncol), col_vec(nrow)   ;

    for(int kfib = 0; kfib < nf; kfib += stride)
      {
        int kf = get_lindex(nf, nfib, kfib); // get the real location of logical index irow
        for(int irow = 0;  irow < nr; irow += stride) // Do the rows existing  in the coarser level
          {
            int ir = get_lindex(nr, nrow, irow); // get the real location of logical index irow
            for (int jcol = 0; jcol < ncol; ++jcol)
              {
                row_vec[jcol] = v[mgard_common::get_index3(ncol, nfib, ir, jcol, kf)];
              }

            pi_lminus1_first( row_vec, coords_x, nc,  ncol);

            for (int jcol = 0; jcol < ncol; ++jcol)
            {
                v[mgard_common::get_index3(ncol, nfib, ir, jcol, kf)] = row_vec[jcol] ;
            }
            
          }
      }



    if (nrow > 1)
      {
        for(int kfib = 0; kfib < nf; kfib += stride)
          {
            int kf = get_lindex(nf, nfib, kfib); 
            for(int jcol = 0;  jcol < nc; jcol += stride) // Do the columns existing  in the coarser level
              {
                int jr = get_lindex(nc, ncol, jcol);  
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    //                int irow_r = get_lindex(nr, nrow, irow);
                    col_vec[irow] = v[mgard_common::get_index3(ncol, nfib, irow, jr, kf)];
                  }
                pi_lminus1_first(col_vec, coords_y, nr,  nrow);
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    v[mgard_common::get_index3(ncol, nfib, irow, jr, kf)] = col_vec[irow];
                  }

              }
          }
      }
    
    if (nfib > 1)
      {
        for(int irow = 0;  irow < nr; irow += stride) // Do the columns existing  in the coarser level
              {
                int ir = get_lindex(nr, nrow, irow); // get the real location of logical index irow
                for(int jcol = 0;  jcol < nc; jcol += stride)
                  {
                    int jr = get_lindex(nc, ncol, jcol);  
                    for (int kfib = 0; kfib < nfib; ++kfib)
                      {
                        fib_vec[kfib] = v[mgard_common::get_index3(ncol, nfib, ir, jr, kfib)];
                      }
                    pi_lminus1_first(fib_vec, coords_z, nf,  nfib);
                    for (int kfib = 0; kfib < nfib; ++kfib)
                      {
                         v[mgard_common::get_index3(ncol, nfib, ir, jr, kfib)] = fib_vec[kfib] ;
                      }
                  }
              }
      }
        

    //        Now the new-new stuff, xy-plane
    for(int kfib = 0 ; kfib < nf; kfib += stride)
      {
        int kf = get_lindex(nf, nfib, kfib); 
        for(int irow = 0;  irow < nr-1; irow += stride) 
          {
            int ir  = get_lindex(nr, nrow, irow);
            int irP = get_lindex(nr, nrow, irow+stride); 

            for(int jcol = 0;  jcol < nc-1; jcol += stride) 
              {

                int jr = get_lindex(nc, ncol, jcol); 
                int jrP = get_lindex(nc, ncol, jcol+stride); 

                 if ((irP != ir+1) && (jrP != jr+1)) //we skipped both a row and a column
              {
                double q11 =  v[mgard_common::get_index3(ncol, nfib, ir, jr, kf)];
                double q12 =  v[mgard_common::get_index3(ncol, nfib, irP, jr, kf)];
                double q21 =  v[mgard_common::get_index3(ncol, nfib, ir, jrP, kf)];
                double q22 =  v[mgard_common::get_index3(ncol, nfib, irP, jrP, kf)];
              
                double x1 =  0.0; //relative coordinate axis centered at irow - Cstride, jcol - Cstride
                double y1 =  0.0;
                double x2 =  mgard_common::get_dist(coords_x,  jr,  jrP);
                double y2 =  mgard_common::get_dist(coords_y,  ir,  irP); 
              
                double x = mgard_common::get_dist(coords_x,  jr,  jr+1); 
                double y = mgard_common::get_dist(coords_y,  ir,  ir+1); 
                double temp = mgard_common::interp_2d( q11,  q12,  q21,  q22,  x1,  x2,  y1,  y2,  x,  y) ;
                //              std::cout << temp <<"\n";
                v[mgard_common::get_index3(ncol, nfib, ir+1, jr+1, kf)]  -= temp; 
              }
              }
          }
      }


    // // //        Now the new-new stuff, xz-plane
    for(int irow = 0;  irow < nr; irow += stride) 
      {
        int irr = get_lindex(nr, nrow, irow); 
        for(int jcol = 0;  jcol < nc-1; jcol += stride) 
          {
            int ir  = get_lindex(nc, ncol, jcol);
            int irP = get_lindex(nc, ncol, jcol+stride); 

            for(int kfib = 0 ; kfib < nf-1; kfib += stride)
              {

                int jr =  get_lindex(nf, nfib, kfib); 
                int jrP = get_lindex(nf, nfib, kfib+stride); 

                if ((irP != ir+1) && (jrP != jr+1)) //we skipped both a row and a column
              {
                
                double q11 =  v[mgard_common::get_index3(ncol, nfib, irr, ir, jr)];
                double q12 =  v[mgard_common::get_index3(ncol, nfib, irr, irP, jr)];
                double q21 =  v[mgard_common::get_index3(ncol, nfib, irr, ir, jrP)];
                double q22 =  v[mgard_common::get_index3(ncol, nfib, irr, irP, jrP)];
              
                double x1 =  0.0; //relative coordinate axis centered at irow - Cstride, jcol - Cstride
                double y1 =  0.0;
                double x2 =  mgard_common::get_dist(coords_z,  jr,  jrP);
                double y2 =  mgard_common::get_dist(coords_x,  ir,  irP); 
              
                double x = mgard_common::get_dist(coords_z,  jr,  jr+1); 
                double y = mgard_common::get_dist(coords_x,  ir,  ir+1); 
                double temp = mgard_common::interp_2d( q11,  q12,  q21,  q22,  x1,  x2,  y1,  y2,  x,  y) ;
                //              std::cout << temp <<"\n";
                v[mgard_common::get_index3(ncol, nfib, irr, ir+1, jr+1)]  -= temp;
              }
              }
          }
      }

    
    //     //        Now the new-new stuff, yz-plane
    for(int jcol = 0;  jcol < nc; jcol += stride)
      {
        int jrr = get_lindex(nc, ncol, jcol);  
          for(int irow = 0;  irow < nr-1; irow += stride) 
          {
            int ir  = get_lindex(nr, nrow, irow);
            int irP = get_lindex(nr, nrow, irow+stride); 
            for(int kfib = 0 ; kfib < nf-1; kfib += stride)
              {
                int jr = get_lindex(nf, nfib, kfib); 
                int jrP = get_lindex(nf, nfib, kfib+stride); 

                if ((irP != ir+1) && (jrP != jr+1)) //we skipped both a row and a column
                  {
                double q11 =  v[mgard_common::get_index3(ncol, nfib, ir, jrr, jr)];
                double q12 =  v[mgard_common::get_index3(ncol, nfib, irP, jrr, jr)];
                double q21 =  v[mgard_common::get_index3(ncol, nfib, ir, jrr, jrP)];
                double q22 =  v[mgard_common::get_index3(ncol, nfib, irP, jrr, jrP)];
              
                double x1 =  0.0; //relative coordinate axis centered at irow - Cstride, jcol - Cstride
                double y1 =  0.0;
                double x2 =  mgard_common::get_dist(coords_z,  jr,  jrP);
                double y2 =  mgard_common::get_dist(coords_y,  ir,  irP); 
              
                double x = mgard_common::get_dist(coords_z,  jr,  jr+1); 
                double y = mgard_common::get_dist(coords_y,  ir,  ir+1); 
                double temp = mgard_common::interp_2d( q11,  q12,  q21,  q22,  x1,  x2,  y1,  y2,  x,  y) ;
                //              std::cout << temp <<"\n";
                v[mgard_common::get_index3(ncol, nfib, ir+1, jrr, jr+1)]  -= temp; 
                  }
              }
          }
         }



    ///    new-new-new stuff

     for(int irow = 0;  irow < nr-1; irow += stride)
      {
        int ir  = get_lindex(nr, nrow, irow);
        int irP = get_lindex(nr, nrow, irow+stride); 
        
        for(int jcol = 0;  jcol < nc-1; jcol += stride)
          {
            int jr =  get_lindex(nc, ncol, jcol); 
            int jrP = get_lindex(nc, ncol, jcol+stride); 
            
            for(int kfib = 0;  kfib < nf-1; kfib += stride)
              {
                int kr = get_lindex(nf, nfib, kfib); 
                int krP = get_lindex(nf, nfib, kfib+stride); 

                if ((irP != ir+1) && (jrP != jr+1) && (krP != kr+1)) //we skipped both a row and a column
              {
                double x1 = 0.0;
                double y1 = 0.0;
                double z1 = 0.0;

                double x2 =  mgard_common::get_dist(coords_x,  jr,  jrP);
                double y2 =  mgard_common::get_dist(coords_y,  ir,  irP);
                double z2 =  mgard_common::get_dist(coords_z,  kr,  krP); 
              
                double x = mgard_common::get_dist(coords_x,  jr,  jr+1); 
                double y = mgard_common::get_dist(coords_y,  ir,  ir+1);
                double z = mgard_common::get_dist(coords_z,  kr,  kr+1); 

                double q000 = v[mgard_common::get_index3(ncol, nfib, ir, jr, kr)];
                double q100 = v[mgard_common::get_index3(ncol, nfib, ir, jrP, kr)];
                double q110 = v[mgard_common::get_index3(ncol, nfib, ir, jrP, krP)];

                double q010 = v[mgard_common::get_index3(ncol, nfib, ir, jr, krP)];
                
                double q001 = v[mgard_common::get_index3(ncol, nfib, irP, jr, kr)];
                double q101 = v[mgard_common::get_index3(ncol, nfib, irP, jrP, kr)];
                double q111 = v[mgard_common::get_index3(ncol, nfib, irP, jrP, krP)];

                double q011 = v[mgard_common::get_index3(ncol, nfib, irP, jr, krP)];

                
                                
                double temp =  mgard_common::interp_3d( q000,  q100,  q110,  q010,  q001,  q101,  q111,  q011,  x1,  x2,  y1,  y2,  z1,  z2,  x,  y,  z);

                v[mgard_common::get_index3(ncol, nfib, ir+1, jr+1, kr+1)]  -= temp;
              }
              }
          }
      }
        
  }

void  assign_num_level(const int  l, std::vector<double>& v , double num, int n, int no)
{
  // int stride = std::pow(2,l);//current stride
  // for (int i = 0; i< n-1; i += stride)
  //   {
  //    *get_ref(v, n, no, i) = num;
  //   }
  // v.back() = num;
  int stride = std::pow(2,l);
  for (int i = 0; i< n; i += stride)
    {
      int il = get_lindex(n, no, i);
      v[il] = num;
    }
}


  

  void assign_num_level_l(const int  l, double* v, double num, int nr, int nc, const int nrow, const int ncol)
  {
    // set the value of nodal values at level l to number num


    int stride = std::pow(2,l);//current stride

    for(int irow = 0;  irow < nr; irow += stride)
      {
        int ir = get_lindex(nr, nrow, irow);
        for(int jcol = 0;  jcol < nc; jcol += stride)
          {
            int jr = get_lindex(nc, ncol, jcol); 
            v[mgard_common::get_index(ncol,ir, jr)]  = num ;
          }
      }
  }

void restrict_first(std::vector<double>& v,  std::vector<double>& coords, int n, int no)
  {
    // calculate the result of restriction

    for(int i =  0; i < n - 1; ++i) //loop over the logical array
      {
        int i_logic  = get_lindex(n,  no,  i);
        int i_logicP = get_lindex(n,  no, i+1);
        
        if (i_logicP != i_logic+1) // next real memory location was jumped over, so need to restrict
          {
             double h1 = mgard_common::get_h(coords, i_logic,  1);
             double h2 = mgard_common::get_h(coords, i_logic + 1, 1);
             double hsum = h1 + h2;           
             // v[i_logic]  = 0.5*v[i_logic]  + 0.5*h2*v[i_logic+1]/hsum;
             // v[i_logicP] = 0.5*v[i_logicP] + 0.5*h1*v[i_logic+1]/hsum;
             v[i_logic]  += h2*v[i_logic+1]/hsum;
             v[i_logicP] += h1*v[i_logic+1]/hsum;
          }

      }
  }

  void solve_tridiag_M_l(const int  l, std::vector<double>& v,  std::vector<double>& coords, int n, int no)
  {

    //  int my_level = nlevel - l;
    int stride = std::pow(2,l);//current stride

    double am, bm, h1, h2;
    am = 2.0*get_h_l(coords, n, no, 0, stride) ; // first element of upper diagonal U.

    //    bm = get_h(coords, 0, stride) / am;
    bm = get_h_l(coords, n, no, 0, stride)/am;
    int nlevel = static_cast<int> (std::log2(n - 1));
    //    std::cout << nlevel;
    int nc = std::pow(2,nlevel - l) + 1 ;
    std::vector<double> coeff(nc);
    int counter = 1;
    coeff.front() = am;

    //std::cout <<  am<< "\t"<< bm<<"\n";
    // forward sweep
    for(int i = stride; i < n - 1 ; i += stride)
      {
        h1 = get_h_l(coords, n, no,  i-stride, stride);
        h2 = get_h_l(coords, n, no,  i, stride);

        *get_ref(v, n, no, i) -= *get_ref(v, n, no, i-stride)*bm;

        am = 2.0*(h1+h2) - bm*h1;
        bm = h2 / am;

        coeff.at(counter) = am;
        ++counter;
      }


    h2 = get_h_l(coords, n, no, n-1-stride, stride);  
    am = 2.0*h2 - bm*h2;

    //    *get_ref(v, n, no, n-1) -= *get_ref(v, n, no, n-1-stride)*bm;
    v.back()  -= *get_ref(v, n, no, n-1-stride)*bm;
    coeff.at(counter) = am;

    //backward sweep

    //    *get_ref(v, n, no, n-1) /= am;
    v.back() /= am;    
    --counter;

    for(int i = n-1-stride; i >= 0 ; i -= stride)
      {
        h2 = get_h_l(coords, n, no, i, stride);
        *get_ref(v, n, no, i) = (*get_ref(v, n, no, i) - h2*(*get_ref(v, n, no, i+stride)) )/coeff.at(counter)  ;

        //        *get_ref(v, n, no, i) = 3  ;
        
        --counter;
      }
  }


  void add_level_l(const int  l, double* v, double* work, int nr, int nc, int nrow, int ncol)
  {
    // v += work at level l


    int stride = std::pow(2,l);//current stride


    for(int irow = 0;  irow < nr; irow += stride)
      {
        int ir  = get_lindex(nr,  nrow,  irow);
        for(int jcol = 0;  jcol < nc; jcol += stride)
          {
            int jr  = get_lindex(nc,  ncol,  jcol);
            v[mgard_common::get_index(ncol,ir, jr)]  += work[mgard_common::get_index(ncol,ir, jr)];
          }
      }

  }


  void add3_level_l(const int  l, double* v, double* work, int nr, int nc, int nf, int nrow, int ncol, int nfib)
  {
    // v += work at level l


    int stride = std::pow(2,l);//current stride


    for(int irow = 0;  irow < nr; irow += stride)
      {
        int ir  = get_lindex(nr,  nrow,  irow);
        for(int jcol = 0;  jcol < nc; jcol += stride)
          {
            int jr  = get_lindex(nc,  ncol,  jcol);
            for(int kfib = 0;  kfib < nf; kfib += stride)
              {
                int kr  = get_lindex(nf,  nfib,  kfib);
                v[mgard_common::get_index3(ncol, nfib, ir, jr, kr)]  += work[mgard_common::get_index3(ncol, nfib, ir, jr, kr)];                
              }

          }
      }

  }


void sub3_level_l(const int  l, double* v, double* work, int nr, int nc, int nf, int nrow, int ncol, int nfib)
  {
    // v += work at level l


    int stride = std::pow(2,l);//current stride


    for(int irow = 0;  irow < nr; irow += stride)
      {
        int ir  = get_lindex(nr,  nrow,  irow);
        for(int jcol = 0;  jcol < nc; jcol += stride)
          {
            int jr  = get_lindex(nc,  ncol,  jcol);
            for(int kfib = 0;  kfib < nf; kfib += stride)
              {
                int kr  = get_lindex(nf,  nfib,  kfib);
                v[mgard_common::get_index3(ncol, nfib, ir, jr, kr)]  -= work[mgard_common::get_index3(ncol, nfib, ir, jr, kr)];                
              }

          }
      }
}


void sub3_level(const int  l, double* v, double* work, int nrow, int ncol, int nfib)
  {
    // v += work at level l


    int stride = std::pow(2,l);//current stride


    for(int irow = 0;  irow < nrow; irow += stride)
      {
        for(int jcol = 0;  jcol < ncol; jcol += stride)
          {
            for(int kfib = 0;  kfib < nfib; kfib += stride)
              {
                v[mgard_common::get_index3(ncol, nfib, irow, jcol, kfib)]  -= work[mgard_common::get_index3(ncol, nfib, irow, jcol, kfib)];                
              }

          }
      }}

  
    void sub_level_l(const int  l, double* v, double* work, int nr, int nc, int nf, int nrow, int ncol, int nfib)
  {
    // v += work at level l


    int stride = std::pow(2,l);//current stride


    for(int irow = 0;  irow < nr; irow += stride)
      {
        int ir  = get_lindex(nr,  nrow,  irow);
        for(int jcol = 0;  jcol < nc; jcol += stride)
          {
            int jr  = get_lindex(nc,  ncol,  jcol);
            for(int kfib = 0;  kfib < nf; kfib += stride)
              {
                int kr  = get_lindex(nf,  nfib,  kfib);
                v[mgard_common::get_index3(ncol, nfib, ir, jr, kr)]  -= work[mgard_common::get_index3(ncol, nfib, ir, jr, kr)];                
              }

          }
      }

  }

    void project_first(const int nr, const int nc, const int nrow, const int ncol,  const int l_target, double* v, std::vector<double>& work, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& row_vec, std::vector<double>& col_vec )
    {
      
    }
  
  void prep_2D(const int nr, const int nc, const int nrow, const int ncol,  const int l_target, double* v, std::vector<double>& work, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& row_vec, std::vector<double>& col_vec )
  {
    int l = 0;
    int stride = 1;
    pi_Ql_first(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec, col_vec); //(I-\Pi u) this is the initial move to 2^k+1 nodes

    mgard_cannon::copy_level(nrow, ncol, l, v,  work);
    mgard_gen::assign_num_level_l(0, work.data(), 0.0, nr, nc, nrow, ncol);

    // row-sweep
    for(int irow = 0;  irow < nrow; ++irow)
    //    for(int irow = 0;  irow < nrow; ++irow)
      {
        //  int ir = get_lindex(nr, nrow, irow); // get the real location of logical index irow
        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            row_vec[jcol] = work[mgard_common::get_index(ncol, irow, jcol)];
          }
        
        mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);
        
        restrict_first(row_vec, coords_x, nc, ncol);
        

        
        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            work[mgard_common::get_index(ncol, irow, jcol)] = row_vec[jcol] ;
          }
        
      }

for(int irow = 0;  irow < nr; ++irow)
  {
    int ir = get_lindex(nr, nrow, irow);
    for(int jcol = 0; jcol < ncol; ++jcol)
      {
        row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
      }
    solve_tridiag_M_l(0,  row_vec, coords_x, nc, ncol);

    for(int jcol = 0; jcol < ncol; ++jcol)
      {
        work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol] ;
      }
  }
 
    //      std::cout << "Row sweep done!"<<"\n";

    // column-sweep
    if (nrow > 1) //do this if we have an 2-dimensional array
      {
        for(int jcol = 0;  jcol < ncol; jcol += stride)
          {
            // int jr  = get_lindex(nc, ncol, jcol);
            for(int irow = 0;  irow < nrow; ++irow)
              {
                col_vec[irow] = work[mgard_common::get_index(ncol,irow, jcol)] ;
              }
            
            mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);
            
            restrict_first(col_vec, coords_y, nr, nrow);

            
            
            for(int irow = 0;  irow < nrow; ++irow)
              {
                work[mgard_common::get_index(ncol, irow, jcol)]  = col_vec[irow] ;
              }
            
          }
        
        for(int jcol = 0;  jcol < nc; ++jcol)
          {
            int jr  = get_lindex(nc,  ncol,  jcol);
            for(int irow = 0;  irow < nrow; ++irow)
              {
                col_vec[irow] = work[mgard_common::get_index(ncol, irow, jr )];
              }
            solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow);

            for(int irow = 0;  irow < nrow; ++irow)
              {
                //work[mgard_common::get_index(ncol, irow, jr )] = col_vec[irow];
              }
          }
      }
        // Solved for (z_l, phi_l) = (c_{l+1}, vl)
    add_level_l(0, v, work.data(),  nr,  nc,  nrow,  ncol);
  }



  void mass_mult_l(const int  l, std::vector<double>& v,  std::vector<double>& coords, const int n, const int no)
  {

    int stride = std::pow(2,l);
    double temp1, temp2;
    double h1, h2;

    // Mass matrix times nodal value-vec
    temp1 = v.front(); //save u(0) for later use

    h1 = get_h_l(coords, n, no, 0, stride) ;

    v.front() = 2.0*h1*temp1 +  h1* (*get_ref(v, n,  no,  stride));

    for(int i = stride;  i <= n - 1 - stride; i += stride )
      {
        temp2 = *get_ref(v, n,  no,  i);
        h1 = get_h_l(coords, n, no, i-stride, stride);
        h2 = get_h_l(coords, n, no, i, stride);
        
        *get_ref(v, n,  no,  i)  = h1*temp1 + 2*(h1 + h2)*temp2 + h2 * (*get_ref(v, n,  no,  i+stride));
        temp1 = temp2; // save u(n) for later use
      }
    v.back() = get_h_l(coords, n, no, n-stride-1, stride) * temp1 + 2*get_h_l(coords, n, no, n-stride-1, stride)*v.back() ;
  }

void restrict_l(const int  l, std::vector<double>& v,  std::vector<double>& coords, int n, int no)
{
    int stride = std::pow(2,l);
    int Pstride = stride/2;//finer stride
    
    // calculate the result of restriction

    double h1 = get_h_l(coords,n, no, 0, Pstride);
    double h2 = get_h_l(coords,n, no, Pstride , Pstride);
    double hsum = h1+h2;                            
    
    v.front() += h2*(*get_ref(v, n,  no, Pstride))/hsum; //first element
    
    for(int i =  stride; i <= n - stride; i += stride)
      {
        *get_ref(v, n,  no,  i) += h1*(*get_ref(v, n,  no, i-Pstride))/hsum;
        h1 = get_h_l(coords, n, no, i, Pstride);
        h2 = get_h_l(coords, n, no, i+Pstride, Pstride);
        hsum = h1+h2;
        *get_ref(v, n,  no,  i) += h2*(*get_ref(v, n,  no, i+Pstride))/hsum;
      }
    v.back() +=  h1* (*get_ref(v, n,  no,  n-Pstride-1))/hsum; //last element
}




  double ml2_norm3(const int  l, int nr, int nc, int nf, int nrow, int ncol, int nfib,  std::vector<double>& v,  std::vector<double>& coords_x,  std::vector<double>& coords_y,  std::vector<double>& coords_z)
  {
        
    int stride = std::pow(2,l);
    std::vector<double> work(v);
    std::vector<double> row_vec(ncol), col_vec(nrow), fib_vec(nfib);
    
     for (int kfib = 0; kfib < nf; kfib += stride)
         {
           int kf = get_lindex(nf, nfib, kfib);
           for(int irow = 0;  irow < nr; irow += stride)
             {
               int ir = get_lindex(nr, nrow, irow);
               for(int jcol = 0;  jcol < nc; ++jcol)
                 {
                   row_vec[jcol] = work[mgard_common::get_index3(ncol,nfib,ir,jcol,kf)];
                 }
               mgard_gen::mass_mult_l(l, row_vec, coords_x, nc, ncol );
             }
         }

     for (int kfib = 0; kfib < nf; kfib += stride)
       {
         int kf = get_lindex(nf, nfib, kfib);
         for(int jcol = 0;  jcol < nc; jcol += stride)
           {
             int jr  = get_lindex(nc,  ncol,  jcol);
             for(int irow = 0;  irow < nrow; ++irow)
               {
                 col_vec[irow] = col_vec[irow] = work[mgard_common::get_index3(ncol,nfib,irow,jr,kf)] ;
               }
             mgard_gen::mass_mult_l(l,  col_vec, coords_y, nr, nrow);
           }
       }

     for(int irow = 0;  irow < nr; irow += stride) 
       {
         int ir = get_lindex(nr, nrow, irow); 
         for(int jcol = 0;  jcol < nc; jcol += stride)
           {
             int jr = get_lindex(nc, ncol, jcol);  
             for (int kfib = 0; kfib < nfib; ++kfib)
               {
                 fib_vec[kfib] = work[mgard_common::get_index3(ncol, nfib, ir, jr, kfib)];
               }
             mgard_gen::mass_mult_l(l,  fib_vec, coords_z, nf, nfib);
           }
       }


     double norm = 0.0;

       for(int irow = 0;  irow < nr; irow += stride) 
         {
           int ir = get_lindex(nr, nrow, irow); 
           for(int jcol = 0;  jcol < nc; jcol += stride)
             {
               int jr = get_lindex(nc, ncol, jcol);  
               for (int kfib = 0; kfib < nf; kfib += stride)
                 {
                   int kf = get_lindex(nf, nfib, kfib);

                   norm += work[mgard_common::get_index3(ncol, nfib, ir, jr, kf)] * v[mgard_common::get_index3(ncol, nfib, ir, jr, kf)];
                 }
             }
         }

       return norm/216.0; // account for missing 1/6 factors in M_{x,y,z}
     
  }

void prolongate_l(const int  l, std::vector<double>& v,  std::vector<double>& coords, int n, int no)
{
  
  int stride = std::pow(2,l);
  int Pstride = stride/2;

  for(  int i =  stride; i < n ; i += stride)
    {
      double h1 = get_h_l(coords, n, no, i-stride,  Pstride);
      double h2 = get_h_l(coords, n, no, i-Pstride, Pstride);
      double hsum = h1+h2;
      
      *get_ref(v, n,  no,  i-Pstride) = ( h2*(*get_ref(v, n,  no,  i-stride)) + h1*(*get_ref(v, n,  no,  i)) )/hsum;
    }

  // double h1 = get_h_l(coords, n, no, n-1-stride,  Pstride);
  // double h2 = get_h_l(coords, n, no, n-1-Pstride, Pstride);
  // double hsum = h1+h2;
  
  // *get_ref(v, n,  no,  n-1-Pstride) = ( h2*(*get_ref(v, n,  no,  n-1-stride)) + h1*(v.back()) )/hsum;
  
}
  
  void refactor_2D(const int nr, const int nc, const int nrow, const int ncol,  const int l_target, double* v, std::vector<double>& work, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& row_vec, std::vector<double>& col_vec )
  {
    //refactor

   // for (int l = l_target; l < l_target + 1; ++l)
   //   {
    int l = l_target;
       int stride = std::pow(2,l);//current stride
       int Cstride = stride*2; // coarser stride

       // -> change funcs in pi_QL to use _l functions, otherwise distances are wrong!!!
       //       pi_Ql(nr, nc, nrow, ncol, l, v, coords_x, coords_y, row_vec, col_vec); //rename!. v@l has I-\Pi_l Q_l+1 u

       //copy_level_l(l,  v,  work.data(),  nr,  nc,  nrow,  ncol);
       //assign_num_level_l(l+1, work.data(),  0.0,  nr,  nc,  nrow,  ncol);

        // row-sweep
        for(int irow = 0;  irow < nr; ++irow)
          {
            int ir = get_lindex(nr, nrow, irow);
            for(int jcol = 0; jcol < ncol; ++jcol)
              {
                row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
              }

            mgard_gen::mass_mult_l(l, row_vec, coords_x, nc, ncol );

            mgard_gen::restrict_l(l+1, row_vec, coords_x, nc, ncol );

            mgard_gen::solve_tridiag_M_l(l+1, row_vec, coords_x, nc, ncol );

            for(int jcol = 0; jcol < ncol; ++jcol)
              {
                work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol] ;
              }

          }


        // column-sweep
        if (nrow > 1) //do this if we have an 2-dimensional array
          {
            for(int jcol = 0;  jcol < nc; jcol += Cstride)
              {
                int jr  = get_lindex(nc,  ncol,  jcol);
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    col_vec[irow] = work[mgard_common::get_index(ncol, irow, jr)] ;
                  }


                mgard_gen::mass_mult_l(l,  col_vec, coords_y, nr, nrow);
                mgard_gen::restrict_l(l+1, col_vec, coords_y, nr, nrow);
                mgard_gen::solve_tridiag_M_l(l+1,  col_vec, coords_y, nr, nrow);


                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    work[mgard_common::get_index(ncol, irow, jr)]  = col_vec[irow] ;
                  }

              }
          }


        // Solved for (z_l, phi_l) = (c_{l+1}, vl)
        //        add_level_l(l+1, v, work.data(),  nr,  nc,  nrow,  ncol);
        //}


  }


    void refactor_2D_first(const int nr, const int nc, const int nrow, const int ncol,  const int l_target, double* v, std::vector<double>& work, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& row_vec, std::vector<double>& col_vec )
  {
    //refactor

    for(int irow = 0;  irow < nrow; ++irow)
      {
        //        int ir = get_lindex(nr, nrow, irow);
        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            row_vec[jcol] = work[mgard_common::get_index(ncol, irow, jcol)];
          }
        
        mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);
        
        restrict_first(row_vec, coords_x, nc, ncol);
        
        
        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            work[mgard_common::get_index(ncol, irow, jcol)] = row_vec[jcol] ;
          }

      }

    for(int irow = 0;  irow < nr; ++irow)
      {
        int ir = get_lindex(nr, nrow, irow);
        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
          }
        
        mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol );

        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol] ;
          }
      }

    //   //   std::cout << "recomposing-colsweep" << "\n";

    //     // column-sweep, this is the slow one! Need something like column_copy
        if( nrow > 1) // check if we have 1-D array..
          {
            for(int jcol = 0;  jcol < ncol; ++jcol)
              {
                //      int jr  = get_lindex(nc,  ncol,  jcol);
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    col_vec[irow] = work[mgard_common::get_index(ncol, irow, jcol )];
                  }

                
                mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);
                
                mgard_gen::restrict_first(col_vec, coords_y, nr, nrow);

            

                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    work[mgard_common::get_index(ncol, irow, jcol)] = col_vec[irow] ;
                  }
              }
            
            for(int jcol = 0;  jcol < nc; ++jcol)
              {
                int jr  = get_lindex(nc,  ncol,  jcol);
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    col_vec[irow] = work[mgard_common::get_index(ncol, irow, jr )];
                  }

                mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow );
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    work[mgard_common::get_index(ncol, irow, jr )] = col_vec[irow] ;
                  }
              }
          }


  }


  void copy3_level_l(const int  l, double* v, double* work, int nr, int nc, int nf, int nrow, int ncol, int nfib)
  {
    // work_l = v_l
    int stride = std::pow(2,l);//current stride

    for(int irow = 0;  irow < nr; irow += stride)
      {
        int ir  = get_lindex(nr,  nrow,  irow);
        for(int jcol = 0;  jcol < nc; jcol += stride)
          {
            int jr  = get_lindex(nc,  ncol,  jcol);
            for(int kfib = 0; kfib < nf; kfib += stride)
              {
                int kr  = get_lindex(nf,  nfib,  kfib);
                work[mgard_common::get_index3(ncol, nfib, ir, jr, kr)]  =  v[mgard_common::get_index3(ncol, nfib, ir, jr, kr)] ;
              }
          }
      }
  }

  void copy3_level(const int  l, double* v, double* work, int nrow, int ncol, int nfib)
  {
    // work_l = v_l
    int stride = std::pow(2,l);//current stride

    for(int irow = 0;  irow < nrow; irow += stride)
      {
        for(int jcol = 0;  jcol < ncol; jcol += stride)
          {
            for(int kfib = 0; kfib < nfib; kfib += stride)
              {
                work[mgard_common::get_index3(ncol, nfib,irow, jcol, kfib)]  =  v[mgard_common::get_index3(ncol, nfib,irow, jcol, kfib)] ;
              }
          }
      }
  }
void assign3_level_l(const int  l, double* v, double num, int nr, int nc, int nf, int nrow, int ncol, int nfib)
  {
    // work_l = v_l
    int stride = std::pow(2,l);//current stride

    for(int irow = 0;  irow < nr; irow += stride)
      {
        int ir  = get_lindex(nr,  nrow,  irow);
        for(int jcol = 0;  jcol < nc; jcol += stride)
          {
            int jr  = get_lindex(nc,  ncol,  jcol);
            for(int kfib = 0; kfib < nf; kfib += stride)
              {
                int kr  = get_lindex(nf,  nfib,  kfib);
                v[mgard_common::get_index3(ncol, nfib, ir, jr, kr)]  = num ;
              }
          }
      }
  }


void refactor_3D(const int nr, const int nc, const int nf, const int nrow, const int ncol, const int nfib,  const int l_target, double* v, std::vector<double>& work, std::vector<double>& work2d, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z)
{

  std::vector<double> v2d(nrow*ncol), fib_vec(nfib);
  std::vector<double> row_vec(ncol);
  std::vector<double> col_vec(nrow);

  
   for (int l = 0; l < l_target; ++l)
     {
       int stride = std::pow(2,l);
       int Cstride = 2*stride;
       
       pi_Ql3D( nr,  nc,  nf,  nrow,  ncol,  nfib,   l,  v,  coords_x,  coords_y, coords_z,  row_vec,  col_vec,  fib_vec);
       
       mgard_gen::copy3_level_l(l,  v,  work.data(),  nr,  nc, nf,  nrow,  ncol, nfib);
       mgard_gen::assign3_level_l(l+1, work.data(),  0.0,  nr,  nc, nf,  nrow,  ncol, nfib);

       //       for (int kfib = 0; kfib < nfib; ++kfib)
       for (int kfib = 0; kfib < nf; kfib += stride)
         {
           //           int kf = kfib;
           int kf = get_lindex(nf, nfib, kfib); // get the real location of logical index irow
           mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
           mgard_gen::refactor_2D(nr, nc,  nrow, ncol,  l, v2d.data(), work2d, coords_x, coords_y,  row_vec, col_vec);
           mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
         }
       
       for(int irow = 0; irow < nr; irow += Cstride)
         {
           int ir = get_lindex(nr, nrow, irow);
           for(int jcol = 0; jcol < nc; jcol += Cstride)
             {
               int jc  = get_lindex(nc,  ncol,  jcol);
               for(int kfib = 0; kfib < nfib; ++kfib)
                 {
                   fib_vec[kfib] = work[mgard_common::get_index3(ncol,nfib,ir,jc,kfib)];
                 }
               mgard_gen::mass_mult_l(l,  fib_vec, coords_z, nf, nfib);
               mgard_gen::restrict_l(l+1, fib_vec, coords_z, nf, nfib);
               mgard_gen::solve_tridiag_M_l(l+1,  fib_vec, coords_z, nf, nfib);
               for(int kfib = 0; kfib < nfib; ++kfib)
                 {
                   work[mgard_common::get_index3(ncol,nfib,ir,jc,kfib)] = fib_vec[kfib] ;
                 }
          
             }
         }
  
       add3_level_l(l+1, v, work.data(),  nr,  nc, nf,  nrow,  ncol, nfib);
        
     }
}




void compute_zl(const int nr, const int nc, const int nrow, const int ncol,  const int l_target,  std::vector<double>& work, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& row_vec, std::vector<double>& col_vec )
  {
        // recompose
    std::cout << "recomposing" << "\n";
    //    for (int l = l_target ; l > 0 ; --l)
    //  {
    int l = l_target;
    
        int stride = std::pow(2,l);//current stride
        int Pstride = stride/2;

        //        copy_level_l(l-1,  v,  work.data(),  nr,  nc,  nrow,  ncol);

        //assign_num_level_l(l, work.data(),  0.0,  nr,  nc,  nrow,  ncol);

        std::cout << "recomposing-rowsweep" << "\n";
        //  l = 0;
        // row-sweep
        for(int irow = 0;  irow < nr; irow += 1)
          {
            int ir = get_lindex(nr, nrow, irow);
            for(int jcol = 0; jcol < ncol; ++jcol)
              {
                row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
              }


            mgard_gen::mass_mult_l(l-1, row_vec, coords_x, nc, ncol );

            mgard_gen::restrict_l(l, row_vec, coords_x, nc, ncol );

            mgard_gen::solve_tridiag_M_l(l, row_vec, coords_x, nc, ncol );


            for(int jcol = 0; jcol < ncol; ++jcol)
              {
                work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol] ;
              }

          }

      //   std::cout << "recomposing-colsweep" << "\n";

        // column-sweep, this is the slow one! Need something like column_copy
        if( nrow > 1) // check if we have 1-D array..
          {
            for(int jcol = 0;  jcol < nc; jcol += stride)
              {
                int jr  = get_lindex(nc,  ncol,  jcol);
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    col_vec[irow] = work[mgard_common::get_index(ncol, irow, jr )];
                  }

                
                mgard_gen::mass_mult_l(l-1, col_vec, coords_y, nr, nrow );
            
                mgard_gen::restrict_l(l, col_vec, coords_y, nr, nrow );

                mgard_gen::solve_tridiag_M_l(l, col_vec, coords_y, nr, nrow );
            

                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    work[mgard_common::get_index(ncol, irow, jr )] = col_vec[irow] ;
                  }
              }
          }

  }



void compute_zl_last(const int nr, const int nc, const int nrow, const int ncol,  const int l_target,  std::vector<double>& work, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& row_vec, std::vector<double>& col_vec )
  {

    for(int irow = 0;  irow < nr; ++irow )
      {
        int ir = get_lindex(nr, nrow, irow);
        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
          }
        
        mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);
        
        restrict_first(row_vec, coords_x, nc, ncol);
        
        
        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol] ;
          }

      }

    for(int irow = 0;  irow < nr; ++irow)
      {
        int ir = get_lindex(nr, nrow, irow);
        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
          }
        
        mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol );

        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol] ;
          }
      }

    //   //   std::cout << "recomposing-colsweep" << "\n";

    //     // column-sweep, this is the slow one! Need something like column_copy
        if( nrow > 1) // check if we have 1-D array..
          {
            for(int jcol = 0;  jcol < ncol; ++jcol)
              {
                //      int jr  = get_lindex(nc,  ncol,  jcol);
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    col_vec[irow] = work[mgard_common::get_index(ncol, irow, jcol )];
                  }

                
                mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);
                
                mgard_gen::restrict_first(col_vec, coords_y, nr, nrow);

            

                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    work[mgard_common::get_index(ncol, irow, jcol)] = col_vec[irow] ;
                  }
              }
            
            for(int jcol = 0;  jcol < nc; ++jcol)
              {
                int jr  = get_lindex(nc,  ncol,  jcol);
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    col_vec[irow] = work[mgard_common::get_index(ncol, irow, jr )];
                  }

                mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow );
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    work[mgard_common::get_index(ncol, irow, jr )] = col_vec[irow];
                  }
              }
          }

  }


void prolongate_last(std::vector<double>& v,  std::vector<double>& coords, int n, int no)
  {
    // calculate the result of restriction

    for(int i =  0; i < n - 1; ++i) //loop over the logical array
      {
        int i_logic  = get_lindex(n,  no,  i);
        int i_logicP = get_lindex(n,  no, i+1);
        
        if (i_logicP != i_logic+1) // next real memory location was jumped over, so need to restrict
          {
            double h1 = mgard_common::get_h(coords, i_logic,  1);
            double h2 = mgard_common::get_h(coords, i_logic + 1, 1);
             double hsum = h1 + h2;
             v[i_logic+1] = (h2*v[i_logic] + h1*v[i_logicP])/hsum;
             //             v[i_logic+1] = 2*(h1*v[i_logicP])/hsum;

          }

      }
  }


void prolong_add_2D(const int nr, const int nc, const int nrow, const int ncol,  const int l_target,  std::vector<double>& work, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& row_vec, std::vector<double>& col_vec )
 {
   int l = l_target;
    
   int stride = std::pow(2,l);//current stride
   int Pstride = stride/2;
   
   
   for(int irow = 0;  irow < nr; irow += stride )
     {
       int ir = get_lindex(nr, nrow, irow);
       for(int jcol = 0; jcol < ncol; ++jcol)
         {
           row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
         }
       
       mgard_gen::prolongate_l( l, row_vec, coords_x, nc, ncol);
       
       for(int jcol = 0; jcol < ncol; ++jcol)
         {
           work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol];
         }
       
     }


      //   std::cout << "recomposing-colsweep2" << "\n";
        // column-sweep, this is the slow one! Need something like column_copy
        if( nrow > 1)
          {
            for(int jcol = 0;  jcol < nc; jcol+= Pstride)
              {
                int jr  = get_lindex(nc,  ncol,  jcol);
                for(int irow = 0;  irow < nrow; ++irow ) //copy all rows
                  {
                    col_vec[irow] = work[mgard_common::get_index(ncol,irow, jr)];
                  }

                mgard_gen::prolongate_l(l,  col_vec, coords_y, nr, nrow);

                for(int irow = 0;  irow < nrow; ++irow )
                  {
                    work[mgard_common::get_index(ncol,irow, jr)] = col_vec[irow] ;
                  }
              }
          }
 }


void prolong_add_2D_last(const int nr, const int nc, const int nrow, const int ncol,  const int l_target,  std::vector<double>& work, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& row_vec, std::vector<double>& col_vec )
 {
   int l = 0;
    
   int stride = 1;std::pow(2,l);//current stride
   //   int Pstride = stride/2;
   
          for(int irow = 0;  irow < nr; ++irow )
          {
            int ir = get_lindex(nr, nrow, irow);
            for(int jcol = 0; jcol < ncol; ++jcol)
              {
                row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
              }

            mgard_gen::prolongate_last(row_vec, coords_x, nc, ncol);

            for(int jcol = 0; jcol < ncol; ++jcol)
              {
                work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol];
              }

          }


    //   //   std::cout << "recomposing-colsweep2" << "\n";
    //     // column-sweep, this is the slow one! Need something like column_copy
        if( nrow > 1)
          {
            for(int jcol = 0;  jcol < ncol; ++jcol)
              {
                // int jr  = get_lindex(nc,  ncol,  jcol);
                for(int irow = 0;  irow < nrow; ++irow ) //copy all rows
                  {
                    col_vec[irow] = work[mgard_common::get_index(ncol,irow, jcol)];
                  }

                mgard_gen::prolongate_last(col_vec, coords_y, nr, nrow);

                for(int irow = 0;  irow < nrow; ++irow )
                  {
                    work[mgard_common::get_index(ncol,irow, jcol)] = col_vec[irow] ;
                  }
              }
          } 
  
 }


void prep_3D(const int nr, const int nc, const int nf, const int nrow, const int ncol, const int nfib,  const int l_target, double* v, std::vector<double>& work, std::vector<double>& work2d, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z )
{
  int l = 0;
  int stride = 1;
  
  std::vector<double> v2d(nrow*ncol), fib_vec(nfib);
  std::vector<double> row_vec(ncol);
  std::vector<double> col_vec(nrow);

  pi_Ql3D_first( nr,  nc,  nf,  nrow,  ncol,  nfib,   l,  v,  coords_x,  coords_y, coords_z,  row_vec,  col_vec,  fib_vec);

  mgard_gen::copy3_level(0,  v,  work.data(),  nrow,  ncol, nfib);  
  mgard_gen::assign3_level_l(0, work.data(),  0.0,  nr,  nc, nf,  nrow,  ncol, nfib);

  for (int kfib = 0; kfib < nfib; kfib += stride)
    {
      mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
      mgard_gen::refactor_2D_first(nr, nc,  nrow, ncol,  l, v2d.data(), work2d, coords_x, coords_y,  row_vec, col_vec);
      mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
    }


  for(int irow = 0; irow < nr; irow += stride)
    {
      int ir = get_lindex(nr, nrow, irow);
        for(int jcol = 0; jcol < nc; jcol += stride)
        {
          int jc  = get_lindex(nc,  ncol,  jcol);
          for(int kfib = 0; kfib < nfib; ++kfib)
            {
              fib_vec[kfib] = work[mgard_common::get_index3(ncol,nfib,ir,jc,kfib)];
            }
          mgard_cannon::mass_matrix_multiply(l, fib_vec, coords_z);
          mgard_gen::restrict_first(fib_vec, coords_z, nf, nfib);
          mgard_gen::solve_tridiag_M_l(l,  fib_vec, coords_z, nf, nfib);
          for(int kfib = 0; kfib < nfib; ++kfib)
            {
              work[mgard_common::get_index3(ncol,nfib,ir,jc,kfib)] = fib_vec[kfib] ;
            }

          
        }
    }
  // for(int irow = 0; irow < nr; irow += stride)
  //   {
  //     //int ir = irow;
  //     int ir = get_lindex(nr, nrow, irow);
  //     for(int kfib = 0; kfib < nf; kfib += stride)
  //       {
  //         int kf = get_lindex(nf, nfib, kfib);
  //         for(int jc = 0; jc < ncol; jc += stride)
  //           {
  //             row_vec[jc] = work[mgard_common::get_index3(ncol,nfib,ir,jc,kf)];
  //           }
  //         //        assign_num_level(0, row_vec, 0.0, nr, nrow);
  //         mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);
  //         mgard_gen::restrict_first(row_vec, coords_x, nr, nrow);
  //         mgard_gen::solve_tridiag_M_l(0,  row_vec, coords_x, nr, nrow);
  //         for(int jcol = 0; jcol < nc; jcol += stride)
  //           {
  //             int jr = get_lindex(nc, ncol, jcol); 
  //             work[mgard_common::get_index3(ncol,nfib,ir,jr,kf)] = row_vec[jr] ;
  //           }
          
  //       }
  //   }

  // for(int jc = 0; jc < nc; jc += stride)
  //   {
  //     int jr = get_lindex(nc, ncol, jc); 
  //     for(int kfib = 0; kfib < nfib; ++kfib)
  //       {
  //         int kf = get_lindex(nf, nfib, kfib);
  //         for(int irow = 0; irow < nrow; irow += stride)
  //           {
  //             col_vec[irow] = work[mgard_common::get_index3(ncol,nfib,irow,jc,kf)];
  //           }
  //         //          assign_num_level(0, col_vec, 0.0, nc, ncol);
  //         mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);
  //         mgard_gen::restrict_first(col_vec, coords_y, nc, ncol);
  //         mgard_gen::solve_tridiag_M_l(0,  col_vec, coords_y, nc, ncol);
  //         for(int irow = 0; irow < nr; irow += stride)
  //           {
  //             int ir = get_lindex(nr, nrow, irow);
  //             work[mgard_common::get_index3(ncol,nfib,ir,jc,kf)] = col_vec[ir] ;
  //           }
          
  //       }
  //   }

  
  

  // for(int irow = 0; irow < nr; irow += stride)
  // //    for(int ir = 0; ir < nrow; ir += stride)
  //   {
  //     int ir = get_lindex(nr, nrow, irow);
  //     //      for(int jc = 0; jc < ncol; jc += stride)
  //       for(int jcol = 0; jcol < nc; jcol += stride)
  //       {
  //         int jc  = get_lindex(nc,  ncol,  jcol);
  //         for(int kfib = 0; kfib < nfib; ++kfib)
  //           {
  //             fib_vec[kfib] = work[mgard_common::get_index3(ncol,nfib,ir,jc,kfib)];
  //           }
  //         //          assign_num_level(0, fib_vec, 0.0, nf, nfib);
  //         mgard_cannon::mass_matrix_multiply(l, fib_vec, coords_z);
  //         mgard_gen::restrict_first(fib_vec, coords_z, nf, nfib);
  //         mgard_gen::solve_tridiag_M_l(l,  fib_vec, coords_z, nf, nfib);
  //         for(int kfib = 0; kfib < nf; ++kfib)
  //           {
  //             int kf = get_lindex(nf, nfib, kfib);
  //             work[mgard_common::get_index3(ncol,nfib,ir,jc,kf)] = fib_vec[kf] ;
  //           }
          
  //       }
  //   }

  add3_level_l(0, v, work.data(),  nr,  nc, nf,  nrow,  ncol, nfib);
       
}
  




void recompose_3D(const int nr, const int nc, const int nf, const int nrow, const int ncol, const int nfib,  const int l_target, double* v, std::vector<double>& work, std::vector<double>& work2d, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z )
{
        // recompose

    std::vector<double> v2d(nrow*ncol), fib_vec(nfib);
    std::vector<double> row_vec(ncol);
    std::vector<double> col_vec(nrow);

    std::cout << "recomposing" << "\n";
    for (int l = l_target ; l > 0 ; --l)
      {

        int stride = std::pow(2,l);//current stride
        int Pstride = stride/2;

        mgard_gen::copy3_level_l(l-1,  v,  work.data(),  nr,  nc, nf,  nrow,  ncol, nfib);

        mgard_gen::assign3_level_l(l, work.data(),  0.0,  nr,  nc, nf,  nrow,  ncol, nfib);



        //        for (int kfib = 0; kfib < nfib; ++kfib)
        for (int kfib = 0; kfib < nf; kfib += Pstride)
          {
            //    int kf =kfib;
            int kf = get_lindex(nf, nfib, kfib); 
            mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
            //            mgard_gen::compute_zl(nr, nc, nrow, ncol, l,  work2d, coords_x, coords_y, row_vec, col_vec);
            mgard_gen::refactor_2D(nr, nc,  nrow, ncol,  l-1, v2d.data(), work2d, coords_x, coords_y,  row_vec, col_vec);
            mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
          }


        for(int irow = 0; irow < nr; irow += stride)
          {
            int ir = get_lindex(nr, nrow, irow);
            for(int jcol = 0; jcol < nc; jcol += stride)
              {
                int jc  = get_lindex(nc,  ncol,  jcol);
                for(int kfib = 0; kfib < nfib; ++kfib)
                  {
                    fib_vec[kfib] = work[mgard_common::get_index3(ncol,nfib,ir,jc,kfib)];
                  }

                mgard_gen::mass_mult_l(l-1, fib_vec, coords_z, nf, nfib );
                
                mgard_gen::restrict_l(l, fib_vec, coords_z, nf, nfib );
                
                mgard_gen::solve_tridiag_M_l(l, fib_vec, coords_z, nf, nfib );

                for(int kfib = 0; kfib < nfib; ++kfib)
                  {
                    work[mgard_common::get_index3(ncol,nfib,ir,jc,kfib)] = fib_vec[kfib] ;
                  }
               
              }
          }


        //- computed zl -//

        
        sub3_level_l(l, work.data(), v,  nr,  nc,  nf, nrow,  ncol, nfib); //do -(Qu - zl)
        
        //        for (int is = 0; is < nfib; ++is)
        for (int kfib = 0; kfib < nf; kfib += stride)
          {
            int kf = get_lindex(nf, nfib, kfib);
            //            int kf = kfib;
            mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
            mgard_gen::prolong_add_2D(nr, nc, nrow, ncol, l,   work2d, coords_x, coords_y, row_vec, col_vec);
            mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
          }

        for(int irow = 0; irow < nr; irow += Pstride)
          {
            int ir = get_lindex(nr, nrow, irow);
            for(int jcol = 0; jcol < nc; jcol += Pstride)
              {
                int jc  = get_lindex(nc,  ncol,  jcol);
                for(int kfib = 0; kfib < nfib; ++kfib)
                  {
                    fib_vec[kfib] = work[mgard_common::get_index3(ncol,nfib,ir,jc,kfib)];
                  }

                mgard_gen::prolongate_l(l, fib_vec, coords_z, nf, nfib);

          
                for(int kfib = 0; kfib < nfib; ++kfib)
                  {
                    work[mgard_common::get_index3(ncol,nfib,ir,jc,kfib)] = fib_vec[kfib] ;
                  }

              }
          }

        mgard_gen::assign3_level_l(l, v,  0.0,  nr,  nc, nf,  nrow,  ncol, nfib);
        mgard_gen::sub3_level_l(l-1, v, work.data(),   nr,  nc,  nf, nrow,  ncol, nfib);
        
  }
}




void postp_3D(const int nr, const int nc, const int nf, const int nrow, const int ncol, const int nfib,  const int l_target, double* v, std::vector<double>& work, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z)
{
  std::vector<double> work2d(nrow*ncol), fib_vec(nfib),v2d(nrow*ncol);
    std::vector<double> row_vec(ncol);
    std::vector<double> col_vec(nrow);

    int l = 0;
    int stride = 1;//current stride
    int Pstride = stride/2;

    //mgard_gen::copy3_level_l(l,  v,  work.data(),  nrow,  ncol, nfib,  nrow,  ncol, nfib);
    mgard_gen::copy3_level(l,  v,  work.data(),  nrow,  ncol, nfib);
    mgard_gen::assign3_level_l(l, work.data(),  0.0,  nr,  nc, nf,  nrow,  ncol, nfib);

      for (int kfib = 0; kfib < nfib; kfib += stride)
    {
      mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
      mgard_gen::refactor_2D_first(nr, nc,  nrow, ncol,  l, v2d.data(), work2d, coords_x, coords_y,  row_vec, col_vec);
      mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kfib);
    }


  for(int irow = 0; irow < nr; irow += stride)
    {
      int ir = get_lindex(nr, nrow, irow);
        for(int jcol = 0; jcol < nc; jcol += stride)
        {
          int jc  = get_lindex(nc,  ncol,  jcol);
          for(int kfib = 0; kfib < nfib; ++kfib)
            {
              fib_vec[kfib] = work[mgard_common::get_index3(ncol,nfib,ir,jc,kfib)];
            }
          mgard_cannon::mass_matrix_multiply(l, fib_vec, coords_z);
          mgard_gen::restrict_first(fib_vec, coords_z, nf, nfib);
          mgard_gen::solve_tridiag_M_l(l,  fib_vec, coords_z, nf, nfib);
          for(int kfib = 0; kfib < nfib; ++kfib)
            {
              work[mgard_common::get_index3(ncol,nfib,ir,jc,kfib)] = fib_vec[kfib] ;
            }

          
        }
    }

    // for(int irow = 0; irow < nr; irow += 1)
    //   {
    //     //        int ir = irow;
    //     int ir  = get_lindex(nr,  nrow,  irow);
    //     for(int kfib = 0; kfib < nf; ++kfib)
    //       {
    //         //    int kf = kfib;
    //         int kf = get_lindex(nf, nfib, kfib); 
    //         for(int jcol = 0; jcol < ncol; jcol += stride)
    //           {
    //             row_vec[jcol] = work[mgard_common::get_index3(ncol,nfib,ir,jcol,kf)];
    //           }
            
    //         //            mgard_gen::mass_mult_l(l, fib_vec, coords_z, nfib, nfib );
    //         //            assign_num_level(0, row_vec, 0.0, nr, nrow);
    //         mgard_cannon::mass_matrix_multiply(l, row_vec, coords_x);
    //         mgard_gen::restrict_first(row_vec, coords_x, nr, nrow );
            
    //         mgard_gen::solve_tridiag_M_l(l, row_vec, coords_x, nr, nrow );

    //         for(int jcol = 0; jcol < nc; jcol += stride)
    //           {
    //             int jr = get_lindex(nc, ncol, jcol);
    //             work[mgard_common::get_index3(ncol,nfib,ir,jr,kf)] = row_vec[jr] ;
    //           }
            
    //       }
    //   }


    // for(int jcol = 0; jcol < nc; jcol += stride)
    //   {
    //     int jc = get_lindex(nc, ncol, jcol); 
    //     for(int kfib = 0; kfib < nf; ++kfib)
    //       {
    //         //            int kf = kfib;
    //         int kf = get_lindex(nf, nfib, kfib); 
    //         for(int ir = 0; ir < nrow; ir += stride)
    //           {
    //             col_vec[ir] = work[mgard_common::get_index3(ncol,nfib,ir,jc,kf)];
    //           }
    //         //            assign_num_level(0, col_vec, 0.0, nc, ncol);
    //         mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);
    //         mgard_gen::restrict_first(col_vec, coords_y, nc, ncol);
    //         mgard_gen::solve_tridiag_M_l(0,  col_vec, coords_y, nc, ncol);
    //         for(int irow = 0; irow < nr; irow += stride)
    //           {
    //             int ir = get_lindex(nr, nrow, irow);
    //             work[mgard_common::get_index3(ncol,nfib,ir,jc,kf)] = col_vec[ir] ;
    //           }
            
    //       }
    //   }


    
    

    // for(int irow = 0; irow < nr; irow += stride)
    //   {
    //     int ir  = get_lindex(nr,  nrow,  irow);
    //     for(int jcol = 0; jcol < nc; jcol += stride)
    //       {
    //         int jc  = get_lindex(nc,  ncol,  jcol);
    //         for(int kfib = 0; kfib < nfib; ++kfib)
    //           {
    //             fib_vec[kfib] = work[mgard_common::get_index3(ncol,nfib,ir,jc,kfib)];
    //           }
            
    //         //            mgard_gen::mass_mult_l(l, fib_vec, coords_z, nfib, nfib );
    //         //            assign_num_level(0, fib_vec, 0.0, nf, nfib);
    //         mgard_cannon::mass_matrix_multiply(l, fib_vec, coords_z);
    //         mgard_gen::restrict_first(fib_vec, coords_z, nf, nfib );
            
    //         mgard_gen::solve_tridiag_M_l(l, fib_vec, coords_z, nf, nfib );
            
    //         for(int kfib = 0; kfib < nf; ++kfib)
    //           {
    //             int kf = get_lindex(nf, nfib, kfib);
    //             work[mgard_common::get_index3(ncol,nfib,ir,jc,kf)] = fib_vec[kf] ;
                
    //           }
            
    //       }
    //   }


    //- computed zl -//

        
    sub3_level_l(0, work.data(), v,  nr,  nc,  nf, nrow,  ncol, nfib); //do -(Qu - zl)
        
    //    for (int kf = 0; kf < nfib; ++kf)
    for (int kfib = 0; kfib < nf; kfib += stride)
      {
        int kf = get_lindex(nf, nfib, kfib);

        mgard_common::copy_slice(work.data(), work2d, nrow, ncol, nfib, kf);
        mgard_gen::prolong_add_2D_last(nr, nc, nrow, ncol, l,   work2d, coords_x, coords_y, row_vec, col_vec);
        mgard_common::copy_from_slice(work.data(), work2d, nrow, ncol, nfib, kf);
      }
    
    for(int irow = 0; irow < nrow; irow += stride)
      {
        for(int jcol = 0; jcol < ncol; jcol += stride)
          {

            for(int kfib = 0; kfib < nfib; ++kfib)
              {
                fib_vec[kfib] = work[mgard_common::get_index3(ncol,nfib,irow,jcol,kfib)];
              }
            
            mgard_gen::prolongate_last(fib_vec, coords_z, nf, nfib);
            
            for(int kfib = 0; kfib < nfib; ++kfib)
              {
                work[mgard_common::get_index3(ncol,nfib,irow,jcol,kfib)] = fib_vec[kfib] ;
              }
            
          }
      }
    
    mgard_gen::assign3_level_l(0, v,  0.0,  nr,  nc, nf,  nrow,  ncol, nfib);
    mgard_gen::sub3_level(0, v, work.data(), nrow,  ncol,  nfib);
    //    mgard_gen::sub3_level(l, v, work.data(), nrow,  ncol,  nfib);
        
  
}



void recompose_2D(const int nr, const int nc, const int nrow, const int ncol,  const int l_target, double* v, std::vector<double>& work, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& row_vec, std::vector<double>& col_vec )
  {
        // recompose
    std::cout << "recomposing" << "\n";
    //    for (int l = l_target ; l > 0 ; --l)
    //  {
    int l = l_target;
    
        int stride = std::pow(2,l);//current stride
        int Pstride = stride/2;

        //        copy_level_l(l-1,  v,  work.data(),  nr,  nc,  nrow,  ncol);

        //assign_num_level_l(l, work.data(),  0.0,  nr,  nc,  nrow,  ncol);

        std::cout << "recomposing-rowsweep" << "\n";
        //  l = 0;
        // row-sweep
        for(int irow = 0;  irow < nr; ++irow)
          {
            int ir = get_lindex(nr, nrow, irow);
            for(int jcol = 0; jcol < ncol; ++jcol)
              {
                row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
              }


            mgard_gen::mass_mult_l(l-1, row_vec, coords_x, nc, ncol );

            mgard_gen::restrict_l(l, row_vec, coords_x, nc, ncol );

            mgard_gen::solve_tridiag_M_l(l, row_vec, coords_x, nc, ncol );


            for(int jcol = 0; jcol < ncol; ++jcol)
              {
                work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol] ;
              }

          }

      //   std::cout << "recomposing-colsweep" << "\n";

        // column-sweep, this is the slow one! Need something like column_copy
        if( nrow > 1) // check if we have 1-D array..
          {
            for(int jcol = 0;  jcol < nc; jcol += stride)
              {
                int jr  = get_lindex(nc,  ncol,  jcol);
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    col_vec[irow] = work[mgard_common::get_index(ncol, irow, jr )];
                  }

                
                mgard_gen::mass_mult_l(l-1, col_vec, coords_y, nr, nrow );
            
                mgard_gen::restrict_l(l, col_vec, coords_y, nr, nrow );

                mgard_gen::solve_tridiag_M_l(l, col_vec, coords_y, nr, nrow );
            

                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    work[mgard_common::get_index(ncol, irow, jr )] = col_vec[irow] ;
                  }
              }
          }
        //        subtract_level_l(l, work.data(), v,  nr,  nc,  nrow,  ncol); //do -(Qu - zl)
        std::cout << "recomposing-rowsweep2" << "\n";

      //   //int Pstride = stride/2; //finer stride

      //   // row-sweep
        for(int irow = 0;  irow < nr; irow += stride )
          {
            int ir = get_lindex(nr, nrow, irow);
            for(int jcol = 0; jcol < ncol; ++jcol)
              {
                row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
              }

            mgard_gen::prolongate_l( l, row_vec, coords_x, nc, ncol);

            for(int jcol = 0; jcol < ncol; ++jcol)
              {
                work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol];
              }

          }


      //   std::cout << "recomposing-colsweep2" << "\n";
        // column-sweep, this is the slow one! Need something like column_copy
        if( nrow > 1)
          {
            for(int jcol = 0;  jcol < nc; jcol+= Pstride)
              {
                int jr  = get_lindex(nc,  ncol,  jcol);
                for(int irow = 0;  irow < nrow; ++irow ) //copy all rows
                  {
                    col_vec[irow] = work[mgard_common::get_index(ncol,irow, jr)];
                  }

                mgard_gen::prolongate_l(l,  col_vec, coords_y, nr, nrow);

                for(int irow = 0;  irow < nrow; ++irow )
                  {
                    work[mgard_common::get_index(ncol,irow, jr)] = col_vec[irow] ;
                  }
              }
          }
        // std::cout << "last step" << "\n";

        // assign_num_level_l(l, v, 0.0, nr, nc, nrow, ncol);
        // subtract_level_l(l-1, v, work.data(),  nr,  nc,  nrow,  ncol);
        //      }
  }





void postp_2D(const int nr, const int nc, const int nrow, const int ncol,  const int l_target, double* v, std::vector<double>& work, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& row_vec, std::vector<double>& col_vec )
  {
    mgard_cannon::copy_level(nrow,  ncol, 0,  v,  work);

    assign_num_level_l(0, work.data(),  0.0,  nr,  nc,  nrow,  ncol);

    for(int irow = 0;  irow < nrow; ++irow)
      {
        //        int ir = get_lindex(nr, nrow, irow);
        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            row_vec[jcol] = work[mgard_common::get_index(ncol, irow, jcol)];
          }
        
        mgard_cannon::mass_matrix_multiply(0, row_vec, coords_x);
        
        restrict_first(row_vec, coords_x, nc, ncol);
        
        
        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            work[mgard_common::get_index(ncol, irow, jcol)] = row_vec[jcol] ;
          }

      }

    for(int irow = 0;  irow < nr; ++irow)
      {
        int ir = get_lindex(nr, nrow, irow);
        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
          }
        
        mgard_gen::solve_tridiag_M_l(0, row_vec, coords_x, nc, ncol );

        for(int jcol = 0; jcol < ncol; ++jcol)
          {
            work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol] ;
          }
      }

    //   //   std::cout << "recomposing-colsweep" << "\n";

    //     // column-sweep, this is the slow one! Need something like column_copy
        if( nrow > 1) // check if we have 1-D array..
          {
            for(int jcol = 0;  jcol < ncol; ++jcol)
              {
                //      int jr  = get_lindex(nc,  ncol,  jcol);
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    col_vec[irow] = work[mgard_common::get_index(ncol, irow, jcol )];
                  }

                
                mgard_cannon::mass_matrix_multiply(0, col_vec, coords_y);
                
                mgard_gen::restrict_first(col_vec, coords_y, nr, nrow);

            

                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    work[mgard_common::get_index(ncol, irow, jcol)] = col_vec[irow] ;
                  }
              }
            
            for(int jcol = 0;  jcol < nc; ++jcol)
              {
                int jr  = get_lindex(nc,  ncol,  jcol);
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                    col_vec[irow] = work[mgard_common::get_index(ncol, irow, jr )];
                  }

                mgard_gen::solve_tridiag_M_l(0, col_vec, coords_y, nr, nrow );
                for(int irow = 0;  irow < nrow; ++irow)
                  {
                   work[mgard_common::get_index(ncol, irow, jr )] = col_vec[irow];
                  }
              }
          }



        subtract_level_l(0, work.data(), v,  nr,  nc,  nrow,  ncol); //do -(Qu - zl)
        std::cout << "recomposing-rowsweep2" << "\n";


    //     //   //int Pstride = stride/2; //finer stride

    //   //   // row-sweep
        for(int irow = 0;  irow < nr; ++irow )
          {
            int ir = get_lindex(nr, nrow, irow);
            for(int jcol = 0; jcol < ncol; ++jcol)
              {
                row_vec[jcol] = work[mgard_common::get_index(ncol, ir, jcol)];
              }

            mgard_gen::prolongate_last(row_vec, coords_x, nc, ncol);

            for(int jcol = 0; jcol < ncol; ++jcol)
              {
                work[mgard_common::get_index(ncol, ir, jcol)] = row_vec[jcol];
              }

          }


    //   //   std::cout << "recomposing-colsweep2" << "\n";
    //     // column-sweep, this is the slow one! Need something like column_copy
        if( nrow > 1)
          {
            for(int jcol = 0;  jcol < ncol; ++jcol)
              {
                // int jr  = get_lindex(nc,  ncol,  jcol);
                for(int irow = 0;  irow < nrow; ++irow ) //copy all rows
                  {
                    col_vec[irow] = work[mgard_common::get_index(ncol,irow, jcol)];
                  }

                mgard_gen::prolongate_last(col_vec, coords_y, nr, nrow);

                for(int irow = 0;  irow < nrow; ++irow )
                  {
                    work[mgard_common::get_index(ncol,irow, jcol)] = col_vec[irow] ;
                  }
              }
          }
        
        //     std::cout << "last step" << "\n";

        assign_num_level_l(0, v, 0.0, nr, nc, nrow, ncol);
        mgard_cannon::subtract_level(nrow,  ncol, 0, v, work.data() );
  }



void qwrite_2D_l(const int nr, const int nc, const int nrow, const int ncol, const int nlevel,  const int  l,   double* v, double tol, double norm, const std::string outfile)
  {

    int stride = std::pow(2,l);//current stride
    int Cstride = 2*stride;
    tol /=  (double) (nlevel + 1);

    double coeff = 2.0*norm*tol;
    std::cout << "Quantization factor: "<<coeff <<"\n";

    gzFile out_file = gzopen(outfile.c_str(), "w6b");
    int prune_count = 0;
    gzwrite(out_file, &coeff, sizeof(double));

    // level L+1, finest first level
    for(int irow =  0; irow < nr; ++irow) //loop over the logical array
      {
        int ir  = get_lindex(nr,  nrow,  irow);
        
        for(int jcol = 0; jcol < nc-1; ++jcol)
          {
            int jr  = get_lindex(nc,  ncol,  jcol);
            int jrP = get_lindex(nc,  ncol, jcol+1);

            if (jrP != jr+1) // next real memory location was jumped over, so this is level L+1
              {
                int quantum = (int) (v[mgard_common::get_index(ncol, ir, jr + 1 )]/coeff);
                if (quantum == 0) ++prune_count;
                gzwrite(out_file, &quantum, sizeof(int));
              }
          }


      }

    for (int jcol = 0; jcol < nc; ++jcol)
      {
        int jr  = get_lindex(nc,  ncol,  jcol);
        for(int irow =  0; irow < nr-1; ++irow) //loop over the logical array
          {
            int ir  = get_lindex(nr,  nrow,  irow);
            int irP  = get_lindex(nr,  nrow,  irow+1);
            if (irP != ir+1) // next real memory location was jumped over, so this is level L+1
              {
                int quantum = (int) (v[mgard_common::get_index(ncol, ir+1, jr )]/coeff);
                if (quantum == 0) ++prune_count;
                gzwrite(out_file, &quantum, sizeof(int));
              }
          }
      }



        for(int irow = 0;  irow < nr-1; ++irow) 
          {
            int ir  = get_lindex(nr, nrow, irow);
            int irP = get_lindex(nr,  nrow,  irow+1);
            
            for(int jcol = 0;  jcol < nc-1; ++jcol ) 
              {
                int jr = get_lindex(nc, ncol, jcol); 
                int jrP = get_lindex(nc, ncol, jcol+1); 
                if ((irP != ir+1) && (jrP != jr+1)) //we skipped both a row and a column
                  {
                    int quantum = (int) (v[mgard_common::get_index(ncol, ir+1, jr+1)]/coeff) ;
                    if (quantum == 0) ++prune_count;
                    gzwrite(out_file, &quantum, sizeof(int));
                  }
              }
          }
    
    
    //levels from L->0 in 2^k+1
    for (int l = 0; l <= nlevel; l++)
      {
        int stride = std::pow(2,l);
        int Cstride = stride*2;
        int row_counter = 0;

        for(int irow = 0;  irow < nr; irow += stride)
          {
            int ir  = get_lindex(nr,  nrow,  irow);
            if( row_counter % 2 == 0 && l != nlevel)
              {
                for(int jcol = Cstride;  jcol < nc; jcol += Cstride)
                  {
                    int jr  = get_lindex(nc,  ncol,  jcol);
                    int quantum = (int) (v[mgard_common::get_index(ncol, ir, jr - stride)]/coeff);
                    if (quantum == 0) ++prune_count;
                    gzwrite(out_file, &quantum, sizeof(int));
                  }

              } else
              {
                for(int jcol = 0;  jcol < nc; jcol += stride)
                  {
                    int jr  = get_lindex(nc,  ncol,  jcol);
                    int quantum = (int) (v[mgard_common::get_index(ncol, ir, jr)]/coeff);
                    if (quantum == 0) ++prune_count;
                    gzwrite(out_file, &quantum, sizeof(int));
                  }

              }
            ++row_counter;
          }
      }
    

    std::cout << "Pruned : "<< prune_count << " Reduction : " << (double) nrow*ncol /(nrow*ncol - prune_count) << "\n";
    gzclose(out_file);


  }


void quantize_3D(const int nr, const int nc, const int nf, const int nrow, const int ncol, const int nfib, const int nlevel,  double* v, std::vector<int>& work, const std::vector<double>& coords_x, const std::vector<double>& coords_y, const std::vector<double>& coords_z, double s, double norm, double tol)
    {
    double coeff = norm*tol; 
    std::memcpy (work.data(), &coeff, sizeof (double));
    int size_ratio = sizeof (double) / sizeof (int);
    int prune_count = 0;

    //    double s = 0.0;
    int count = 0;
    count += size_ratio;
     //level -1, first level for non 2^k+1

    double dx = mgard_gen::get_h_l(coords_x, ncol, ncol, 0, 1);
    double dy = mgard_gen::get_h_l(coords_y, nrow, nrow, 0, 1);
    double dz = mgard_gen::get_h_l(coords_z, nfib, nfib, 0, 1);
    
    double vol =  std::sqrt(dx*dy*dz);
    vol *= std::pow(2, s*(nlevel+1)); //2^-2sl with l=0, s = 0.5
    //    vol = 1;
    std::cout << "Volume -1: " << vol << std::endl;
    //std::cout << "quantizer "  << coeff << std::endl;
    
    for(int kfib = 0; kfib < nf - 1; ++kfib)
      {
        int kf  = mgard_gen::get_lindex(nf, nfib, kfib);
        int kfp = mgard_gen::get_lindex(nf, nfib, kfib + 1);
        
        if( kfp != kf + 1)//skipped a plane
          {
            for(int irow = 0;  irow < nrow; ++irow )
              {
                for(int jcol = 0;  jcol < ncol; ++jcol )
                  {
                    double val = v[mgard_common::get_index3(ncol, nfib, irow, jcol, kf + 1)];
                    int quantum =  (int)(val/(coeff/vol));
                    //std::cout << "quantized "  << val << std::endl;
                    work[count] = quantum;
                    ++count;
                  }
              }
          }
      }

    
    int count_row = 0;
    int count_col = 0;
    int count_sol = 0;
    

    for (int kfib = 0; kfib < nf ; ++kfib)
      {
        int kf  = mgard_gen::get_lindex(nf, nfib, kfib);
        for(int irow = 0;  irow < nr - 1 ; ++irow )
          {
            int ir = mgard_gen::get_lindex(nr, nrow, irow);
            int irP = mgard_gen::get_lindex(nr, nrow, irow + 1);
            if( irP != ir + 1) //skipped a row
              {
                //  std::cout <<"Skipped row: "  << ir + 1 << "\n";
                for(int jcol = 0;  jcol < ncol; ++jcol )
                  {
                    double val = v[mgard_common::get_index3(ncol, nfib, ir + 1, jcol, kf)];
                    int quantum =  (int)(val/(coeff/vol));
                    //                    std::cout << "quantized "  << val << std::endl;
                    work[count] = quantum;
                    ++count_row;
                    ++count;
                  }
              }
          }
        
        for(int irow = 0;  irow < nr  ; ++irow )
          {
            int ir = mgard_gen::get_lindex(nr, nrow, irow);
            
            //      std::cout <<"Non skipped row: "  << ir  << "\n";
            for(int jcol = 0;  jcol < nc - 1 ; ++jcol)
              {
                int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                int jcP  = mgard_gen::get_lindex(nc,  ncol,  jcol+1);
                if(jcP != jc + 1)//skipped a column
                  {
                    double val = v[mgard_common::get_index3(ncol, nfib, ir, jc + 1, kf)];
                    int quantum =  (int)(val/(coeff/vol));
                    work[count] = quantum;
                    ++count_col;
                    ++count;
                    //                    std::cout <<"Skipped col: " << ir << "\t" << jc + 1 << "\n";
                  }
              }
          }
      }

    //std::cout << "Wrote : "<< count_row <<"\t" << count_col << "\t" << count_sol << "\n";
    

    
    // // 2^k+1 part //
    
    for(int ilevel = 0; ilevel < nlevel ; ++ilevel)
      {
        int stride = std::pow(2,ilevel);
        int Cstride = 2*stride;

        int fib_counter = 0;

        double dx = get_h_l(coords_x, nc, ncol, 0, stride);
        double dy = get_h_l(coords_y, nr, nrow, 0, stride);
        double dz = get_h_l(coords_z, nf, nfib, 0, stride);
        
        double vol =  std::sqrt(dx*dy*dz);
	vol *= std::pow(2, s*(nlevel-ilevel)); //2^-2sl with l=0, s = 0.5
        std::cout << "Volume : " << ilevel << "\t"<< vol << std::endl;
        // std::cout << "Stride : " << stride << "\t"<< vol << std::endl;
        
        for(int kfib = 0; kfib < nf; kfib += stride)
          {
            int kf = mgard_gen::get_lindex(nf, nfib, kfib);
            int row_counter = 0;

            if(fib_counter % 2 == 0)
              {
              for(int irow = 0;  irow < nr; irow += stride)
                {
                  int ir = mgard_gen::get_lindex(nr, nrow, irow);
                  if( row_counter % 2 == 0)
                    {
                      for(int jcol = Cstride;  jcol < nc; jcol += Cstride)
                        {
                          int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                          double val = v[mgard_common::get_index3(ncol, nfib, ir,jc - stride, kf)];
                          int quantum =  (int)(val/(coeff/vol));
                          work[count] = quantum;                                              
                          ++count;
                          //                          outfile.write(reinterpret_cast<char*>( &v[mgard_common::get_index3(ncol, nfib, ir,jc - stride, kf)] ), sizeof(double) ); 
                        //                  std::cout <<  v[irow][icol - stride] << "\t";
                        }
                    
                    }else
                    {
                      for(int jcol = 0;  jcol < nc; jcol += stride)
                        {
                          int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                          double val = v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)];
                          int quantum =  (int)(val/(coeff/vol));
                          work[count] = quantum;
                          ++count;
                          //         outfile.write(reinterpret_cast<char*>( &v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)] ), sizeof(double) ); 
                    //                  std::cout <<  v[irow][icol] << "\t";
                        }
                
                    }
                  ++row_counter;
                }
              }
            else{
              for(int irow = 0;  irow < nr; irow += stride)
                {
                  int ir = mgard_gen::get_lindex(nr, nrow, irow);
                  for(int jcol = 0;  jcol < nc; jcol += stride) 
                    {
                      int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                      double val = v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)];
                      int quantum =  (int)(val/(coeff/vol));
                      work[count] = quantum;
                      ++count;
                      //                      outfile.write(reinterpret_cast<char*>( &v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)] ), sizeof(double) );
                    }
                }
            }
            ++fib_counter;
          }
      }
        


    // last level -> L=0
    int stride = std::pow(2,nlevel);
    dx = get_h_l(coords_x, nc, ncol, 0, stride);
    dy = get_h_l(coords_y, nr, nrow, 0, stride);
    dz = get_h_l(coords_z, nf, nfib, 0, stride);
    
    vol =  std::sqrt(dx*dy*dz);
    //    vol *= std::pow(2, 0);
    std::cout << "Volume : " << nlevel << "\t"<< vol << std::endl;
    // std::cout << "Stride : " << stride << "\t"<< vol << std::endl;
    
    for(int irow = 0;  irow < nr; irow += stride)
      {
        int ir = mgard_gen::get_lindex(nr, nrow, irow);
        for(int jcol = 0;  jcol < nc; jcol += stride) 
          {
            int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
            for(int kfib = 0; kfib < nf; kfib += stride)
              {
                int kf = mgard_gen::get_lindex(nf, nfib, kfib);
                double val = v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)];
                int quantum =  (int)(val/(coeff/vol));
                work[count] = quantum;
                ++count;
              }
          }
      }

    std::cout << "Wrote out: " << count <<"\n";
  }


void dequantize_3D(const int nr, const int nc, const int nf, const int nrow, const int ncol, const int nfib, const int nlevel, double* v, std::vector<int>& work , const std::vector<double>& coords_x, const std::vector<double>& coords_y, const std::vector<double>& coords_z, double s)
  {
    
    int size_ratio = sizeof (double) / sizeof (int);
    double q; //quantizing factor
    
    std::memcpy (&q, work.data(), sizeof (double));
       
    
    double dx = mgard_gen::get_h_l(coords_x, ncol, ncol, 0, 1);
    double dy = mgard_gen::get_h_l(coords_y, nrow, nrow, 0, 1);
    double dz = mgard_gen::get_h_l(coords_z, nfib, nfib, 0, 1);

    double vol =  std::sqrt(dx*dy*dz); 
    vol *= std::pow(2, s*(nlevel+1)); //2^-2sl with l=0, s = 0.5

    int imeg = 0 ; //mega-counter

    imeg += size_ratio;
    
    for(int kfib = 0; kfib < nf - 1; ++kfib)
      {
        int kf  = mgard_gen::get_lindex(nf, nfib, kfib);
        int kfp = mgard_gen::get_lindex(nf, nfib, kfib + 1);
        
        if( kfp != kf + 1)//skipped a plane
          {
            for(int irow = 0;  irow < nrow; ++irow )
              {
                for(int jcol = 0;  jcol < ncol; ++jcol )
                  {
                    double val = (double) work[imeg];
                    v[mgard_common::get_index3(ncol, nfib, irow, jcol, kf + 1)] = q*val/vol ;
                    ++imeg;
                   
                  }
              }
          }
      }

    
    int count_row = 0;
    int count_col = 0;
    int count_sol = 0;
    

    for (int kfib = 0; kfib < nf ; ++kfib)
      {
        int kf  = mgard_gen::get_lindex(nf, nfib, kfib);
        for(int irow = 0;  irow < nr - 1 ; ++irow )
          {
            int ir = mgard_gen::get_lindex(nr, nrow, irow);
            int irP = mgard_gen::get_lindex(nr, nrow, irow + 1);
            if( irP != ir + 1) //skipped a row
              {
                //  std::cout <<"Skipped row: "  << ir + 1 << "\n";
                for(int jcol = 0;  jcol < ncol; ++jcol )
                  {
                    double val = work[imeg];
                    v[mgard_common::get_index3(ncol, nfib, ir + 1, jcol, kf)] = q*val/vol ;
                    ++imeg; 
                  }
              }
          }
        
        for(int irow = 0;  irow < nr  ; ++irow )
          {
            int ir = mgard_gen::get_lindex(nr, nrow, irow);
            
            //      std::cout <<"Non skipped row: "  << ir  << "\n";
            for(int jcol = 0;  jcol < nc - 1 ; ++jcol)
              {
                int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                int jcP  = mgard_gen::get_lindex(nc,  ncol,  jcol+1);
                if(jcP != jc + 1)//skipped a column
                  {
                    double val = (double) work[imeg];
                    v[mgard_common::get_index3(ncol, nfib, ir, jc + 1, kf)] = q*val/vol ;
                    ++imeg;

                  }
              }
          }
      }

    //std::cout << "Wrote : "<< count_row <<"\t" << count_col << "\t" << count_sol << "\n";
    

    
    // // 2^k+1 part //
    
    for(int ilevel = 0; ilevel < nlevel ; ++ilevel)
      {
        int stride = std::pow(2,ilevel);
        int Cstride = 2*stride;

        int fib_counter = 0;

        double dx = get_h_l(coords_x, nc, ncol, 0, stride);
        double dy = get_h_l(coords_y, nr, nrow, 0, stride);
        double dz = get_h_l(coords_z, nf, nfib, 0, stride);
        
        double vol =  std::sqrt(dx*dy*dz);
	vol *= std::pow(2, s*(nlevel-ilevel)); //2^-2sl with l=0, s = 0.5
        std::cout << "Volume : " << ilevel << "\t"<< vol << std::endl;
        std::cout << "Stride : " << stride << "\t"<< vol << std::endl;
        
        for(int kfib = 0; kfib < nf; kfib += stride)
          {
            int kf = mgard_gen::get_lindex(nf, nfib, kfib);
            int row_counter = 0;

            if(fib_counter % 2 == 0)
              {
              for(int irow = 0;  irow < nr; irow += stride)
                {
                  int ir = mgard_gen::get_lindex(nr, nrow, irow);
                  if( row_counter % 2 == 0)
                    {
                      for(int jcol = Cstride;  jcol < nc; jcol += Cstride)
                        {
                          int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                          double val = (double) work[imeg];
                          v[mgard_common::get_index3(ncol, nfib, ir,jc - stride, kf)] = q*val/vol ;
                          ++imeg; ;
                        }
                    
                    }else
                    {
                      for(int jcol = 0;  jcol < nc; jcol += stride)
                        {
                          int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                          double val = (double) work[imeg];
                          v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)] = q*val/vol ;
                          ++imeg; ;;
                        
                        }
                
                    }
                  ++row_counter;
                }
              }
            else{
              for(int irow = 0;  irow < nr; irow += stride)
                {
                  int ir = mgard_gen::get_lindex(nr, nrow, irow);
                  for(int jcol = 0;  jcol < nc; jcol += stride) 
                    {
                      int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
                      double val = (double) work[imeg];
                      v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)] = q*val/vol ;
                          ++imeg; ;;
                    }
                }
            }
            ++fib_counter;
          }
      }
        


    // last level
    int stride = std::pow(2,nlevel);
    dx = get_h_l(coords_x, nc, ncol, 0, stride);
    dy = get_h_l(coords_y, nr, nrow, 0, stride);
    dz = get_h_l(coords_z, nf, nfib, 0, stride);
    
    vol =  std::sqrt(dx*dy*dz);
    std::cout << "Volume : " << nlevel << "\t"<< vol << std::endl;
    std::cout << "Stride : " << stride << "\t"<< vol << std::endl;
    
    for(int irow = 0;  irow < nr; irow += stride)
      {
        int ir = mgard_gen::get_lindex(nr, nrow, irow);
        for(int jcol = 0;  jcol < nc; jcol += stride) 
          {
            int jc  = mgard_gen::get_lindex(nc,  ncol,  jcol);
            for(int kfib = 0; kfib < nf; kfib += stride)
              {
                int kf = mgard_gen::get_lindex(nf, nfib, kfib);
                double val = (double) work[imeg];
                v[mgard_common::get_index3(ncol, nfib, ir, jc, kf)] = q*val/vol ;
                ++imeg; ;;
              }
          }
      }
    


    std::cout << "Mega count : "<< imeg << "\n";
  
  }
  
}
