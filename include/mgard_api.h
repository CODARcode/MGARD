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


#ifndef MGARD_API_H
#define MGARD_API_H



// MGARD double //

//comments!!!!
/// Use this version of mgard_compress to compress your data with a tolerance measured in  relative L-infty norm

unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, double tol); // ... 
unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z ,double tol); // ... 1a

// Use this version of mgard_compress to compress your data with a tolerance measured in  relative s-norm.
//Set s=0 for L2-norm
// 2)
unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, double tol, double s); // ... 2

// Use this version of mgard_compress to compress your data to preserve the error in a given quantity of interest
// Here qoi denotes the quantity of interest  which is a bounded linear functional in s-norm.
// This version recomputes the s-norm of the supplied linear functional every time it is invoked. If the same functional
// is to be reused for different sets of data then you are recommended to use one of the functions below (4, 5) to compute and store the norm and call MGARD using (6).
// 

unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, double tol, double (*qoi) (int, int, int, double*), double s ); // ... 3

// Use this version of mgard_compress to compute the  s-norm of a quantity of interest. Store this for further use if you wish to work with the same qoi in the future for different datasets.
double  mgard_compress( int n1, int n2, int n3,  double (*qoi) (int, int, int, std::vector<double>), double s ); // ... 4
// c-compatible version
double  mgard_compress( int n1, int n2, int n3,  double (*qoi) (int, int, int, double*), double s ); // ... 5
 
 // Use this version of mgard_compress to compress your data with a tolerance in -s norm
 // with given s-norm of quantity of interest qoi
unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, double tol, double norm_of_qoi, double s ); // ... 6


double  *mgard_decompress(int itype_flag, unsigned char *data, int data_len, int n1, int n2, int n3); // decompress L-infty compressed data

double  *mgard_decompress(int itype_flag, unsigned char *data, int data_len, int n1, int n2, int n3, double s); // decompress s-norm

///// -- Non-uniform grids -- /////

/// Use this version of mgard_compress to compress your data with a tolerance measured in  relative L-infty norm

unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, double tol); // ... 1

// Use this version of mgard_compress to compress your data with a tolerance measured in  relative s-norm.
//Set s=0 for L2-norm
// 2)
unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, double tol, double s); // ... 2

// Use this version of mgard_compress to compress your data to preserve the error in a given quantity of interest
// Here qoi denotes the quantity of interest  which is a bounded linear functional in s-norm.
// This version recomputes the s-norm of the supplied linear functional every time it is invoked. If the same functional
// is to be reused for different sets of data then you are recommended to use one of the functions below (4, 5) to compute and store the norm and call MGARD using (6).
// 

unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, std::vector<double>& coords_x,  std::vector<double>& coords_y,  std::vector<double>& coords_z, double tol, double (*qoi) (int, int, int, double*), double s ); 

// Use this version of mgard_compress to compute the  s-norm of a quantity of interest. Store this for further use if you wish to work with the same qoi in the future for different datasets.
double  mgard_compress( int n1, int n2, int n3,  double (*qoi) (int, int, int, std::vector<double>), double s ); // ... 4
// c-compatible version
double  mgard_compress( int n1, int n2, int n3,  double (*qoi) (int, int, int, double*), double s ); // ... 5
 
 // Use this version of mgard_compress to compress your data with a tolerance in -s norm
 // with given s-norm of quantity of interest qoi
unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, double tol, double norm_of_qoi, double s ); // ... 6


double  *mgard_decompress(int itype_flag, unsigned char *data, int data_len, int n1, int n2, int n3); // decompress L-infty compressed data

double  *mgard_decompress(int itype_flag, unsigned char *data, int data_len, int n1, int n2, int n3, double s); // decompress s-norm






// MGARD float //

#endif


//CORRECT THE spelling of canonical
