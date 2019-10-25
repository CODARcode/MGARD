// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details

// DOUBLE PRECISON version of MGARD API //

#include "mgard.h"

#ifndef MGARD_API_H
#define MGARD_API_H


/// Use this version of mgard_compress to compress your data with a tolerance measured in  relative L-infty norm, version 1 for equispaced grids, 1a for tensor product grids with arbitrary spacing

unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, double tol); // ...  1
unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z ,double tol); // ... 1a

// Use this version of mgard_compress to compress your data with a tolerance measured in  relative s-norm.
//Set s=0 for L2-norm
// 2)
unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, double tol, double s); // ... 2
unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z, double tol, double s); // ... 2a

// Use this version of mgard_compress to compress your data to preserve the error in a given quantity of interest
// Here qoi denotes the quantity of interest  which is a bounded linear functional in s-norm.
// This version recomputes the s-norm of the supplied linear functional every time it is invoked. If the same functional
// is to be reused for different sets of data then you are recommended to use one of the functions below (4, 5) to compute and store the norm and call MGARD using (6).
//

unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, double tol, double (*qoi) (int, int, int, double*), double s ); // ... 3
unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z , double tol, double (*qoi) (int, int, int, double*), double s ); // ... 3a

// Use this version of mgard_compress to compute the  s-norm of a quantity of interest. Store this for further use if you wish to work with the same qoi in the future for different datasets.
double  mgard_compress( int n1, int n2, int n3,  double (*qoi) (int, int, int, std::vector<double>), double s ); // ... 4
double  mgard_compress( int n1, int n2, int n3,  std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z, double (*qoi) (int, int, int, std::vector<double>), double s ); // ... 4a

// c-compatible version
double  mgard_compress( int n1, int n2, int n3,  double (*qoi) (int, int, int, double*), double s ); // ... 5
double  mgard_compress( int n1, int n2, int n3,  std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z , double (*qoi) (int, int, int, double*), double s ); // ... 5a

 // Use this version of mgard_compress to compress your data with a tolerance in -s norm
 // with given s-norm of quantity of interest qoi
unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, double tol, double norm_of_qoi, double s ); // ... 6
unsigned char *mgard_compress(int itype_flag, double  *data, int &out_size, int n1, int n2, int n3, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z , double tol, double norm_of_qoi, double s ); // ... 6a


double  *mgard_decompress(int itype_flag, double& quantizer, unsigned char *data, int data_len, int n1, int n2, int n3); // decompress L-infty compressed data
double  *mgard_decompress(int itype_flag, double& quantizer, unsigned char *data, int data_len, int n1, int n2, int n3, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z ); // decompress L-infty compressed data

double  *mgard_decompress(int itype_flag, double& quantizer, unsigned char *data, int data_len, int n1, int n2, int n3, double s); // decompress s-norm
double  *mgard_decompress(int itype_flag, double& quantizer, unsigned char *data, int data_len, int n1, int n2, int n3, std::vector<double>& coords_x, std::vector<double>& coords_y, std::vector<double>& coords_z, double s); // decompress s-norm



#endif
