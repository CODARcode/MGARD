// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.

#include "mgard_float.h"

#ifndef MGARD_API_FLOAT_H
#define MGARD_API_FLOAT_H

/// FLOAT version of API ///

/// Use this version of mgard_compress to compress your data with a tolerance
/// measured in  relative L-infty norm, version 1 for equispaced grids, 1a for
/// tensor product grids with arbitrary spacing

unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3, float tol); // ...  1
unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3,
                              std::vector<float> &coords_x,
                              std::vector<float> &coords_y,
                              std::vector<float> &coords_z,
                              float tol); // ... 1a

// Use this version of mgard_compress to compress your data with a tolerance
// measured in  relative s-norm.
// Set s=0 for L2-norm
// 2)
unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3, float tol,
                              float s); // ... 2
unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3,
                              std::vector<float> &coords_x,
                              std::vector<float> &coords_y,
                              std::vector<float> &coords_z, float tol,
                              float s); // ... 2a

// Use this version of mgard_compress to compress your data to preserve the
// error in a given quantity of interest Here qoi denotes the quantity of
// interest  which is a bounded linear functional in s-norm. This version
// recomputes the s-norm of the supplied linear functional every time it is
// invoked. If the same functional is to be reused for different sets of data
// then you are recommended to use one of the functions below (4, 5) to compute
// and store the norm and call MGARD using (6).
//

unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3, float tol,
                              float (*qoi)(int, int, int, float *),
                              float s); // ... 3
unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3,
                              std::vector<float> &coords_x,
                              std::vector<float> &coords_y,
                              std::vector<float> &coords_z, float tol,
                              float (*qoi)(int, int, int, float *),
                              float s); // ... 3a

// Use this version of mgard_compress to compute the  s-norm of a quantity of
// interest. Store this for further use if you wish to work with the same qoi in
// the future for different datasets.
float mgard_compress(int n1, int n2, int n3,
                     float (*qoi)(int, int, int, std::vector<float>),
                     float s); // ... 4
float mgard_compress(int n1, int n2, int n3, std::vector<float> &coords_x,
                     std::vector<float> &coords_y, std::vector<float> &coords_z,
                     float (*qoi)(int, int, int, std::vector<float>),
                     float s); // ... 4a

// c-compatible version
float mgard_compress(int n1, int n2, int n3,
                     float (*qoi)(int, int, int, float *), float s); // ... 5
float mgard_compress(int n1, int n2, int n3, std::vector<float> &coords_x,
                     std::vector<float> &coords_y, std::vector<float> &coords_z,
                     float (*qoi)(int, int, int, float *), float s); // ... 5a

// Use this version of mgard_compress to compress your data with a tolerance in
// -s norm with given s-norm of quantity of interest qoi
unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3, float tol,
                              float norm_of_qoi, float s); // ... 6
unsigned char *mgard_compress(int itype_flag, float *data, int &out_size,
                              int n1, int n2, int n3,
                              std::vector<float> &coords_x,
                              std::vector<float> &coords_y,
                              std::vector<float> &coords_z, float tol,
                              float norm_of_qoi, float s); // ... 6a

float *mgard_decompress(int itype_flag, float &quantizer, unsigned char *data,
                        int data_len, int n1, int n2,
                        int n3); // decompress L-infty compressed data
float *mgard_decompress(
    int itype_flag, float &quantizer, unsigned char *data, int data_len, int n1,
    int n2, int n3, std::vector<float> &coords_x, std::vector<float> &coords_y,
    std::vector<float> &coords_z); // decompress L-infty compressed data

float *mgard_decompress(int itype_flag, float &quantizer, unsigned char *data,
                        int data_len, int n1, int n2, int n3,
                        float s); // decompress s-norm
float *mgard_decompress(int itype_flag, float &quantizer, unsigned char *data,
                        int data_len, int n1, int n2, int n3,
                        std::vector<float> &coords_x,
                        std::vector<float> &coords_y,
                        std::vector<float> &coords_z,
                        float s); // decompress s-norm

/// END FLOAT///

#endif
