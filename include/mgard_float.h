// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.

#ifndef MGARD_FLOAT_H
#define MGARD_FLOAT_H

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <zlib.h>

#include <zlib.h>

#include "mgard_nuni_float.h"

namespace mgard

{

inline int get_index(const int ncol, const int i, const int j);

inline float interp_2d(float q11, float q12, float q21, float q22, float x1,
                       float x2, float y1, float y2, float x, float y);

inline float interp_0d(const float x1, const float x2, const float y1,
                       const float y2, const float x);
void mass_matrix_multiply(const int l, std::vector<float> &v);

void solve_tridiag_M(const int l, std::vector<float> &v);

void restriction(const int l, std::vector<float> &v);

void interpolate_from_level_nMl(const int l, std::vector<float> &v);

void print_level_2D(const int nrow, const int ncol, const int l, float *v);

void write_level_2D(const int nrow, const int ncol, const int l, float *v,
                    std::ofstream &outfile);

void write_level_2D_exc(const int nrow, const int ncol, const int l, float *v,
                        std::ofstream &outfile);

void pi_lminus1(const int l, std::vector<float> &v0);

void pi_Ql(const int nrow, const int ncol, const int l, float *v,
           std::vector<float> &row_vec, std::vector<float> &col_vec);

void assign_num_level(const int nrow, const int ncol, const int l, float *v,
                      float num);

void copy_level(const int nrow, const int ncol, const int l, float *v,
                std::vector<float> &work);

void add_level(const int nrow, const int ncol, const int l, float *v,
               float *work);

void subtract_level(const int nrow, const int ncol, const int l, float *v,
                    float *work);

void compute_correction_loadv(const int l, std::vector<float> &v);

void qwrite_level_2D(const int nrow, const int ncol, const int nlevel,
                     const int l, float *v, float tol,
                     const std::string outfile);

void quantize_2D_interleave(const int nrow, const int ncol, float *v,
                            std::vector<int> &work, float norm, float tol);

void dequantize_2D_interleave(const int nrow, const int ncol, float *v,
                              const std::vector<int> &work);

void zwrite_2D_interleave(std::vector<int> &qv, const std::string outfile);

void compress_memory_z(void *in_data, size_t in_data_size,
                       std::vector<uint8_t> &out_data);

void qread_level_2D(const int nrow, const int ncol, const int nlevel, float *v,
                    std::string infile);

void set_number_of_levels(const int nrow, const int ncol, int &nlevel);

void resample_1d(const float *inbuf, float *outbuf, const int ncol,
                 const int ncol_new);

void resample_2d(const float *inbuf, float *outbuf, const int nrow,
                 const int ncol, const int nrow_new, const int ncol_new);

void resample_2d_inv2(const float *inbuf, float *outbuf, const int nrow,
                      const int ncol, const int nrow_new, const int ncol_new);

unsigned char *refactor_qz(int nrow, int ncol, int nfib, const float *v,
                           int &outsize, float tol);

unsigned char *refactor_qz(int nrow, int ncol, int nfib, const float *v,
                           int &outsize, float tol, float s);

unsigned char *refactor_qz(int nrow, int ncol, int nfib,
                           std::vector<float> &coords_x,
                           std::vector<float> &coords_y,
                           std::vector<float> &coords_z, const float *v,
                           int &outsize, float tol);

unsigned char *refactor_qz(int nrow, int ncol, int nfib,
                           std::vector<float> &coords_x,
                           std::vector<float> &coords_y,
                           std::vector<float> &coords_z, const float *v,
                           int &outsize, float tol, float s);

unsigned char *refactor_qz_2D(int nrow, int ncol, const float *v, int &outsize,
                              float tol);

unsigned char *refactor_qz_2D(int nrow, int ncol, const float *v, int &outsize,
                              float tol, float s);

unsigned char *refactor_qz_2D(int nrow, int ncol, std::vector<float> &coords_x,
                              std::vector<float> &coords_y, const float *v,
                              int &outsize, float tol);

unsigned char *refactor_qz_2D(int nrow, int ncol, std::vector<float> &coords_x,
                              std::vector<float> &coords_y, const float *v,
                              int &outsize, float tol, float s);

float *recompose_udq(float dummyf, int nrow, int ncol, int nfib,
                     unsigned char *data, int data_len);

float *recompose_udq(int nrow, int ncol, int nfib, std::vector<float> &coords_x,
                     std::vector<float> &coords_y, std::vector<float> &coords_z,
                     unsigned char *data, int data_len);

float *recompose_udq(int nrow, int ncol, int nfib, unsigned char *data,
                     int data_len, float s);

float *recompose_udq(int nrow, int ncol, int nfib, std::vector<float> &coords_x,
                     std::vector<float> &coords_y, std::vector<float> &coords_z,
                     unsigned char *data, int data_len, float s);

float *recompose_udq_2D(float dummyf, int nrow, int ncol, unsigned char *data,
                        int data_len);

float *recompose_udq_2D(int nrow, int ncol, unsigned char *data, int data_len,
                        float s);

float *recompose_udq_2D(int nrow, int ncol, std::vector<float> &coords_x,
                        std::vector<float> &coords_y, unsigned char *data,
                        int data_len);

float *recompose_udq_2D(int nrow, int ncol, std::vector<float> &coords_x,
                        std::vector<float> &coords_y, unsigned char *data,
                        int data_len, float s);

// unsigned char *
//   refactor_qz_1D (int nrow,  const float *v, int &outsize, float tol);

// float*
//   recompose_udq_1D(int nrow,  unsigned char *data, int data_len);

int parse_cmdl(int argc, char **argv, int &nrow, int &ncol, float &tol,
               std::string &in_file);

bool is_2kplus1(float num);

void refactor(const int nrow, const int ncol, const int l_target, float *v,
              std::vector<float> &work, std::vector<float> &row_vec,
              std::vector<float> &col_vec);

void recompose(const int nrow, const int ncol, const int l_target, float *v,
               std::vector<float> &work, std::vector<float> &row_vec,
               std::vector<float> &col_vec);

void decompress_memory_z(const void *src, int srcLen, int *dst, int dstLen);

} // namespace mgard

#endif
