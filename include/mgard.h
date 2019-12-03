// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.

#ifndef MGARD_H
#define MGARD_H

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

#include "mgard_nuni.h"

namespace mgard

{

inline int get_index(const int ncol, const int i, const int j);

inline double interp_2d(double q11, double q12, double q21, double q22,
                        double x1, double x2, double y1, double y2, double x,
                        double y);

inline double interp_0d(const double x1, const double x2, const double y1,
                        const double y2, const double x);
void mass_matrix_multiply(const int l, std::vector<double> &v);

void solve_tridiag_M(const int l, std::vector<double> &v);

void restriction(const int l, std::vector<double> &v);

void interpolate_from_level_nMl(const int l, std::vector<double> &v);

void print_level_2D(const int nrow, const int ncol, const int l, double *v);

void write_level_2D(const int nrow, const int ncol, const int l, double *v,
                    std::ofstream &outfile);

void write_level_2D_exc(const int nrow, const int ncol, const int l, double *v,
                        std::ofstream &outfile);

void pi_lminus1(const int l, std::vector<double> &v0);

void pi_Ql(const int nrow, const int ncol, const int l, double *v,
           std::vector<double> &row_vec, std::vector<double> &col_vec);

void assign_num_level(const int nrow, const int ncol, const int l, double *v,
                      double num);

void copy_level(const int nrow, const int ncol, const int l, double *v,
                std::vector<double> &work);

void add_level(const int nrow, const int ncol, const int l, double *v,
               double *work);

void subtract_level(const int nrow, const int ncol, const int l, double *v,
                    double *work);

void compute_correction_loadv(const int l, std::vector<double> &v);

void qwrite_level_2D(const int nrow, const int ncol, const int nlevel,
                     const int l, double *v, double tol,
                     const std::string outfile);

void quantize_2D_interleave(const int nrow, const int ncol, double *v,
                            std::vector<int> &work, double norm, double tol);

void dequantize_2D_interleave(const int nrow, const int ncol, double *v,
                              const std::vector<int> &work);

void zwrite_2D_interleave(std::vector<int> &qv, const std::string outfile);

void compress_memory_z(void *in_data, size_t in_data_size,
                       std::vector<uint8_t> &out_data);

void qread_level_2D(const int nrow, const int ncol, const int nlevel, double *v,
                    std::string infile);

void set_number_of_levels(const int nrow, const int ncol, int &nlevel);

void resample_1d(const double *inbuf, double *outbuf, const int ncol,
                 const int ncol_new);

void resample_2d(const double *inbuf, double *outbuf, const int nrow,
                 const int ncol, const int nrow_new, const int ncol_new);

void resample_2d_inv2(const double *inbuf, double *outbuf, const int nrow,
                      const int ncol, const int nrow_new, const int ncol_new);

unsigned char *refactor_qz(int nrow, int ncol, int nfib, const double *v,
                           int &outsize, double tol);

unsigned char *refactor_qz(int nrow, int ncol, int nfib, const double *v,
                           int &outsize, double tol, double s);

unsigned char *refactor_qz(int nrow, int ncol, int nfib,
                           std::vector<double> &coords_x,
                           std::vector<double> &coords_y,
                           std::vector<double> &coords_z, const double *v,
                           int &outsize, double tol);

unsigned char *refactor_qz(int nrow, int ncol, int nfib,
                           std::vector<double> &coords_x,
                           std::vector<double> &coords_y,
                           std::vector<double> &coords_z, const double *v,
                           int &outsize, double tol, double s);

unsigned char *refactor_qz_2D(int nrow, int ncol, const double *v, int &outsize,
                              double tol);

unsigned char *refactor_qz_2D(int nrow, int ncol, const double *v, int &outsize,
                              double tol, double s);

unsigned char *refactor_qz_2D(int nrow, int ncol, std::vector<double> &coords_x,
                              std::vector<double> &coords_y, const double *v,
                              int &outsize, double tol);

unsigned char *refactor_qz_2D(int nrow, int ncol, std::vector<double> &coords_x,
                              std::vector<double> &coords_y, const double *v,
                              int &outsize, double tol, double s);

double *recompose_udq(double dummyd, int nrow, int ncol, int nfib,
                      unsigned char *data, int data_len);

double *recompose_udq(int nrow, int ncol, int nfib,
                      std::vector<double> &coords_x,
                      std::vector<double> &coords_y,
                      std::vector<double> &coords_z, unsigned char *data,
                      int data_len);

double *recompose_udq(int nrow, int ncol, int nfib, unsigned char *data,
                      int data_len, double s);

double *recompose_udq(int nrow, int ncol, int nfib,
                      std::vector<double> &coords_x,
                      std::vector<double> &coords_y,
                      std::vector<double> &coords_z, unsigned char *data,
                      int data_len, double s);

double *recompose_udq_2D(double dummyd, int nrow, int ncol, unsigned char *data,
                         int data_len);

double *recompose_udq_2D(int nrow, int ncol, unsigned char *data, int data_len,
                         double s);

double *recompose_udq_2D(int nrow, int ncol, std::vector<double> &coords_x,
                         std::vector<double> &coords_y, unsigned char *data,
                         int data_len);

double *recompose_udq_2D(int nrow, int ncol, std::vector<double> &coords_x,
                         std::vector<double> &coords_y, unsigned char *data,
                         int data_len, double s);

unsigned char *refactor_qz_1D(int nrow, const double *v, int &outsize,
                              double tol);

double *recompose_udq_1D(int nrow, unsigned char *data, int data_len);

int parse_cmdl(int argc, char **argv, int &nrow, int &ncol, double &tol,
               std::string &in_file);

bool is_2kplus1(double num);

void refactor(const int nrow, const int ncol, const int l_target, double *v,
              std::vector<double> &work, std::vector<double> &row_vec,
              std::vector<double> &col_vec);

void recompose(const int nrow, const int ncol, const int l_target, double *v,
               std::vector<double> &work, std::vector<double> &row_vec,
               std::vector<double> &col_vec);

void decompress_memory_z(const void *src, int srcLen, int *dst, int dstLen);

} // namespace mgard

#endif
