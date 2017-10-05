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


#ifndef MGARD_H
#define MGARD_H

#include<cmath>
#include<vector>
#include<iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <sys/stat.h>
#include <assert.h>
#include <zlib.h>
#include <cstring>

namespace mgard

{
  
  
  inline int
    get_index (const int ncol, const int i, const int j);


  inline double
    interp_2d (double q11, double q12, double q21, double q22, double x1,
               double x2, double y1, double y2, double x, double y);

  inline double
    interp_0d (const double x1, const double x2, const double y1, const double y2,
               const double x);
  void
    mass_matrix_multiply (const int l, std::vector<double> &v);

  void
    solve_tridiag_M (const int l, std::vector<double> &v);

  void
    restrict (const int l, std::vector<double> &v);

  void
    interpolate_from_level_nMl (const int l, std::vector<double> &v);

  void
    print_level_2D (const int nrow, const int ncol, const int l, double *v);

  void
    write_level_2D (const int nrow, const int ncol, const int l, double *v,
                    std::ofstream &outfile);

  void
    write_level_2D_exc (const int nrow, const int ncol, const int l, double *v,
                        std::ofstream &outfile);

  void
    pi_lminus1 (const int l, std::vector<double> &v0);

  void
    pi_Ql (const int nrow, const int ncol, const int l, double *v,
           std::vector<double> &row_vec, std::vector<double> &col_vec);

  void
    assign_num_level (const int nrow, const int ncol, const int l, double *v,
                      double num);

  void
    copy_level (const int nrow, const int ncol, const int l, double *v,
                std::vector<double> &work);

  void
    add_level (const int nrow, const int ncol, const int l, double *v,
               double *work);

  void
    subtract_level (const int nrow, const int ncol, const int l, double *v,
                    double *work);

  void
    compute_correction_loadv (const int l, std::vector<double> &v);

  void
    qwrite_level_2D (const int nrow, const int ncol, const int nlevel, const int l,
                     double *v, double tol, const std::string outfile);

  void
    quantize_2D_iterleave (const int nrow, const int ncol, double *v,
                           std::vector<int> &work, double tol);

  void
    dequantize_2D_iterleave (const int nrow, const int ncol, double *v,
                             const std::vector<int> &work);

  void
    zwrite_2D_interleave (std::vector<int> &qv, const std::string outfile);


  void
    compress_memory (void *in_data, size_t in_data_size,
                     std::vector<uint8_t> &out_data);

  void
    decompress_memory (const void *src, int srcLen, void *dst, int dstLen);

  void
    qread_level_2D (const int nrow, const int ncol, const int nlevel, double *v,
                    std::string infile);

  void
    set_number_of_levels (const int nrow, const int ncol, int &nlevel);



  void
    resample_1d (const double *inbuf, double *outbuf, const int ncol,
                 const int ncol_new);

  void
    resample_2d (const double *inbuf, double*  outbuf, const int nrow,
                 const int ncol, const int nrow_new, const int ncol_new);

  void
    resample_2d_inv2 (const double *inbuf, double*  outbuf, const int nrow,
                 const int ncol, const int nrow_new, const int ncol_new);

  unsigned char *
    refactor_qz (int nrow, int ncol, double *v, int &outsize, double tol);

  double*
    recompose_udq(int nrow, int ncol, unsigned char *data, int data_len);
  
  int
    parse_cmdl (int argc, char **argv, int &nrow, int &ncol, double &tol,
                std::string &in_file);
  
  bool
    is_2kplus1 (double num);
  
  void
    refactor (const int nrow, const int ncol, const int l_target, double *v,
              std::vector<double> &work, std::vector<double> &row_vec,
              std::vector<double> &col_vec);

  void
    recompose (const int nrow, const int ncol, const int l_target, double *v,
               std::vector<double> &work, std::vector<double> &row_vec,
               std::vector<double> &col_vec);
}
  
#endif
