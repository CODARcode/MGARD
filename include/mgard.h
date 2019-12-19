// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.
#ifndef MGARD_H
#define MGARD_H

#include <string>
#include <vector>

namespace mgard {

template <typename Real>
inline Real interp_2d(Real q11, Real q12, Real q21, Real q22, Real x1, Real x2,
                      Real y1, Real y2, Real x, Real y);

template <typename Real>
inline Real interp_0d(const Real x1, const Real x2, const Real y1,
                      const Real y2, const Real x);

template <typename Real>
void mass_matrix_multiply(const int l, std::vector<Real> &v);

template <typename Real>
void solve_tridiag_M(const int l, std::vector<Real> &v);

template <typename Real> void restriction(const int l, std::vector<Real> &v);

template <typename Real>
void interpolate_from_level_nMl(const int l, std::vector<Real> &v);

template <typename Real>
void print_level_2D(const int nrow, const int ncol, const int l, Real *v);

template <typename Real>
void write_level_2D(const int nrow, const int ncol, const int l, Real *v,
                    std::ofstream &outfile);

template <typename Real>
void write_level_2D_exc(const int nrow, const int ncol, const int l, Real *v,
                        std::ofstream &outfile);

template <typename Real> void pi_lminus1(const int l, std::vector<Real> &v0);

template <typename Real>
void pi_Ql(const int nrow, const int ncol, const int l, Real *v,
           std::vector<Real> &row_vec, std::vector<Real> &col_vec);

template <typename Real>
void assign_num_level(const int nrow, const int ncol, const int l, Real *v,
                      Real num);

template <typename Real>
void copy_level(const int nrow, const int ncol, const int l, Real *v,
                std::vector<Real> &work);

template <typename Real>
void add_level(const int nrow, const int ncol, const int l, Real *v,
               Real *work);

template <typename Real>
void subtract_level(const int nrow, const int ncol, const int l, Real *v,
                    Real *work);

template <typename Real>
void compute_correction_loadv(const int l, std::vector<Real> &v);

template <typename Real>
void qwrite_level_2D(const int nrow, const int ncol, const int nlevel,
                     const int l, Real *v, Real tol, const std::string outfile);

template <typename Real>
void quantize_2D_interleave(const int nrow, const int ncol, Real *v,
                            std::vector<int> &work, Real norm, Real tol);

template <typename Real>
void dequantize_2D_interleave(const int nrow, const int ncol, Real *v,
                              const std::vector<int> &work);

template <typename Real>
void qread_level_2D(const int nrow, const int ncol, const int nlevel, Real *v,
                    std::string infile);

template <typename Real>
void resample_1d(const Real *inbuf, Real *outbuf, const int ncol,
                 const int ncol_new);

template <typename Real>
void resample_2d(const Real *inbuf, Real *outbuf, const int nrow,
                 const int ncol, const int nrow_new, const int ncol_new);

template <typename Real>
void resample_2d_inv2(const Real *inbuf, Real *outbuf, const int nrow,
                      const int ncol, const int nrow_new, const int ncol_new);

template <typename Real>
unsigned char *refactor_qz(int nrow, int ncol, int nfib, const Real *v,
                           int &outsize, Real tol);

template <typename Real>
unsigned char *refactor_qz(int nrow, int ncol, int nfib, const Real *v,
                           int &outsize, Real tol, Real s);

template <typename Real>
unsigned char *
refactor_qz(int nrow, int ncol, int nfib, std::vector<Real> &coords_x,
            std::vector<Real> &coords_y, std::vector<Real> &coords_z,
            const Real *v, int &outsize, Real tol);

template <typename Real>
unsigned char *
refactor_qz(int nrow, int ncol, int nfib, std::vector<Real> &coords_x,
            std::vector<Real> &coords_y, std::vector<Real> &coords_z,
            const Real *v, int &outsize, Real tol, Real s);

template <typename Real>
unsigned char *refactor_qz_1D(int n, const Real *v, int &outsize, Real tol);

template <typename Real>
unsigned char *refactor_qz_1D(int n, const Real *v, int &outsize, Real tol,
                              Real s);

template <typename Real>
unsigned char *refactor_qz_2D(int nrow, int ncol, const Real *v, int &outsize,
                              Real tol);

template <typename Real>
unsigned char *refactor_qz_2D(int nrow, int ncol, const Real *v, int &outsize,
                              Real tol, Real s);

template <typename Real>
unsigned char *refactor_qz_2D(int nrow, int ncol, std::vector<Real> &coords_x,
                              std::vector<Real> &coords_y, const Real *v,
                              int &outsize, Real tol);

template <typename Real>
unsigned char *refactor_qz_2D(int nrow, int ncol, std::vector<Real> &coords_x,
                              std::vector<Real> &coords_y, const Real *v,
                              int &outsize, Real tol, Real s);

template <typename Real>
Real *recompose_udq(int nrow, int ncol, int nfib, unsigned char *data,
                    int data_len);

template <typename Real>
Real *recompose_udq(int nrow, int ncol, int nfib, std::vector<Real> &coords_x,
                    std::vector<Real> &coords_y, std::vector<Real> &coords_z,
                    unsigned char *data, int data_len);

template <typename Real>
Real *recompose_udq(int nrow, int ncol, int nfib, unsigned char *data,
                    int data_len, Real s);

template <typename Real>
Real *recompose_udq(int nrow, int ncol, int nfib, std::vector<Real> &coords_x,
                    std::vector<Real> &coords_y, std::vector<Real> &coords_z,
                    unsigned char *data, int data_len, Real s);

template <typename Real>
Real *recompose_udq_2D(int nrow, int ncol, unsigned char *data, int data_len);

template <typename Real>
Real *recompose_udq_2D(int nrow, int ncol, unsigned char *data, int data_len,
                       Real s);

template <typename Real>
Real *recompose_udq_2D(int nrow, int ncol, std::vector<Real> &coords_x,
                       std::vector<Real> &coords_y, unsigned char *data,
                       int data_len);

template <typename Real>
Real *recompose_udq_2D(int nrow, int ncol, std::vector<Real> &coords_x,
                       std::vector<Real> &coords_y, unsigned char *data,
                       int data_len, Real s);

template <typename Real>
unsigned char *refactor_qz_1D(int ncol, const Real *v, int &outsize, Real tol);

template <typename Real>
Real *recompose_udq_1D(int ncol, unsigned char *data, int data_len);

// Gary new
template <typename Real>
void refactor_1D(const int ncol, const int l_target, Real *v,
                 std::vector<Real> &work, std::vector<Real> &row_vec);

template <typename Real>
void refactor(const int nrow, const int ncol, const int l_target, Real *v,
              std::vector<Real> &work, std::vector<Real> &row_vec,
              std::vector<Real> &col_vec);

template <typename Real>
void recompose_1D(const int ncol, const int l_target, Real *v,
                  std::vector<Real> &work, std::vector<Real> &row_vec);

template <typename Real>
void recompose(const int nrow, const int ncol, const int l_target, Real *v,
               std::vector<Real> &work, std::vector<Real> &row_vec,
               std::vector<Real> &col_vec);

} // namespace mgard

#include "mgard.tpp"

#endif
