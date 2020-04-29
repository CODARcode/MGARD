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

//! Multiply a nodal value vector by the piecewise linear mass matrix.
//!
//! The mesh is assumed to be uniform, with the *finest* level having cells of
//! width `6`. The input vector is overwritten with the product.
//!
//!\param[in] l Difference between the index of the finest mesh level and the
//! index of this mesh level. That is, `0` corresponds to the finest level, `1`
//! to the second finest, and so on.
//!\param[in, out] Nodal value vector.
template <typename Real>
void mass_matrix_multiply(const int l, std::vector<Real> &v);

//! Apply the inverse the piecewise mass linear mass matrix.
//!
//! The mesh is assumed to be uniform, with the *finest* level having cells of
//! width `6`. The input vector is overwritten with the product.
//!
//!\param[in] l Difference between the index of the finest mesh level and the
//! index of this mesh level, as in `mass_matrix_multiply`.
//!\param[in, out] Mass matrix–nodal value vector product.
template <typename Real>
void solve_tridiag_M(const int l, std::vector<Real> &v);

//! Restrict a mass matrix–nodal value vector product from a fine mesh to the
//! immediately coarser mesh.
//!
//! The mesh is assumed to be uniform. The input entries corresponding to nodes
//! on the immediately coarser level will be overwritten.
//!
//!\param[in] l Difference between the index of the finest mesh level and the
//! index of the coarse mesh level, as in `mass_matrix_multiply`.
//!\param[in, out] Mass matrix–nodal value vector product on the fine mesh.
template <typename Real> void restriction(const int l, std::vector<Real> &v);

//! Interpolate a function from one level to the level immediately finer.
//!
//! The mesh is assumed to be uniform. The input entries corresponding to nodes
//! on the immediately finer level will be overwritten.
//!
//!\param[in] l Difference between the index of the finest mesh level and the
//! index of the coarser mesh level, as in `mass_matrix_multiply`.
//!\param[in, out] v Nodal values to be interpolated.
template <typename Real>
void interpolate_from_level_nMl(const int l, std::vector<Real> &v);

//! Interpolate a function defined on 'old' nodes and subtract from values on
//! 'new' nodes.
//!
//! The mesh is assumed to be uniform. The input entries corresponding to nodes
//! on the finer level will be overwritten.
//!
//!\param[in] l Difference between the index of the finest mesh level and the
//! index of the finer mesh level, as in `mass_matrix_multiply`.
//!\param[in, out] v Nodal values to be interpolated.
template <typename Real> void pi_lminus1(const int l, std::vector<Real> &v);

//! Interpolate a function defined on 'old' nodes and subtract from values on
//! 'new' nodes.
//!
//! The mesh is assumed to be uniform. The input entries corresponding to nodes
//! on the finer level will be overwritten.
//!
//!\param[in] nrow Number of rows in the dataset.
//!\param[in] ncol Number of columns in the dataset.
//!\param[in] l Difference between the index of the finest mesh level and the
//! index of the finer mesh level, as in `mass_matrix_multiply`.
//!\param[in, out] v Nodal values to be interpolated.
//!\param[in] row_vec Work buffer of size `ncol`.
//!\param[in] col_vec Work buffer of size `nrow`.
template <typename Real>
void pi_Ql(const int nrow, const int ncol, const int l, Real *const v,
           std::vector<Real> &row_vec, std::vector<Real> &col_vec);

//! Set the entries corresponding to nodes on some level to some number.
//!
//! The entries corresponding to nodes on the given level will be overwritten.
//!
//!\param[in] nrow Number of rows in the dataset.
//!\param[in] ncol Number of columns in the dataset.
//!\param[in] l Difference between the index of the finest mesh level and the
//! index of this mesh level, as in `mass_matrix_multiply`.
//!\param[out] v Nodal values (a subset of which) to be overwritten.
//!\param[in] num Value to write.
template <typename Real>
void assign_num_level(const int nrow, const int ncol, const int l,
                      Real *const v, const Real num);

//! Copy the entries corresponding to nodes on some level.
//!
//! The entries corresponding to nodes on the given level will be overwritten.
//!
//!\param[in] nrow Number of rows in the dataset.
//!\param[in] ncol Number of columns in the dataset.
//!\param[in] l Difference between the index of the finest mesh level and the
//! index of this mesh level, as in `mass_matrix_multiply`.
//!\param[in] v Nodal values (a subset of which) to copy from.
//!\param[out] work Nodal values (a subset of which) to copy to.
template <typename Real>
void copy_level(const int nrow, const int ncol, const int l,
                Real const *const v, Real *const work);

//! Add one function to another one on the nodes of a mesh.
//!
//! The entries corresponding to nodes on the given level will be overwritten.
//!
//!\param[in] nrow Number of rows in the dataset.
//!\param[in] ncol Number of columns in the dataset.
//!\param[in] l Difference between the index of the finest mesh level and the
//! index of this mesh level, as in `mass_matrix_multiply`.
//!\param[in, out] v Nodal values (a subset of which) to add to.
//!\param[out] work Nodal values (a subset of which) to add.
template <typename Real>
void add_level(const int nrow, const int ncol, const int l, Real *const v,
               Real const *const work);

//! Subtract one function from another one on the nodes of a mesh.
//!
//! The entries corresponding to nodes on the given level will be overwritten.
//!
//!\param[in] nrow Number of rows in the dataset.
//!\param[in] ncol Number of columns in the dataset.
//!\param[in] l Difference between the index of the finest mesh level and the
//! index of this mesh level, as in `mass_matrix_multiply`.
//!\param[in, out] v Nodal values (a subset of which) to subtract from.
//!\param[out] work Nodal values (a subset of which) to subtract.
template <typename Real>
void subtract_level(const int nrow, const int ncol, const int l, Real *const v,
                    Real const *const work);

//! Quantize an array of multilevel coefficients.
//!
//! The quantized coefficients will be preceded by the quantization quantum in
//! the output array. Currently `tol` is a rescaling of the original error
//! tolerance, so that `tol * norm` is the correct quantum. This will likely be
//! changed in the future.
//!
//!\param[in] nrow Number of rows in the dataset.
//!\param[in] ncol Number of columns in the dataset.
//!\param[in] v Array of multilevel coefficients to be quantized.
//!\param[out] work Array in which to store quantized coefficients.
//!\param[in] norm Norm of the input function.
//!\param[in] tol Error tolerance, appropriately scaled.
template <typename Real>
void quantize_2D_interleave(const int nrow, const int ncol, Real const *const v,
                            std::vector<int> &work, const Real norm,
                            const Real tol);

//! Dequantize an array of quantized multilevel coefficients.
//!
//! The quantization quantum is read from the beginning of the input array.
//!
//!\param[in] nrow Number of rows in the dataset.
//!\param[in] ncol Number of columns in the dataset.
//!\param[out] v Array in which to store dequantized coefficients.
//!\param[in] work Array of quantized multilevel coefficients to be dequantized.
template <typename Real>
void dequantize_2D_interleave(const int nrow, const int ncol, Real *const v,
                              const std::vector<int> &work);

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

template <typename Real>
Real *recompose_udq_1D_huffman(int ncol, unsigned char *data, int data_len);

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

#endif
