// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.
#ifndef MGARD_API_TPP
#define MGARD_API_TPP

#include <cassert>
#include <cstddef>

#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#include "TensorNorms.hpp"
#include "mgard.hpp"
#include "mgard_nuni.h"

// This should eventually be folded into `TensorMeshHierarchy`.
static std::vector<int> dataset_dimensions(const std::array<int, 3> input) {
  std::vector<int> trimmed;
  for (const int d : input) {
    if (d <= 0) {
      throw std::invalid_argument("all dimensions must be positive");
    } else if (d == 1) {
      // Skip.
    } else if (d < 3) {
      throw std::invalid_argument("no dimension can be 2 or 3");
    } else {
      trimmed.push_back(d);
    }
  }
  if (trimmed.empty()) {
    throw std::invalid_argument(
        "at least one dimension must be greater than 1");
  }
  return trimmed;
}

template <typename Real>
unsigned char *mgard_compress(Real *v, int &out_size, int nrow, int ncol,
                              int nfib, Real tol_in)

// Perform compression preserving the tolerance in the L-infty norm
{

  Real tol = tol_in;
  assert(tol >= 1e-7);
  unsigned char *p = nullptr;
  const std::vector<int> dims = dataset_dimensions({nrow, ncol, nfib});
  switch (dims.size()) {
  case 3:
    p = mgard::refactor_qz(dims.at(0), dims.at(1), dims.at(2), v, out_size,
                           tol);
    break;
  case 2:
    p = mgard::refactor_qz_2D(dims.at(0), dims.at(1), v, out_size, tol);
    break;
  case 1:
    p = mgard::refactor_qz_1D(dims.at(0), v, out_size, tol);
    break;
  default:
    throw std::logic_error("dataset dimension must be 1, 2, or 3");
  }
  return p;
}

template <typename Real>
unsigned char *mgard_compress(Real *v, int &out_size, int nrow, int ncol,
                              int nfib, std::vector<Real> &coords_x,
                              std::vector<Real> &coords_y,
                              std::vector<Real> &coords_z, Real tol)
// Perform compression preserving the tolerance in the L-infty norm, arbitrary
// tensor grids
{
  assert(tol >= 1e-7);
  unsigned char *p = nullptr;
  const std::vector<int> dims = dataset_dimensions({nrow, ncol, nfib});
  switch (dims.size()) {
  case 3:
    p = mgard::refactor_qz(dims.at(0), dims.at(1), dims.at(2), coords_x,
                           coords_y, coords_z, v, out_size, tol);
    break;
  case 2:
    // This is to spare us the trouble of somehow incorporating the coordinate
    // vectors into `dims` (note that the `refactor_qz_2D` arguments are
    // hardcoded to `coords_x` and `coords_y`). Meant to be a stopgap measure
    // before `TensorMeshHierarchy` is adopted here.
    assert(nfib == 1);
    p = mgard::refactor_qz_2D(dims.at(0), dims.at(1), coords_x, coords_y, v,
                              out_size, tol);
    break;
  default:
    // Waiting on the code being templated by dimension.
    throw std::logic_error("dataset dimension must be 2 or 3");
  }
  return p;
}

template <typename Real>
unsigned char *mgard_compress(Real *v, int &out_size, int nrow, int ncol,
                              int nfib, Real tol_in, Real s) {
  // Perform compression preserving the tolerance in s norm by defaulting to the
  // s-norm
  Real tol = tol_in;
  assert(tol >= 1e-7);

  unsigned char *p = nullptr;
  const std::vector<int> dims = dataset_dimensions({nrow, ncol, nfib});
  switch (dims.size()) {
  case 3:
    p = mgard::refactor_qz(dims.at(0), dims.at(1), dims.at(2), v, out_size, tol,
                           s);
    break;
  case 2:
    p = mgard::refactor_qz_2D(dims.at(0), dims.at(1), v, out_size, tol, s);
    break;
  default:
    // Waiting on the code being templated by dimension.
    throw std::logic_error("dataset dimension must be 2 or 3");
  }
  return p;
}

template <typename Real>
unsigned char *mgard_compress(Real *v, int &out_size, int nrow, int ncol,
                              int nfib, Real tol_in,
                              Real (*qoi)(int, int, int, Real *), Real s) {
  // Perform compression preserving the tolerance in s norm by defaulting to the
  // L-2 norm
  Real tol = tol_in;
  assert(tol >= 1e-7);

  const Real xi_norm = mgard::norm(nrow, ncol, nfib, qoi, s);

  unsigned char *p = nullptr;
  const std::vector<int> dims = dataset_dimensions({nrow, ncol, nfib});
  switch (dims.size()) {
  case 3:
    p = mgard::refactor_qz(dims.at(0), dims.at(1), dims.at(2), v, out_size,
                           xi_norm * tol, -s);
    break;
  case 2:
    p = mgard::refactor_qz_2D(dims.at(0), dims.at(1), v, out_size,
                              xi_norm * tol, -s);
    break;
  default:
    // Waiting on the code being templated by dimension.
    throw std::logic_error("dataset dimension must be 2 or 3");
  }
  return p;
}

template <typename Real>
Real *mgard_decompress(unsigned char *data, int data_len, int nrow, int ncol,
                       int nfib) {
  Real *p = nullptr;
  const std::vector<int> dims = dataset_dimensions({nrow, ncol, nfib});
  switch (dims.size()) {
  case 3:
    p = mgard::recompose_udq<Real>(dims.at(0), dims.at(1), dims.at(2), data,
                                   data_len);
    break;
  case 2:
    p = mgard::recompose_udq_2D<Real>(dims.at(0), dims.at(1), data, data_len);
    break;
  case 1:
    p = mgard::recompose_udq_1D_huffman<Real>(dims.at(0), data, data_len);
    break;
  default:
    throw std::logic_error("dataset dimension must be 1, 2, or 3");
  }
  return p;
}

template <typename Real>
Real *mgard_decompress(unsigned char *data, int data_len, int nrow, int ncol,
                       int nfib, Real s) {
  Real *p = nullptr;
  const std::vector<int> dims = dataset_dimensions({nrow, ncol, nfib});
  switch (dims.size()) {
  case 3:
    p = mgard::recompose_udq(dims.at(0), dims.at(1), dims.at(2), data, data_len,
                             s);
    break;
  case 2:
    p = mgard::recompose_udq_2D(dims.at(0), dims.at(1), data, data_len, s);
    break;
  default:
    // Waiting on the code being templated by dimension.
    throw std::logic_error("dataset dimension must be 2 or 3");
  }
  return p;
}

template <typename Real>
unsigned char *mgard_compress(Real *v, int &out_size, int nrow, int ncol,
                              int nfib, Real tol_in, Real norm_of_qoi, Real s) {
  return mgard_compress(v, out_size, nrow, ncol, nfib, norm_of_qoi * tol_in, s);
}

#endif
