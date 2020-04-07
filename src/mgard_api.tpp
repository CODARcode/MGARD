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

#include <iostream>
#include <numeric>

#include "mgard.h"
#include "mgard_norms.hpp"
#include "mgard_nuni.h"

template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *v, int &out_size, int nrow,
                              int ncol, int nfib, Real tol_in)

// Perform compression preserving the tolerance in the L-infty norm
{

  Real tol = tol_in;
  assert(tol >= 1e-7);
  unsigned char *mgard_compressed_ptr = nullptr;
  if (nrow > 1 && ncol > 1 && nfib > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    assert(nfib > 3);

    mgard_compressed_ptr =
        mgard::refactor_qz(nrow, ncol, nfib, v, out_size, tol);
    return mgard_compressed_ptr;

  } else if (nrow > 1 && ncol > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    mgard_compressed_ptr = mgard::refactor_qz_2D(nrow, ncol, v, out_size, tol);
    return mgard_compressed_ptr;
  } else if (ncol > 1) {
    assert(ncol > 3);

    mgard_compressed_ptr = mgard::refactor_qz_1D(ncol, v, out_size, tol);

    return mgard_compressed_ptr;
  }

  return nullptr;
}

template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *v, int &out_size, int nrow,
                              int ncol, int nfib, std::vector<Real> &coords_x,
                              std::vector<Real> &coords_y,
                              std::vector<Real> &coords_z, Real tol)
// Perform compression preserving the tolerance in the L-infty norm, arbitrary
// tensor grids
{

  assert(tol >= 1e-7);
  unsigned char *mgard_compressed_ptr = nullptr;
  if (nrow > 1 && ncol > 1 && nfib > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    assert(nfib > 3);

    mgard_compressed_ptr = mgard::refactor_qz(
        nrow, ncol, nfib, coords_x, coords_y, coords_z, v, out_size, tol);
    return mgard_compressed_ptr;

  } else if (nrow > 1 && ncol > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    mgard_compressed_ptr =
        mgard::refactor_qz_2D(nrow, ncol, coords_x, coords_y, v, out_size, tol);
    return mgard_compressed_ptr;
  } else if (nrow > 1) {
    assert(nrow > 3);
    // To be cleaned up.
    //    mgard_compressed_ptr =
    //        mgard::refactor_qz_1D(ncol, coords_x, coords_y, v, out_size, tol);

    return mgard_compressed_ptr;
  }
  return nullptr;
}

template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *v, int &out_size, int nrow,
                              int ncol, int nfib, Real tol_in, Real s) {
  // Perform compression preserving the tolerance in s norm by defaulting to the
  // s-norm
  Real tol = tol_in;
  assert(tol >= 1e-7);

  unsigned char *mgard_compressed_ptr = nullptr;
  if (nrow > 1 && ncol > 1 && nfib > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    assert(nfib > 3);

    mgard_compressed_ptr =
        mgard::refactor_qz(nrow, ncol, nfib, v, out_size, tol, s);
    return mgard_compressed_ptr;

  } else if (nrow > 1 && ncol > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    mgard_compressed_ptr =
        mgard::refactor_qz_2D(nrow, ncol, v, out_size, tol, s);

    return mgard_compressed_ptr;
  } else if (nrow > 1) {
    assert(nrow > 3);
    std::cerr << "MGARD: Not impemented!  Let us know if you need 1D "
                 "compression...\n";
    return nullptr;
  }
  return nullptr;
}

// unsigned char *mgard_compress(int itype_flag,  Real  *v, int &out_size, int
// nrow, int ncol, int nfib, Real tol_in, Real (*qoi) (int, int, int,
// std::vector<Real>), Real s)

template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *v, int &out_size, int nrow,
                              int ncol, int nfib, Real tol_in,
                              Real (*qoi)(int, int, int, Real *), Real s) {
  // Perform compression preserving the tolerance in s norm by defaulting to the
  // L-2 norm
  Real tol = tol_in;
  assert(tol >= 1e-7);
  unsigned char *mgard_compressed_ptr = nullptr;
  if (nrow > 1 && ncol > 1 && nfib > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    assert(nfib > 3);

    std::vector<Real> coords_x(ncol), coords_y(nrow),
        coords_z(nfib); // coordinate arrays
    // dummy equispaced coordinates
    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);
    std::iota(std::begin(coords_z), std::end(coords_z), 0);

    Real xi_norm =
        mgard::qoi_norm(nrow, ncol, nfib, coords_x, coords_y, coords_z, qoi, s);
    tol *= xi_norm;
    mgard_compressed_ptr =
        mgard::refactor_qz(nrow, ncol, nfib, v, out_size, tol, -s);
    return mgard_compressed_ptr;

  } else if (nrow > 1 && ncol > 1) {
    assert(nrow > 3);
    assert(ncol > 3);

    std::vector<Real> coords_x(ncol), coords_y(nrow),
        coords_z(nfib); // coordinate arrays
    // dummy equispaced coordinates
    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);
    std::iota(std::begin(coords_z), std::end(coords_z), 0);

    Real xi_norm =
        mgard::qoi_norm(nrow, ncol, nfib, coords_x, coords_y, coords_z, qoi, s);
    tol *= xi_norm;

    mgard_compressed_ptr =
        mgard::refactor_qz_2D(nrow, ncol, v, out_size, tol, -s);
    return mgard_compressed_ptr;
  } else if (nrow > 1) {
    assert(nrow > 3);
    std::cerr << "MGARD: Not impemented!  Let us know if you need 1D "
                 "compression...\n";
    // mgard_compressed_ptr = mgard::refactor_qz_1D(nrow, v, out_size, *tol);
  }
  return nullptr;
}

template <typename Real>
Real *mgard_decompress(int itype_flag, unsigned char *data, int data_len,
                       int nrow, int ncol, int nfib) {
  Real *mgard_decompressed_ptr = nullptr;

  if (nrow > 1 && ncol > 1 && nfib > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    assert(nfib > 3);

    mgard_decompressed_ptr =
        mgard::recompose_udq<Real>(nrow, ncol, nfib, data, data_len);
    return mgard_decompressed_ptr;
  } else if (nrow > 1 && ncol > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    mgard_decompressed_ptr =
        mgard::recompose_udq_2D<Real>(nrow, ncol, data, data_len);
    //          mgard_decompressed_ptr = mgard::recompose_udq_2D(nrow, ncol,
    //          data, data_len);
    return mgard_decompressed_ptr;
  } else if (ncol > 1) {
    assert(ncol > 3);

    mgard_decompressed_ptr =
        mgard::recompose_udq_1D_huffman<Real>(ncol, data, data_len);
    return mgard_decompressed_ptr;
  }
  return nullptr;
}

template <typename Real>
Real *mgard_decompress(int itype_flag, unsigned char *data, int data_len,
                       int nrow, int ncol, int nfib, Real s) {

  Real *mgard_decompressed_ptr = nullptr;

  if (nrow > 1 && ncol > 1 && nfib > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    assert(nfib > 3);

    mgard_decompressed_ptr =
        mgard::recompose_udq(nrow, ncol, nfib, data, data_len, s);
    return mgard_decompressed_ptr;
  } else if (nrow > 1 && ncol > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    mgard_decompressed_ptr =
        mgard::recompose_udq_2D(nrow, ncol, data, data_len, s);
    return mgard_decompressed_ptr;
  } else if (nrow > 1) {
    assert(nrow > 3);
    std::cerr << "MGARD: Not impemented!  Let us know if you need 1D "
                 "compression...\n";
    // mgard_decompressed_ptr = mgard::recompose_udq_1D(nrow,  data, data_len);
  }
  return nullptr;
}

template <typename Real>
Real mgard_compress(int nrow, int ncol, int nfib,
                    Real (*qoi)(int, int, int, std::vector<Real>), Real s) {
  std::vector<Real> coords_x(ncol), coords_y(nrow),
      coords_z(nfib); // coordinate arrays
  // dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  Real xi_norm =
      mgard::qoi_norm(nrow, ncol, nfib, coords_x, coords_y, coords_z, qoi, s);

  return xi_norm;
}

template <typename Real>
Real mgard_compress(int nrow, int ncol, int nfib,
                    Real (*qoi)(int, int, int, Real *), Real s) {
  std::vector<Real> coords_x(ncol), coords_y(nrow),
      coords_z(nfib); // coordinate arrays
  // dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  Real xi_norm =
      mgard::qoi_norm(nrow, ncol, nfib, coords_x, coords_y, coords_z, qoi, s);

  return xi_norm;
}

template <typename Real>
unsigned char *mgard_compress(int itype_flag, Real *v, int &out_size, int nrow,
                              int ncol, int nfib, Real tol_in, Real norm_of_qoi,
                              Real s) {
  tol_in *= norm_of_qoi;
  return mgard_compress(itype_flag, v, out_size, nrow, ncol, nfib, tol_in, s);
}

#endif
