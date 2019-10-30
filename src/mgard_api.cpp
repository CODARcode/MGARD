// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.

#include "mgard_api.h"
#include "mgard.h"

unsigned char *mgard_compress(int itype_flag, double *v, int &out_size,
                              int nrow, int ncol, int nfib, double tol_in)

// Perform compression preserving the tolerance in the L-infty norm
{

  double tol = tol_in;
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
  } else if (nrow > 1) {
    assert(nrow > 3);
    std::cerr << "MGARD: Not impemented!  Let us know if you need 1D "
                 "compression...\n";
    return nullptr;
  }

  return nullptr;
}

unsigned char *mgard_compress(int itype_flag, double *v, int &out_size,
                              int nrow, int ncol, int nfib,
                              std::vector<double> &coords_x,
                              std::vector<double> &coords_y,
                              std::vector<double> &coords_z, double tol)
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
    std::cerr << "MGARD: Not impemented!  Let us know if you need 1D "
                 "compression...\n";
    return nullptr;
  }
  return nullptr;
}

unsigned char *mgard_compress(int itype_flag, double *v, int &out_size,
                              int nrow, int ncol, int nfib, double tol_in,
                              double s) {
  // Perform compression preserving the tolerance in s norm by defaulting to the
  // s-norm
  double tol = tol_in;
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

// unsigned char *mgard_compress(int itype_flag,  double  *v, int &out_size, int
// nrow, int ncol, int nfib, double tol_in, double (*qoi) (int, int, int,
// std::vector<double>), double s)

unsigned char *mgard_compress(int itype_flag, double *v, int &out_size,
                              int nrow, int ncol, int nfib, double tol_in,
                              double (*qoi)(int, int, int, double *),
                              double s) {
  // Perform compression preserving the tolerance in s norm by defaulting to the
  // L-2 norm
  double tol = tol_in;
  assert(tol >= 1e-7);
  unsigned char *mgard_compressed_ptr = nullptr;
  if (nrow > 1 && ncol > 1 && nfib > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    assert(nfib > 3);

    std::vector<double> coords_x(ncol), coords_y(nrow),
        coords_z(nfib); // coordinate arrays
    // dummy equispaced coordinates
    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);
    std::iota(std::begin(coords_z), std::end(coords_z), 0);

    double xi_norm = mgard_gen::qoi_norm(nrow, ncol, nfib, coords_x, coords_y,
                                         coords_z, qoi, s);
    tol *= xi_norm;
    mgard_compressed_ptr =
        mgard::refactor_qz(nrow, ncol, nfib, v, out_size, tol, -s);
    return mgard_compressed_ptr;

  } else if (nrow > 1 && ncol > 1) {
    assert(nrow > 3);
    assert(ncol > 3);

    std::vector<double> coords_x(ncol), coords_y(nrow),
        coords_z(nfib); // coordinate arrays
    // dummy equispaced coordinates
    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);
    std::iota(std::begin(coords_z), std::end(coords_z), 0);

    double xi_norm = mgard_gen::qoi_norm(nrow, ncol, nfib, coords_x, coords_y,
                                         coords_z, qoi, s);
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

double *mgard_decompress(int itype_flag, double &quantizer, unsigned char *data,
                         int data_len, int nrow, int ncol, int nfib) {
  double *mgard_decompressed_ptr = nullptr;

  if (nrow > 1 && ncol > 1 && nfib > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    assert(nfib > 3);

    mgard_decompressed_ptr =
        mgard::recompose_udq(nrow, ncol, nfib, data, data_len);
    return mgard_decompressed_ptr;
  } else if (nrow > 1 && ncol > 1) {
    assert(nrow > 3);
    assert(ncol > 3);
    mgard_decompressed_ptr =
        mgard::recompose_udq_2D(nrow, ncol, data, data_len);
    //          mgard_decompressed_ptr = mgard::recompose_udq_2D(nrow, ncol,
    //          data, data_len);
    return mgard_decompressed_ptr;
  } else if (nrow > 1) {
    assert(nrow > 3);
    std::cerr << "MGARD: Not impemented!  Let us know if you need 1D "
                 "compression...\n";
    // mgard_decompressed_ptr = mgard::recompose_udq_1D(nrow,  data, data_len);
  }
  return nullptr;
}

double *mgard_decompress(int itype_flag, double &quantizer, unsigned char *data,
                         int data_len, int nrow, int ncol, int nfib, double s) {

  double *mgard_decompressed_ptr = nullptr;

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

double mgard_compress(int nrow, int ncol, int nfib,
                      double (*qoi)(int, int, int, std::vector<double>),
                      double s) {
  std::vector<double> coords_x(ncol), coords_y(nrow),
      coords_z(nfib); // coordinate arrays
  // dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  double xi_norm = mgard_gen::qoi_norm(nrow, ncol, nfib, coords_x, coords_y,
                                       coords_z, qoi, s);

  return xi_norm;
}

double mgard_compress(int nrow, int ncol, int nfib,
                      double (*qoi)(int, int, int, double *), double s) {
  std::vector<double> coords_x(ncol), coords_y(nrow),
      coords_z(nfib); // coordinate arrays
  // dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  double xi_norm = mgard_gen::qoi_norm(nrow, ncol, nfib, coords_x, coords_y,
                                       coords_z, qoi, s);

  return xi_norm;
}

unsigned char *mgard_compress(int itype_flag, double *v, int &out_size,
                              int nrow, int ncol, int nfib, double tol_in,
                              double norm_of_qoi, double s) {
  tol_in *= norm_of_qoi;
  return mgard_compress(itype_flag, v, out_size, nrow, ncol, nfib, tol_in, s);
}
