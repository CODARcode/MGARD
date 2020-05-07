// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.
#ifndef MGARD_TPP
#define MGARD_TPP

#include "mgard.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>

#include <zlib.h>

#include <bitset>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "mgard_compress.hpp"
#include "mgard_mesh.hpp"
#include "mgard_nuni.h"

#include "LinearQuantizer.hpp"

namespace mgard {

template <typename Real>
unsigned char *refactor_qz(int nrow, int ncol, int nfib, const Real *u,
                           int &outsize, Real tol) {
  // Dummy equispaced coordinates.
  std::vector<Real> coords_x(ncol);
  std::vector<Real> coords_y(nrow);
  std::vector<Real> coords_z(nfib);
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);
  return refactor_qz(nrow, ncol, nfib, coords_x, coords_y, coords_z, u, outsize,
                     tol);
}

template <typename Real>
unsigned char *
refactor_qz(int nrow, int ncol, int nfib, std::vector<Real> &coords_x,
            std::vector<Real> &coords_y, std::vector<Real> &coords_z,
            const Real *u, int &outsize, Real tol) {
  const TensorMeshHierarchy<3, Real> hierarchy({nrow, ncol, nfib});
  std::vector<Real> v(u, u + nrow * ncol * nfib), work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array

  const Dimensions2kPlus1<3> dims({nrow, ncol, nfib});
  const int l_target = dims.nlevel - 1;

  Real norm = mgard_common::max_norm(v);

  // TODO: in the `float` implementation, we divide by `nlevel + 2`.
  tol /= dims.nlevel + 1;

  mgard_gen::prep_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2], dims.input[0],
                     dims.input[1], dims.input[2], l_target, v.data(), work,
                     work2d, coords_x, coords_y, coords_z);

  mgard_gen::refactor_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2],
                         dims.input[0], dims.input[1], dims.input[2], l_target,
                         v.data(), work, work2d, coords_x, coords_y, coords_z);

  work.clear();
  work2d.clear();

  const int size_ratio = sizeof(Real) / sizeof(int);
  std::vector<int> qv(nrow * ncol * nfib + size_ratio);

  // rename this to quantize Linfty or smthng!!!!
  mgard::quantize_interleave(hierarchy, v.data(), qv, norm, tol);

  std::vector<unsigned char> out_data;

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);
  return buffer;
}

template <typename Real>
unsigned char *refactor_qz(int nrow, int ncol, int nfib, const Real *u,
                           int &outsize, Real tol, Real s) {
  // Dummy equispaced coordinates.
  std::vector<Real> coords_x(ncol);
  std::vector<Real> coords_y(nrow);
  std::vector<Real> coords_z(nfib);
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);
  return refactor_qz(nrow, ncol, nfib, coords_x, coords_y, coords_z, u, outsize,
                     tol, s);
}

template <typename Real>
unsigned char *
refactor_qz(int nrow, int ncol, int nfib, std::vector<Real> &coords_x,
            std::vector<Real> &coords_y, std::vector<Real> &coords_z,
            const Real *u, int &outsize, Real tol, Real s) {
  std::vector<Real> v(u, u + nrow * ncol * nfib), work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array

  const Dimensions2kPlus1<3> dims({nrow, ncol, nfib});
  const int l_target = dims.nlevel - 1;

  Real norm = 1.0;

  if (std::abs(s) < 1e-10) {
    norm = mgard_gen::ml2_norm3(0, nrow, ncol, nfib, nrow, ncol, nfib, v,
                                coords_x, coords_y, coords_z);
    norm = std::sqrt(norm /
                     (nrow * nfib * ncol)); //<- quant scaling goes here for s
  }

  mgard_gen::prep_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2], dims.input[0],
                     dims.input[1], dims.input[2], l_target, v.data(), work,
                     work2d, coords_x, coords_y, coords_z);

  mgard_gen::refactor_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2],
                         dims.input[0], dims.input[1], dims.input[2], l_target,
                         v.data(), work, work2d, coords_x, coords_y, coords_z);

  work.clear();
  work2d.clear();

  const int size_ratio = sizeof(Real) / sizeof(int);
  std::vector<int> qv(nrow * ncol * nfib + size_ratio);

  mgard_gen::quantize_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2],
                         dims.input[0], dims.input[1], dims.input[2],
                         dims.nlevel, v.data(), qv, coords_x, coords_y,
                         coords_z, s, norm, tol);

  std::vector<unsigned char> out_data;

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);
  return buffer;
}

template <typename Real>
unsigned char *
refactor_qz(int nrow, int ncol, int nfib, const Real *u, int &outsize, Real tol,
            Real (*qoi)(int, int, int, std::vector<Real>), Real s) {
  std::vector<Real> v(u, u + nrow * ncol * nfib), work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array
  std::vector<Real> coords_x(ncol), coords_y(nrow),
      coords_z(nfib); // coordinate arrays

  const Dimensions2kPlus1<3> dims({nrow, ncol, nfib});
  const int l_target = dims.nlevel - 1;

  // dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  //  Real norm =  mgard_gen::ml2_norm3(0,  nrow,  ncol,  nfib ,  nrow,  ncol,
  //  nfib,   v, coords_x, coords_y, coords_z);

  // Real norm = mgard_common::max_norm(v);

  //  Real norm = 1.0; // absolute s-norm, need a switch for relative errors
  //  tol /= nlevel + 1 ;
  //  Real s = 0; // Defaulting to L8' compression for a start.

  //  norm = std::sqrt(norm/(nrow*nfib*ncol)); <- quant scaling goes here for s
  //  != 8'

  Real norm = mgard_gen::ml2_norm3(0, nrow, ncol, nfib, nrow, ncol, nfib, v,
                                   coords_x, coords_y, coords_z);

  norm = std::sqrt(norm) / std::sqrt(nrow * ncol * nfib);

  mgard_gen::prep_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2], dims.input[0],
                     dims.input[1], dims.input[2], l_target, v.data(), work,
                     work2d, coords_x, coords_y, coords_z);

  mgard_gen::refactor_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2],
                         dims.input[0], dims.input[1], dims.input[2], l_target,
                         v.data(), work, work2d, coords_x, coords_y, coords_z);

  work.clear();
  work2d.clear();

  const int size_ratio = sizeof(Real) / sizeof(int);
  std::vector<int> qv(nrow * ncol * nfib + size_ratio);

  mgard_gen::quantize_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2],
                         dims.input[0], dims.input[1], dims.input[2],
                         dims.nlevel, v.data(), qv, coords_x, coords_y,
                         coords_z, s, norm, tol);

  std::vector<unsigned char> out_data;

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);
  return buffer;
}

template <typename Real>
Real *recompose_udq(int nrow, int ncol, int nfib, unsigned char *data,
                    int data_len) {
  // Dummy equispaced coordinates.
  std::vector<Real> coords_x(ncol);
  std::vector<Real> coords_y(nrow);
  std::vector<Real> coords_z(nfib);
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);
  return recompose_udq(nrow, ncol, nfib, coords_x, coords_y, coords_z, data,
                       data_len);
}

template <typename Real>
Real *recompose_udq(int nrow, int ncol, int nfib, std::vector<Real> &coords_x,
                    std::vector<Real> &coords_y, std::vector<Real> &coords_z,
                    unsigned char *data, int data_len) {
  const TensorMeshHierarchy<3, Real> hierarchy({nrow, ncol, nfib});
  const int size_ratio = sizeof(Real) / sizeof(int);
  std::vector<int> out_data(nrow * ncol * nfib + size_ratio);
  std::vector<Real> work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array

  const Dimensions2kPlus1<3> dims({nrow, ncol, nfib});
  const int l_target = dims.nlevel - 1;

  mgard::decompress_memory_z(data, data_len, out_data.data(),
                             out_data.size() *
                                 sizeof(int)); // decompress input buffer
  Real *v = (Real *)malloc(nrow * ncol * nfib * sizeof(Real));

  mgard::dequantize_interleave(hierarchy, v, out_data);

  mgard_gen::recompose_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2],
                          dims.input[0], dims.input[1], dims.input[2], l_target,
                          v, work, work2d, coords_x, coords_y, coords_z);

  mgard_gen::postp_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2],
                      dims.input[0], dims.input[1], dims.input[2], l_target, v,
                      work, coords_x, coords_y, coords_z);

  return v;
}

template <typename Real>
Real *recompose_udq(int nrow, int ncol, int nfib, unsigned char *data,
                    int data_len, Real s) {
  // Dummy equispaced coordinates.
  std::vector<Real> coords_x(ncol);
  std::vector<Real> coords_y(nrow);
  std::vector<Real> coords_z(nfib);
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);
  return recompose_udq(nrow, ncol, nfib, coords_x, coords_y, coords_z, data,
                       data_len, s);
}

template <typename Real>
Real *recompose_udq(int nrow, int ncol, int nfib, std::vector<Real> &coords_x,
                    std::vector<Real> &coords_y, std::vector<Real> &coords_z,
                    unsigned char *data, int data_len, Real s) {
  int size_ratio = sizeof(Real) / sizeof(int);

  std::vector<int> out_data(nrow * ncol * nfib + size_ratio);
  std::vector<Real> work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array

  //    Real s = 0; // Defaulting to L2 compression for a start.
  Real norm = 1; // defaulting to absolute s, may switch to relative

  const Dimensions2kPlus1<3> dims({nrow, ncol, nfib});
  const int l_target = dims.nlevel - 1;

  mgard::decompress_memory_z(data, data_len, out_data.data(),
                             out_data.size() *
                                 sizeof(int)); // decompress input buffer
  Real *v = (Real *)malloc(nrow * ncol * nfib * sizeof(Real));

  //    outfile.write( reinterpret_cast<char*>( out_data.data() ),
  //    (nrow*ncol*nfib + size_ratio)*sizeof(int) );

  mgard_gen::dequantize_3D(
      dims.rnded[0], dims.rnded[1], dims.rnded[2], dims.input[0], dims.input[1],
      dims.input[2], dims.nlevel, v, out_data, coords_x, coords_y, coords_z, s);
  //    mgard::dequantize_2D_interleave(nrow, ncol*nfib, v, out_data) ;

  //    mgard_common::qread_2D_interleave(nrow,  ncol, nlevel, work.data(),
  //    out_file);

  // mgard_gen::dequant_3D(
  //     dims.rnded[0], dims.rnded[1], dims.rnded[2],
  //     dims.input[0], dims.input[1], dims.input[2],
  //     dims.nlevel, dims.nlevel,
  //     v, work.data(), coords_x, coords_y,  coords_z, s
  // );

  // std::ofstream outfile(out_file, std::ios::out | std::ios::binary);

  //    w
  mgard_gen::recompose_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2],
                          dims.input[0], dims.input[1], dims.input[2], l_target,
                          v, work, work2d, coords_x, coords_y, coords_z);

  mgard_gen::postp_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2],
                      dims.input[0], dims.input[1], dims.input[2], l_target, v,
                      work, coords_x, coords_y, coords_z);

  //    outfile.write( reinterpret_cast<char*>( v ),
  //    nrow*ncol*nfib*sizeof(Real) ;)
  return v;
}

template <typename Real>
unsigned char *refactor_qz_1D(int ncol, const Real *u, int &outsize, Real tol) {
  const Dimensions2kPlus1<1> dims({ncol});
  const TensorMeshHierarchy<1, Real> hierarchy({ncol});

  std::vector<Real> row_vec(ncol);
  std::vector<Real> v(u, u + ncol), work(ncol);

  Real norm = mgard_2d::mgard_common::max_norm(v);

  if (dims.is_2kplus1()) {
    // to be clean up.

    tol /= dims.nlevel + 1;

    const int l_target = dims.nlevel - 1;
    mgard::refactor_1D(ncol, l_target, v.data(), work, row_vec);

    work.clear();
    row_vec.clear();

    int size_ratio = sizeof(Real) / sizeof(int);
    std::vector<int> qv(ncol + size_ratio);

    mgard::quantize_interleave(hierarchy, v.data(), qv, norm, tol);

    std::vector<unsigned char> out_data;

    return mgard::compress_memory_huffman(qv, out_data, outsize);
  } else {
    std::vector<Real> coords_x(ncol);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);

    tol /= dims.nlevel + 1;

    const int l_target = dims.nlevel - 1;

    mgard_2d::mgard_gen::prep_1D(dims.rnded[0], dims.input[0], l_target,
                                 v.data(), work, coords_x, row_vec);

    mgard_2d::mgard_gen::refactor_1D(dims.rnded[0], dims.input[0], l_target,
                                     v.data(), work, coords_x, row_vec);

    work.clear();
    row_vec.clear();

    const int size_ratio = sizeof(Real) / sizeof(int);
    std::vector<int> qv(ncol + size_ratio);

    mgard::quantize_interleave(hierarchy, v.data(), qv, norm, tol);

    std::vector<unsigned char> out_data;

    return mgard::compress_memory_huffman(qv, out_data, outsize);
  }
}

template <typename Real>
unsigned char *refactor_qz_2D(int nrow, int ncol, const Real *u, int &outsize,
                              Real tol) {
  const Dimensions2kPlus1<2> dims({nrow, ncol});
  const TensorMeshHierarchy<2, Real> hierarchy({nrow, ncol});
  if (dims.is_2kplus1()) {
    std::vector<Real> row_vec(ncol);
    std::vector<Real> col_vec(nrow);
    std::vector<Real> v(u, u + nrow * ncol), work(nrow * ncol);

    Real norm = mgard_2d::mgard_common::max_norm(v);

    // TODO: Elsewhere we have divided by `nlevel + 2`. I believe it has to do
    // with the extra level (not present here) with dimensions not of the form
    // `2^k + 1`.
    tol /= dims.nlevel + 1;

    const int l_target = dims.nlevel - 1;
    mgard::refactor(nrow, ncol, l_target, v.data(), work, row_vec, col_vec);
    work.clear();
    row_vec.clear();
    col_vec.clear();

    const int size_ratio = sizeof(Real) / sizeof(int);
    std::vector<int> qv(nrow * ncol + size_ratio);

    mgard::quantize_interleave(hierarchy, v.data(), qv, norm, tol);

    std::vector<unsigned char> out_data;

    mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);
    outsize = out_data.size();
    unsigned char *buffer = (unsigned char *)malloc(outsize);
    std::copy(out_data.begin(), out_data.end(), buffer);
    return buffer;
  } else {
    // Dummy equispaced coordinates.
    std::vector<Real> coords_x(ncol);
    std::vector<Real> coords_y(nrow);
    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);
    return refactor_qz_2D(nrow, ncol, coords_x, coords_y, u, outsize, tol);
  }
}

template <typename Real>
unsigned char *refactor_qz_2D(int nrow, int ncol, std::vector<Real> &coords_x,
                              std::vector<Real> &coords_y, const Real *u,
                              int &outsize, Real tol) {
  const TensorMeshHierarchy<2, Real> hierarchy({nrow, ncol});

  std::vector<Real> row_vec(ncol);
  std::vector<Real> col_vec(nrow);
  std::vector<Real> v(u, u + nrow * ncol), work(nrow * ncol);

  Real norm = mgard_2d::mgard_common::max_norm(v);

  const Dimensions2kPlus1<2> dims({nrow, ncol});

  // TODO: in the `float` implementation, we divide by `nlevel + 2`.
  tol /= dims.nlevel + 1;

  const int l_target = dims.nlevel - 1;

  mgard_2d::mgard_gen::prep_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                               dims.input[1], l_target, v.data(), work,
                               coords_x, coords_y, row_vec, col_vec);

  mgard_2d::mgard_gen::refactor_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                                   dims.input[1], l_target, v.data(), work,
                                   coords_x, coords_y, row_vec, col_vec);

  work.clear();
  col_vec.clear();
  row_vec.clear();

  const int size_ratio = sizeof(Real) / sizeof(int);
  std::vector<int> qv(nrow * ncol + size_ratio);

  tol /= dims.nlevel + 1;
  mgard::quantize_interleave(hierarchy, v.data(), qv, norm, tol);

  std::vector<unsigned char> out_data;

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);
  return buffer;
}

template <typename Real>
unsigned char *refactor_qz_2D(int nrow, int ncol, const Real *u, int &outsize,
                              Real tol, Real s) {
  const Dimensions2kPlus1<2> dims({nrow, ncol});
  if (dims.is_2kplus1()) {
    std::vector<Real> row_vec(ncol);
    std::vector<Real> col_vec(nrow);
    std::vector<Real> v(u, u + nrow * ncol), work(nrow * ncol);

    Real norm = mgard_2d::mgard_common::max_norm(v);

    tol /= dims.nlevel + 1;

    const int l_target = dims.nlevel - 1;
    mgard::refactor(nrow, ncol, l_target, v.data(), work, row_vec, col_vec);
    work.clear();
    row_vec.clear();
    col_vec.clear();

    std::vector<Real> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);

    const int size_ratio = sizeof(Real) / sizeof(int);
    std::vector<int> qv(nrow * ncol + size_ratio);

    // mgard::quantize_2D_interleave (nrow, ncol, v.data(), qv, norm, tol);

    mgard_gen::quantize_2D(nrow, ncol, nrow, ncol, dims.nlevel, v.data(), qv,
                           coords_x, coords_y, s, norm, tol);

    std::vector<unsigned char> out_data;

    mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);
    outsize = out_data.size();
    unsigned char *buffer = (unsigned char *)malloc(outsize);
    std::copy(out_data.begin(), out_data.end(), buffer);
    return buffer;
  } else {
    // Dummy equispaced coordinates.
    std::vector<Real> coords_x(ncol);
    std::vector<Real> coords_y(nrow);
    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);
    return refactor_qz_2D(nrow, ncol, coords_x, coords_y, u, outsize, tol, s);
  }
}

template <typename Real>
unsigned char *refactor_qz_2D(int nrow, int ncol, std::vector<Real> &coords_x,
                              std::vector<Real> &coords_y, const Real *u,
                              int &outsize, Real tol, Real s) {

  std::vector<Real> row_vec(ncol);
  std::vector<Real> col_vec(nrow);
  std::vector<Real> v(u, u + nrow * ncol), work(nrow * ncol);

  Real norm = mgard_2d::mgard_common::max_norm(v);

  const Dimensions2kPlus1<2> dims({nrow, ncol});
  tol /= dims.nlevel + 1;

  const int l_target = dims.nlevel - 1;

  mgard_2d::mgard_gen::prep_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                               dims.input[1], l_target, v.data(), work,
                               coords_x, coords_y, row_vec, col_vec);

  mgard_2d::mgard_gen::refactor_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                                   dims.input[1], l_target, v.data(), work,
                                   coords_x, coords_y, row_vec, col_vec);

  work.clear();
  col_vec.clear();
  row_vec.clear();

  const int size_ratio = sizeof(Real) / sizeof(int);
  std::vector<int> qv(nrow * ncol + size_ratio);

  mgard_gen::quantize_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                         dims.input[1], dims.nlevel, v.data(), qv, coords_x,
                         coords_y, s, norm, tol);

  std::vector<unsigned char> out_data;

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);
  return buffer;
}

template <typename Real>
Real *recompose_udq_1D_huffman(int ncol, unsigned char *data, int data_len) {
  const Dimensions2kPlus1<1> dims({ncol});
  const TensorMeshHierarchy<1, Real> hierarchy({ncol});
  const int size_ratio = sizeof(Real) / sizeof(int);

  if (dims.is_2kplus1()) // input is (2^p + 1)
  {
    // to be cleaned up.
    const int l_target = dims.nlevel - 1;
#if 0
    int ncol_new = ncol;

    int nlevel_new;
    set_number_of_levels(nrow_new, ncol_new, nlevel_new);

    const int l_target = nlevel_new - 1;
#endif
    std::vector<int> out_data(ncol + size_ratio);

    mgard::decompress_memory_huffman(data, data_len, out_data);

    Real *v = (Real *)malloc(ncol * sizeof(Real));

    mgard::dequantize_interleave(hierarchy, v, out_data);
    out_data.clear();

    std::vector<Real> row_vec(ncol);
    std::vector<Real> work(ncol);
#if 1
    mgard::recompose_1D(ncol, l_target, v, work, row_vec);

    return v;
#endif
  } else {
    std::vector<Real> coords_x(ncol);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);

    const int l_target = dims.nlevel - 1;

    std::vector<int> out_data(ncol + size_ratio);

    mgard::decompress_memory_huffman(data, data_len, out_data);

    Real *v = (Real *)malloc(ncol * sizeof(Real));

    mgard::dequantize_interleave(hierarchy, v, out_data);

    std::vector<Real> row_vec(ncol);
    std::vector<Real> work(ncol);

    mgard_2d::mgard_gen::recompose_1D(dims.rnded[0], dims.input[0], l_target, v,
                                      work, coords_x, row_vec);

    mgard_2d::mgard_gen::postp_1D(dims.rnded[0], dims.input[0], l_target, v,
                                  work, coords_x, row_vec);

    return v;
  }
}

template <typename Real>
Real *recompose_udq_1D(int ncol, unsigned char *data, int data_len) {
  const Dimensions2kPlus1<1> dims({ncol});
  const TensorMeshHierarchy<1, Real> hierarchy({ncol});
  const int size_ratio = sizeof(Real) / sizeof(int);

  if (dims.is_2kplus1()) {
    // to be cleaned up.
    const int l_target = dims.nlevel - 1;
#if 0
    int ncol_new = ncol;

    int nlevel_new;
    set_number_of_levels(nrow_new, ncol_new, nlevel_new);

    const int l_target = nlevel_new - 1;
#endif
    std::vector<int> out_data(ncol + size_ratio);

    mgard::decompress_memory_z(data, data_len, out_data.data(),
                               out_data.size() * sizeof(int));

    Real *v = (Real *)malloc(ncol * sizeof(Real));

    mgard::dequantize_interleave(hierarchy, v, out_data);
    out_data.clear();

    std::vector<Real> row_vec(ncol);
    std::vector<Real> work(ncol);
#if 1
    mgard::recompose_1D(ncol, l_target, v, work, row_vec);

    return v;
#endif
  } else {
    std::vector<Real> coords_x(ncol);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);

    const Dimensions2kPlus1<1> dims({ncol});
    const int l_target = dims.nlevel - 1;

    std::vector<int> out_data(ncol + size_ratio);
    mgard::decompress_memory_z(data, data_len, out_data.data(),
                               out_data.size() * sizeof(int));

    Real *v = (Real *)malloc(ncol * sizeof(Real));

    mgard::dequantize_interleave(hierarchy, v, out_data);

    std::vector<Real> row_vec(ncol);
    std::vector<Real> work(ncol);

    mgard_2d::mgard_gen::recompose_1D(dims.rnded[0], dims.input[0], l_target, v,
                                      work, coords_x, row_vec);

    mgard_2d::mgard_gen::postp_1D(dims.rnded[0], dims.input[0], l_target, v,
                                  work, coords_x, row_vec);

    return v;
  }
}

template <typename Real>
Real *recompose_udq_2D(int nrow, int ncol, unsigned char *data, int data_len) {
  const Dimensions2kPlus1<2> dims({nrow, ncol});
  const TensorMeshHierarchy<2, Real> hierarchy({nrow, ncol});
  if (dims.is_2kplus1()) {
    const int size_ratio = sizeof(Real) / sizeof(int);
    const int l_target = dims.nlevel - 1;

    std::vector<int> out_data(nrow * ncol + size_ratio);

    mgard::decompress_memory_z(data, data_len, out_data.data(),
                               out_data.size() *
                                   sizeof(int)); // decompress input buffer

    Real *v = (Real *)malloc(nrow * ncol * sizeof(Real));

    mgard::dequantize_interleave(hierarchy, v, out_data);
    out_data.clear();

    std::vector<Real> row_vec(ncol);
    std::vector<Real> col_vec(nrow);
    std::vector<Real> work(nrow * ncol);

    mgard::recompose(nrow, ncol, l_target, v, work, row_vec, col_vec);

    return v;

  } else {
    // Dummy equispaced coordinates.
    std::vector<Real> coords_x(ncol);
    std::vector<Real> coords_y(nrow);
    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);
    return recompose_udq_2D(nrow, ncol, coords_x, coords_y, data, data_len);
  }
}

template <typename Real>
Real *recompose_udq_2D(int nrow, int ncol, std::vector<Real> &coords_x,
                       std::vector<Real> &coords_y, unsigned char *data,
                       int data_len) {
  const TensorMeshHierarchy<2, Real> hierarchy({nrow, ncol});
  const int size_ratio = sizeof(Real) / sizeof(int);

  const Dimensions2kPlus1<2> dims({nrow, ncol});
  const int l_target = dims.nlevel - 1;

  std::vector<int> out_data(nrow * ncol + size_ratio);

  mgard::decompress_memory_z(data, data_len, out_data.data(),
                             out_data.size() *
                                 sizeof(int)); // decompress input buffer

  Real *v = (Real *)malloc(nrow * ncol * sizeof(Real));

  mgard::dequantize_interleave(hierarchy, v, out_data);

  std::vector<Real> row_vec(ncol);
  std::vector<Real> col_vec(nrow);
  std::vector<Real> work(nrow * ncol);

  mgard_2d::mgard_gen::recompose_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                                    dims.input[1], l_target, v, work, coords_x,
                                    coords_y, row_vec, col_vec);

  mgard_2d::mgard_gen::postp_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                                dims.input[1], l_target, v, work, coords_x,
                                coords_y, row_vec, col_vec);

  return v;
}

template <typename Real>
Real *recompose_udq_2D(int nrow, int ncol, unsigned char *data, int data_len,
                       Real s) {
  const Dimensions2kPlus1<2> dims({nrow, ncol});
  if (dims.is_2kplus1()) {
    const int size_ratio = sizeof(Real) / sizeof(int);
    const int l_target = dims.nlevel - 1;

    std::vector<Real> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);

    std::vector<int> out_data(nrow * ncol + size_ratio);

    mgard::decompress_memory_z(data, data_len, out_data.data(),
                               out_data.size() *
                                   sizeof(int)); // decompress input buffer

    Real *v = (Real *)malloc(nrow * ncol * sizeof(Real));

    //      mgard::dequantize_2D_interleave(nrow, ncol, v, out_data) ;
    mgard_gen::dequantize_2D(nrow, ncol, nrow, ncol, dims.nlevel, v, out_data,
                             coords_x, coords_y, s);
    out_data.clear();

    std::vector<Real> row_vec(ncol);
    std::vector<Real> col_vec(nrow);
    std::vector<Real> work(nrow * ncol);

    mgard::recompose(nrow, ncol, l_target, v, work, row_vec, col_vec);

    return v;

  } else {
    // Dummy equispaced coordinates.
    std::vector<Real> coords_x(ncol);
    std::vector<Real> coords_y(nrow);
    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);
    return recompose_udq_2D(nrow, ncol, coords_x, coords_y, data, data_len, s);
  }
}

template <typename Real>
Real *recompose_udq_2D(int nrow, int ncol, std::vector<Real> &coords_x,
                       std::vector<Real> &coords_y, unsigned char *data,
                       int data_len, Real s) {
  const int size_ratio = sizeof(Real) / sizeof(int);
  std::vector<int> out_data(nrow * ncol + size_ratio);

  const Dimensions2kPlus1<2> dims({nrow, ncol});
  const int l_target = dims.nlevel - 1;

  mgard::decompress_memory_z(data, data_len, out_data.data(),
                             out_data.size() * sizeof(int));

  Real *v = (Real *)malloc(nrow * ncol * sizeof(Real));

  mgard_gen::dequantize_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                           dims.input[1], dims.nlevel, v, out_data, coords_x,
                           coords_y, s);

  std::vector<Real> row_vec(ncol);
  std::vector<Real> col_vec(nrow);
  std::vector<Real> work(nrow * ncol);

  mgard_2d::mgard_gen::recompose_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                                    dims.input[1], l_target, v, work, coords_x,
                                    coords_y, row_vec, col_vec);

  mgard_2d::mgard_gen::postp_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                                dims.input[1], l_target, v, work, coords_x,
                                coords_y, row_vec, col_vec);

  return v;
}

//   unsigned char *
//   refactor_qz_1D (int nrow,  const Real *u, int &outsize, Real tol)
//   {

//   std::vector<Real> v(u, u+nrow), work(nrow);

//   Real norm = mgard_common::max_norm(v);

//   std::vector<Real> coords_x;
//   const Dimensions2kPlus1<1> dims({nrow});
//   const int l_target = dims.nlevel-1;

//   std::iota(std::begin(coords_x), std::end(coords_x), 0);

//  // mgard::refactor_1D(
//  //     dims.nlevel - 1, v, work, coords_x,
//  //     dims.rnded[0], dims.input[0]
//  // );

//   work.clear ();

//   int size_ratio = sizeof (Real) / sizeof (int);
//   std::vector<int> qv (nrow * ncol + size_ratio);

//   tol /= nlevel + 1;
//   mgard::quantize_2D_interleave (nrow, 1, v.data (), qv, norm, tol);

//   std::vector<unsigned char> out_data;
//   mgard::compress_memory (qv.data (), sizeof (int) * qv.size (), out_data);

//   outsize = out_data.size ();
//   unsigned char *buffer = (unsigned char *)malloc (outsize);
//   std::copy (out_data.begin (), out_data.end (), buffer);
//   return buffer;
// }

//  Real* recompose_udq_1D(int nrow,  unsigned char *data, int data_len)
//   {
//   int size_ratio = sizeof(Real)/sizeof(int);
//     {
//       std::vector<Real> coords_x;

// This looks wrong: `nc` depends on `nrow`, not `ncol`.
//       int nlevel = std::log2(nrow-1);
//       int nc = std::pow(2, nlevel ) + 1; //ncol new
// k
//       const int l_target = nlevel-1;

//       std::vector<int> out_data(nrow + size_ratio);

//       std::iota(std::begin(coords_x), std::end(coords_x), 0);

//       mgard::decompress_memory(data, data_len, out_data.data(),
//       out_data.size()*sizeof(int)); // decompress input buffer

//       Real *v = (Real *)malloc (nrow*sizeof(Real));

//       mgard::dequantize_2D_interleave(nrow, 1, v, out_data) ;

//       std::vector<Real> work(nrow);

//       mgard::recompose_1D(nlevel, v,   work, coords_x,  nr,  nrow );

//       return v;
//     }
// }

template <typename Real>
void mass_matrix_multiply(const int l, std::vector<Real> &v) {
  const std::size_t stride = stride_from_index_difference(l);
  // The entries of the mass matrix are scaled by `h / 6`. We assume that the
  // cells of the finest level have width `6`, so that the cells on this level
  // have width `6 * stride`. `factor` is then `h / 6`.
  const Real factor = stride;
  Real left, middle, right;
  middle = v.front();
  right = v.at(stride);
  v.front() = factor * (2 * middle + right);
  for (auto it = v.begin() + stride; it < v.end() - stride; it += stride) {
    left = middle;
    middle = right;
    right = *(it + stride);
    *it = factor * (left + 4 * middle + right);
  }
  left = middle;
  middle = right;
  v.back() = factor * (left + 2 * middle);
}

template <typename Real>
void solve_tridiag_M(const int l, std::vector<Real> &v) {
  // The system is solved using the Thomas algorithm. See <https://
  // en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm>. In case the article
  // changes, the algorithm is copied here.
  //    // [b_1, …, b_n] is the diagonal of the matrix. `[a_2, …, a_n]` is the
  //    // subdiagonal, and `[c_1, …, c_{n - 1}]` is the superdiagonal.
  //    // `[d_1, …, d_n]` is the righthand side, and `[x_1, …, x_n]` is the
  //    // solution.
  //    for i = 2, …, n do
  //      w_i = a_i / b_{i - 1}
  //      b_i := b_i - w_i * c_{i - 1}
  //      d_i := d_i - w_i * d_{i - 1}
  //    end
  //    x_n = d_n / b_n
  //    for i = n - 1, …, 1 do
  //      x_i = (d_i - c_i * x_{i + 1}) / b_i
  //    end
  // The mass matrix  entries are scaled by `h / 6`. We postpone accounting for
  // the scaling to the backward sweep.
  const std::size_t stride = stride_from_index_difference(l);
  // See the note in `mass_matrix_multiply`.
  const Real factor = stride;

  // The system size is `(v.size() - 1) / stride + 1`, but we don't need to
  // store the final divisor. In the notation above, this vector will be
  // `[b_1, …, b_{n - 1}]` (after the modification in the forward sweep).
  std::vector<Real> divisors((v.size() - 1) / stride);
  typename std::vector<Real>::iterator p = divisors.begin();
  // This is `b_{i - 1}`.
  Real previous_divisor = *p = 2;
  // This is `d_{i - 1}`.
  Real previous_entry = v.front();
  // Forward sweep (except for last entry).
  for (auto it = std::begin(v) + stride; it < std::end(v) - stride;
       it += stride) {
    // The numerator is really `a`.
    const Real w = 1 / previous_divisor;
    // The last term is really `w * c`.
    previous_divisor = *++p = 4 - w;
    previous_entry = *it -= w * previous_entry;
  }

  // Forward sweep (last entry) and start of backward sweep (first entry).
  {
    // The numerator is really `a`.
    const Real w = 1 / previous_divisor;
    // Don't need to update `previous_divisor` and `previous_entry` as we won't
    // be using them.
    Real &entry = v.back();
    entry -= w * previous_entry;
    // Don't need need to write to `*d` or increment `d` as we're using the
    // divisor immediately.
    // The last term is really `w * c`.
    previous_entry = entry /= 2 - w;
  }

  // Backward sweep (remaining entries).
  typename std::vector<Real>::reverse_iterator q = v.rbegin();
  for (auto it = v.rbegin() + stride; it < v.rend(); it += stride) {
    Real &entry = *it;
    // The subtrahend is really `c * previous_entry`.
    entry -= previous_entry;
    previous_entry = entry /= *p--;
    *q /= factor;
    q += stride;
  }
  *q /= factor;
}

template <typename Real> void restriction(const int l, std::vector<Real> &v) {
  if (!l) {
    throw std::domain_error("cannot restrict from the finest level");
  }
  const std::size_t stride = stride_from_index_difference(l);
  const std::size_t Pstride = stride_from_index_difference(l - 1);

  Real left, right;
  right = *(std::begin(v) + Pstride);
  v.front() += 0.5 * right; // first element
  for (auto it = std::begin(v) + stride; it <= std::end(v) - stride;
       it += stride) {
    left = right;
    right = *(it + Pstride);
    *it += 0.5 * (left + right);
  }
  left = right;
  v.back() += 0.5 * left; // last element
}

template <typename Real>
void interpolate_from_level_nMl(const int l, std::vector<Real> &v) {
  if (!l) {
    throw std::domain_error("cannot interpolate from the finest level");
  }
  const std::size_t stride = stride_from_index_difference(l);
  const std::size_t Pstride = stride_from_index_difference(l - 1);

  Real left = v.front();
  for (auto p = std::begin(v) + Pstride, q = std::begin(v) + stride;
       q < std::end(v); p += stride, q += stride) {
    const Real right = *q;
    *p = 0.5 * (left + right);
    left = right;
  }
}

template <typename Real> void pi_lminus1(const int l, std::vector<Real> &v) {
  const Dimensions2kPlus1<1> dims({v.size()});
  if (dims.nlevel == l) {
    throw std::domain_error("cannot interpolate from the coarsest level");
  }
  const std::size_t stride = stride_from_index_difference(l);
  const std::size_t Cstride = stride_from_index_difference(l + 1);

  Real left = v.front();
  for (auto p = v.begin() + stride, q = v.begin() + Cstride; q < v.end();
       p += Cstride, q += Cstride) {
    const Real right = *q;
    *p -= 0.5 * (left + right);
    left = right;
  }
}

// Gary New
template <typename Real>
void pi_Ql(const int ncol, const int l, Real *v, std::vector<Real> &row_vec) {
  // Restrict data to coarser level

  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  //  std::vector<Real> row_vec(ncol), col_vec(nrow)   ;

  for (int jcol = 0; jcol < ncol; ++jcol) {
    row_vec[jcol] = v[jcol];
  }

  pi_lminus1(l, row_vec);

  for (int jcol = 0; jcol < ncol; ++jcol) {
    v[jcol] = row_vec[jcol];
  }
}

template <typename Real>
void pi_Ql(const int nrow, const int ncol, const int l, Real *const v,
           std::vector<Real> &row_vec, std::vector<Real> &col_vec) {
  const Dimensions2kPlus1<2> dims({nrow, ncol});
  if (dims.nlevel == l) {
    throw std::domain_error("cannot interpolate from the coarsest level");
  }
  // TODO: Remove these casts once everything has been moved to `std::size_t`.
  const int stride = static_cast<int>(stride_from_index_difference(l));
  const int Cstride = static_cast<int>(stride_from_index_difference(l + 1));

  // Do the rows existing in the coarser level.
  for (int irow = 0; irow < nrow; irow += Cstride) {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = v[get_index(ncol, irow, jcol)];
    }

    pi_lminus1(l, row_vec);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      v[get_index(ncol, irow, jcol)] = row_vec[jcol];
    }
  }

  if (nrow > 1) {
    // Do the columns existing in the coarser level.
    for (int jcol = 0; jcol < ncol; jcol += Cstride) {
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = v[get_index(ncol, irow, jcol)];
      }

      pi_lminus1(l, col_vec);

      for (int irow = 0; irow < nrow; ++irow) {
        v[get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }

    // Now the new-new stuff
    for (int irow = stride; irow + stride < nrow; irow += Cstride) {
      for (int jcol = stride; jcol + stride < ncol; jcol += Cstride) {
        v[get_index(ncol, irow, jcol)] -=
            0.25 * (v[get_index(ncol, irow - stride, jcol - stride)] +
                    v[get_index(ncol, irow - stride, jcol + stride)] +
                    v[get_index(ncol, irow + stride, jcol - stride)] +
                    v[get_index(ncol, irow + stride, jcol + stride)]);
      }
    }
  }
}

template <std::size_t N, typename Real>
void assign_num_level(const TensorMeshHierarchy<N, Real> &hierarchy,
                      const int l, Real *const v, const Real num) {
  const Dimensions2kPlus1<N> dims(hierarchy.meshes.back().shape);
  using It = LevelValuesIterator<N, Real>;
  const RangeSlice<It> values = dims.on_nodes(v, l);
  It p = values.begin();
  const It values_end = values.end();
  while (p != values_end) {
    *p++ = num;
  }
}

template <std::size_t N, typename Real>
void copy_level(const TensorMeshHierarchy<N, Real> &hierarchy, const int l,
                Real const *const v, Real *const work) {
  const Dimensions2kPlus1<N> dims(hierarchy.meshes.back().shape);
  using It = LevelValuesIterator<N, const Real>;
  using Jt = LevelValuesIterator<N, Real>;
  const RangeSlice<It> source = dims.on_nodes(v, l);
  const RangeSlice<Jt> destination = dims.on_nodes(work, l);
  It p = source.begin();
  Jt q = destination.begin();
  const It source_end = source.end();
  while (p != source.end()) {
    *q++ = *p++;
  }
}

template <std::size_t N, typename Real>
void add_level(const TensorMeshHierarchy<N, Real> &hierarchy, const int l,
               Real *const v, Real const *const work) {
  const Dimensions2kPlus1<N> dims(hierarchy.meshes.back().shape);
  using It = LevelValuesIterator<N, Real>;
  using Jt = LevelValuesIterator<N, const Real>;
  const RangeSlice<It> target = dims.on_nodes(v, l);
  const RangeSlice<Jt> increment = dims.on_nodes(work, l);
  It p = target.begin();
  Jt q = increment.begin();
  const It target_end = target.end();
  while (p != target_end) {
    *p++ += *q++;
  }
}

template <std::size_t N, typename Real>
void subtract_level(const TensorMeshHierarchy<N, Real> &hierarchy, const int l,
                    Real *const v, Real const *const work) {
  const Dimensions2kPlus1<N> dims(hierarchy.meshes.back().shape);
  using It = LevelValuesIterator<N, Real>;
  using Jt = LevelValuesIterator<N, const Real>;
  const RangeSlice<It> target = dims.on_nodes(v, l);
  const RangeSlice<Jt> decrement = dims.on_nodes(work, l);
  It p = target.begin();
  Jt q = decrement.begin();
  const It target_end = target.end();
  while (p != target_end) {
    *p++ -= *q++;
  }
}

template <std::size_t N, typename Real>
void quantize_interleave(const TensorMeshHierarchy<N, Real> &hierarchy,
                         Real const *const v, std::vector<int> &work,
                         const Real norm, const Real tol) {
  static_assert(sizeof(Real) % sizeof(int) == 0,
                "`int` size does not divide `Real` size");
  const std::size_t size_ratio = sizeof(Real) / sizeof(int);
  const mgard::LinearQuantizer<Real, int> quantizer(norm * tol);
  std::memcpy(work.data(), &quantizer.quantum, sizeof(Real));
  for (std::size_t index = 0; index < hierarchy.ndof(); ++index) {
    work[size_ratio + index] = quantizer(v[index]);
  }
}

template <std::size_t N, typename Real>
void dequantize_interleave(const TensorMeshHierarchy<N, Real> &hierarchy,
                           Real *const v, const std::vector<int> &work) {
  static_assert(sizeof(Real) % sizeof(int) == 0,
                "`int` size does not divide `Real` size");
  const std::size_t size_ratio = sizeof(Real) / sizeof(int);

  Real quantum;
  std::memcpy(&quantum, work.data(), sizeof(Real));
  const mgard::LinearDequantizer<int, Real> quantizer(quantum);

  for (std::size_t index = 0; index < hierarchy.ndof(); ++index) {
    v[index] = quantizer(work[size_ratio + index]);
  }
}

template <typename Real>
void qwrite_2D_interleave(const int nrow, const int ncol, const int nlevel,
                          const int l, Real *v, const Real tol,
                          const std::string outfile) {

  int stride = std::pow(2, l); // current stride

  Real norm = 0;

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      Real ntest = std::abs(v[get_index(ncol, irow, jcol)]);
      if (ntest > norm)
        norm = ntest;
    }
  }

  const mgard::LinearQuantizer<Real, int> quantizer(norm * tol / (nlevel + 1));

  gzFile out_file = gzopen(outfile.c_str(), "w6b");
  gzwrite(out_file, &quantizer.quantum, sizeof(Real));

  int prune_count = 0;
  for (auto index = 0; index < ncol * nrow; ++index) {
    const int n = quantizer(v[index]);
    if (n == 0)
      ++prune_count;
    gzwrite(out_file, &n, sizeof(int));
  }

  // std::cout  << "Pruned : " << prune_count << " Reduction : "
  //            << (Real)nrow * ncol / (nrow * ncol - prune_count) << "\n";
  gzclose(out_file);
}

// Gary New
template <typename Real>
void refactor_1D(const int ncol, const int l_target, Real *v,
                 std::vector<Real> &work, std::vector<Real> &row_vec) {
  const TensorMeshHierarchy<1, Real> hierarchy({ncol});
  for (int l = 0; l < l_target; ++l) {

    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride
#if 1
    pi_Ql(ncol, l, v, row_vec); // rename!. v@l has I-\Pi_l Q_l+1 u
#endif
    // copy the nodal values of v on l  to matrix work
    copy_level(hierarchy, l, v, work.data());

    assign_num_level(hierarchy, l + 1, work.data(), static_cast<Real>(0.0));

    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[jcol];
    }

    mass_matrix_multiply(l, row_vec);

    restriction(l + 1, row_vec);

    solve_tridiag_M(l + 1, row_vec);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[jcol] = row_vec[jcol];
    }

    add_level(hierarchy, l + 1, v, work.data()); // Qu_l = \Pi_l Q_{l+1}u + z_l
  }
}

template <typename Real>
void refactor(const int nrow, const int ncol, const int l_target, Real *v,
              std::vector<Real> &work, std::vector<Real> &row_vec,
              std::vector<Real> &col_vec) {
  const TensorMeshHierarchy<2, Real> hierarchy({nrow, ncol});
  // refactor
  //  //std::cout  << "refactoring" << "\n";

  for (int l = 0; l < l_target; ++l) {

    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride

    pi_Ql(nrow, ncol, l, v, row_vec,
          col_vec); // rename!. v@l has I-\Pi_l Q_l+1 u
    // copy the nodal values of v on l  to matrix work
    copy_level(hierarchy, l, v, work.data());

    assign_num_level(hierarchy, l + 1, work.data(), static_cast<Real>(0.0));

    // row-sweep
    for (int irow = 0; irow < nrow; ++irow) {
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[get_index(ncol, irow, jcol)];
      }

      mass_matrix_multiply(l, row_vec);

      restriction(l + 1, row_vec);

      solve_tridiag_M(l + 1, row_vec);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[get_index(ncol, irow, jcol)] = row_vec[jcol];
      }
    }

    // column-sweep
    if (nrow > 1) // do this if we have an 2-dimensional array
    {
      for (int jcol = 0; jcol < ncol; jcol += Cstride) {
        for (int irow = 0; irow < nrow; ++irow) {
          col_vec[irow] = work[get_index(ncol, irow, jcol)];
        }

        mass_matrix_multiply(l, col_vec);

        restriction(l + 1, col_vec);
        solve_tridiag_M(l + 1, col_vec);

        for (int irow = 0; irow < nrow; ++irow) {
          work[get_index(ncol, irow, jcol)] = col_vec[irow];
        }
      }
    }

    // Solved for (z_l, phi_l) = (c_{l+1}, vl)

    // Qu_l = \Pi_l Q_{l+1}u + z_l
    add_level(hierarchy, l + 1, v, work.data());
  }
}

// Gary New
template <typename Real>
void recompose_1D(const int ncol, const int l_target, Real *v,
                  std::vector<Real> &work, std::vector<Real> &row_vec) {
  const TensorMeshHierarchy<1, Real> hierarchy({ncol});

  // recompose

  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;

    // copy the nodal values of cl on l-1 (finer level) to matrix work
    copy_level(hierarchy, l - 1, v, work.data());
    // zero out nodes of l on cl
    assign_num_level(hierarchy, l, work.data(), static_cast<Real>(0.0));

    // row-sweep
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[jcol];
    }

    mass_matrix_multiply(l - 1, row_vec);

    restriction(l, row_vec);
    solve_tridiag_M(l, row_vec);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[jcol] = row_vec[jcol];
    }

    subtract_level(hierarchy, l, work.data(), v); // do -(Qu - zl)

    // row-sweep
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[jcol];
    }

    interpolate_from_level_nMl(l, row_vec);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[jcol] = row_vec[jcol];
    }

    // zero out nodes of l on cl
    assign_num_level(hierarchy, l, v, static_cast<Real>(0.0));
    subtract_level(hierarchy, l - 1, v, work.data());
  }
}

template <typename Real>
void recompose(const int nrow, const int ncol, const int l_target, Real *v,
               std::vector<Real> &work, std::vector<Real> &row_vec,
               std::vector<Real> &col_vec) {
  const TensorMeshHierarchy<2, Real> hierarchy({nrow, ncol});

  // recompose

  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;

    // copy the nodal values of cl on l-1 (finer level) to matrix work
    copy_level(hierarchy, l - 1, v, work.data());
    // zero out nodes of l on cl
    assign_num_level(hierarchy, l, work.data(), static_cast<Real>(0.0));

    // row-sweep
    for (int irow = 0; irow < nrow; ++irow) {
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[get_index(ncol, irow, jcol)];
      }

      mass_matrix_multiply(l - 1, row_vec);

      restriction(l, row_vec);
      solve_tridiag_M(l, row_vec);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[get_index(ncol, irow, jcol)] = row_vec[jcol];
      }
    }

    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) // check if we have 1-D array..
    {
      for (int jcol = 0; jcol < ncol; jcol += stride) {
        for (int irow = 0; irow < nrow; ++irow) {
          col_vec[irow] = work[get_index(ncol, irow, jcol)];
        }

        mass_matrix_multiply(l - 1, col_vec);

        restriction(l, col_vec);
        solve_tridiag_M(l, col_vec);

        for (int irow = 0; irow < nrow; ++irow) {
          work[get_index(ncol, irow, jcol)] = col_vec[irow];
        }
      }
    }
    subtract_level(hierarchy, l, work.data(), v); // do -(Qu - zl)

    // row-sweep
    for (int irow = 0; irow < nrow; irow += stride) {
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[get_index(ncol, irow, jcol)];
      }

      interpolate_from_level_nMl(l, row_vec);

      for (int jcol = 0; jcol < ncol; ++jcol) {
        work[get_index(ncol, irow, jcol)] = row_vec[jcol];
      }
    }

    // column-sweep, this is the slow one! Need something like column_copy
    if (nrow > 1) {
      for (int jcol = 0; jcol < ncol; jcol += Pstride) {
        for (int irow = 0; irow < nrow; ++irow) // copy all rows
        {
          col_vec[irow] = work[get_index(ncol, irow, jcol)];
        }

        interpolate_from_level_nMl(l, col_vec);

        for (int irow = 0; irow < nrow; ++irow) {
          work[get_index(ncol, irow, jcol)] = col_vec[irow];
        }
      }
    }
    // zero out nodes of l on cl
    assign_num_level(hierarchy, l, v, static_cast<Real>(0.0));
    subtract_level(hierarchy, l - 1, v, work.data());
  }
}

} // end namespace mgard

#endif
