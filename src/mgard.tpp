// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ozan Tugluk
//
// version: 0.0.0.2
// See LICENSE for details.
#ifndef MGARD_TPP
#define MGARD_TPP

#include "mgard.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>

#include <zlib.h>

#include <fstream>
#include <numeric>
#include <bitset>

#include "mgard_compress.hpp"
#include "mgard_mesh.hpp"
#include "mgard_nuni.h"

#include "LinearQuantizer.hpp"

static void set_number_of_levels(const int nrow, const int ncol, int &nlevel) {
  // set the depth of levels in isotropic case
  if (nrow == 1) {
    nlevel = mgard::Dimensions2kPlus1<1>({ncol}).nlevel;
  } else if (nrow > 1) {
    nlevel = mgard::Dimensions2kPlus1<2>({nrow, ncol}).nlevel;
  }
}

namespace mgard {

template <typename Real>
unsigned char *refactor_qz(int nrow, int ncol, int nfib, const Real *u,
                           int &outsize, Real tol) {
  std::vector<Real> v(u, u + nrow * ncol * nfib), work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array
  std::vector<Real> coords_x(ncol), coords_y(nrow),
      coords_z(nfib); // coordinate arrays
  // dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

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

  mgard::quantize_2D_interleave(
      nrow, ncol * nfib, v.data(), qv, norm,
      tol); // rename this to quantize Linfty or smthng!!!!

  std::vector<unsigned char> out_data;

  mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

  outsize = out_data.size();
  unsigned char *buffer = (unsigned char *)malloc(outsize);
  std::copy(out_data.begin(), out_data.end(), buffer);
  return buffer;
}

template <typename Real>
unsigned char *
refactor_qz(int nrow, int ncol, int nfib, std::vector<Real> &coords_x,
            std::vector<Real> &coords_y, std::vector<Real> &coords_z,
            const Real *u, int &outsize, Real tol) {
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

  mgard::quantize_2D_interleave(
      nrow, ncol * nfib, v.data(), qv, norm,
      tol); // rename this to quantize Linfty or smthng!!!!

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
  const int size_ratio = sizeof(Real) / sizeof(int);
  std::vector<Real> coords_x(ncol), coords_y(nrow),
      coords_z(nfib); // coordinate arrays
  std::vector<int> out_data(nrow * ncol * nfib + size_ratio);
  std::vector<Real> work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array

  //      dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  //    //std::cout  <<"**** coord check : "  << coords_x[4] << "\n";

  const Dimensions2kPlus1<3> dims({nrow, ncol, nfib});
  const int l_target = dims.nlevel - 1;

  mgard::decompress_memory_z(data, data_len, out_data.data(),
                             out_data.size() *
                                 sizeof(int)); // decompress input buffer
  Real *v = (Real *)malloc(nrow * ncol * nfib * sizeof(Real));

  mgard::dequantize_2D_interleave(nrow, ncol * nfib, v, out_data);

  mgard_gen::recompose_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2],
                          dims.input[0], dims.input[1], dims.input[2], l_target,
                          v, work, work2d, coords_x, coords_y, coords_z);

  mgard_gen::postp_3D(dims.rnded[0], dims.rnded[1], dims.rnded[2],
                      dims.input[0], dims.input[1], dims.input[2], l_target, v,
                      work, coords_x, coords_y, coords_z);

  return v;
}

template <typename Real>
Real *recompose_udq(int nrow, int ncol, int nfib, std::vector<Real> &coords_x,
                    std::vector<Real> &coords_y, std::vector<Real> &coords_z,
                    unsigned char *data, int data_len) {
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

  mgard::dequantize_2D_interleave(nrow, ncol * nfib, v, out_data);

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
  const int size_ratio = sizeof(Real) / sizeof(int);
  std::vector<Real> coords_x(ncol), coords_y(nrow),
      coords_z(nfib); // coordinate arrays
  std::vector<int> out_data(nrow * ncol * nfib + size_ratio);
  std::vector<Real> work(nrow * ncol * nfib),
      work2d(nrow * ncol); // duplicate data and create work array

  //    Real s = 0; // Defaulting to L2 compression for a start.
  Real norm = 1; // defaulting to absolute s, may switch to relative

  //      dummy equispaced coordinates
  std::iota(std::begin(coords_x), std::end(coords_x), 0);
  std::iota(std::begin(coords_y), std::end(coords_y), 0);
  std::iota(std::begin(coords_z), std::end(coords_z), 0);

  //    //std::cout  <<"**** coord check : "  << coords_x[4] << "\n";

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
//}

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

  std::vector<Real> row_vec(ncol);
  std::vector<Real> v(u, u + ncol), work(ncol);

  Real norm = mgard_2d::mgard_common::max_norm(v);

  if (is_2kplus1(ncol)) // input is (2^p + 1)
  {
    // to be clean up.

    int nlevel;
    set_number_of_levels(1, ncol, nlevel);
    tol /= nlevel + 1;

    const int l_target = nlevel - 1;
    mgard::refactor_1D(ncol, l_target, v.data(), work, row_vec);

    work.clear();
    row_vec.clear();

    int size_ratio = sizeof(Real) / sizeof(int);
    std::vector<int> qv(ncol + size_ratio);

    mgard::quantize_2D_interleave(1, ncol, v.data(), qv, norm, tol);

    std::vector<unsigned char> out_data;

    return mgard::compress_memory_huffman(qv, out_data, outsize);
  } else {
    std::vector<Real> coords_x(ncol);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);

    const Dimensions2kPlus1<1> dims({ncol});
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

    mgard::quantize_2D_interleave(1, ncol, v.data(), qv, norm, tol);

    std::vector<unsigned char> out_data;

    return mgard::compress_memory_huffman(qv, out_data, outsize);
  }
}

template <typename Real>
unsigned char *refactor_qz_2D(int nrow, int ncol, const Real *u, int &outsize,
                              Real tol) {

  std::vector<Real> row_vec(ncol);
  std::vector<Real> col_vec(nrow);
  std::vector<Real> v(u, u + nrow * ncol), work(nrow * ncol);

  Real norm = mgard_2d::mgard_common::max_norm(v);

  if (is_2kplus1(nrow) && is_2kplus1(ncol)) // input is (2^q + 1) x (2^p + 1)
  {
    int nlevel;
    set_number_of_levels(nrow, ncol, nlevel);
    tol /= nlevel + 1;

    const int l_target = nlevel - 1;
    mgard::refactor(nrow, ncol, l_target, v.data(), work, row_vec, col_vec);
    work.clear();
    row_vec.clear();
    col_vec.clear();

    const int size_ratio = sizeof(Real) / sizeof(int);
    std::vector<int> qv(nrow * ncol + size_ratio);

    mgard::quantize_2D_interleave(nrow, ncol, v.data(), qv, norm, tol);

    std::vector<unsigned char> out_data;

    mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);
    outsize = out_data.size();
    unsigned char *buffer = (unsigned char *)malloc(outsize);
    std::copy(out_data.begin(), out_data.end(), buffer);
    return buffer;
  } else {

    std::vector<Real> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);

    const Dimensions2kPlus1<2> dims({nrow, ncol});
    tol /= dims.nlevel + 1;

    const int l_target = dims.nlevel - 1;

    mgard_2d::mgard_gen::prep_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                                 dims.input[1], l_target, v.data(), work,
                                 coords_x, coords_y, row_vec, col_vec);

    mgard_2d::mgard_gen::refactor_2D(
        dims.rnded[0], dims.rnded[1], dims.input[0], dims.input[1], l_target,
        v.data(), work, coords_x, coords_y, row_vec, col_vec);

    work.clear();
    col_vec.clear();
    row_vec.clear();

    const int size_ratio = sizeof(Real) / sizeof(int);
    std::vector<int> qv(nrow * ncol + size_ratio);

    // Uncomment the following. Otherwise the tolerence is divided twice.
    // Q. Liu 3/2/2020.
    //tol /= dims.nlevel + 1;
    mgard::quantize_2D_interleave(nrow, ncol, v.data(), qv, norm, tol);

    std::vector<unsigned char> out_data;

    mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);

    outsize = out_data.size();
    unsigned char *buffer = (unsigned char *)malloc(outsize);
    std::copy(out_data.begin(), out_data.end(), buffer);
    return buffer;
  }
}

template <typename Real>
unsigned char *refactor_qz_2D(int nrow, int ncol, std::vector<Real> &coords_x,
                              std::vector<Real> &coords_y, const Real *u,
                              int &outsize, Real tol) {

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
  mgard::quantize_2D_interleave(nrow, ncol, v.data(), qv, norm, tol);

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

  std::vector<Real> row_vec(ncol);
  std::vector<Real> col_vec(nrow);
  std::vector<Real> v(u, u + nrow * ncol), work(nrow * ncol);

  Real norm = mgard_2d::mgard_common::max_norm(v);

  if (is_2kplus1(nrow) && is_2kplus1(ncol)) // input is (2^q + 1) x (2^p + 1)
  {
    int nlevel;
    set_number_of_levels(nrow, ncol, nlevel);
    tol /= nlevel + 1;

    const int l_target = nlevel - 1;
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

    mgard_gen::quantize_2D(nrow, ncol, nrow, ncol, nlevel, v.data(), qv,
                           coords_x, coords_y, s, norm, tol);

    std::vector<unsigned char> out_data;

    mgard::compress_memory_z(qv.data(), sizeof(int) * qv.size(), out_data);
    outsize = out_data.size();
    unsigned char *buffer = (unsigned char *)malloc(outsize);
    std::copy(out_data.begin(), out_data.end(), buffer);
    return buffer;
  } else {

    std::vector<Real> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);

    const Dimensions2kPlus1<2> dims({nrow, ncol});
    tol /= dims.nlevel + 1;

    const int l_target = dims.nlevel - 1;

    mgard_2d::mgard_gen::prep_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                                 dims.input[1], l_target, v.data(), work,
                                 coords_x, coords_y, row_vec, col_vec);

    mgard_2d::mgard_gen::refactor_2D(
        dims.rnded[0], dims.rnded[1], dims.input[0], dims.input[1], l_target,
        v.data(), work, coords_x, coords_y, row_vec, col_vec);

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
  const int size_ratio = sizeof(Real) / sizeof(int);

  if (is_2kplus1(ncol)) // input is (2^p + 1)
  {
    // to be cleaned up.
    const Dimensions2kPlus1<1> dims({ncol});
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

    mgard::dequantize_2D_interleave(1, ncol, v, out_data);
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

    mgard::decompress_memory_huffman(data, data_len, out_data);

    Real *v = (Real *)malloc(ncol * sizeof(Real));

    mgard::dequantize_2D_interleave(1, ncol, v, out_data);

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
  const int size_ratio = sizeof(Real) / sizeof(int);

  if (is_2kplus1(ncol)) // input is (2^p + 1)
  {
    // to be cleaned up.
    const Dimensions2kPlus1<1> dims({ncol});
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

    mgard::dequantize_2D_interleave(1, ncol, v, out_data);
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

    mgard::dequantize_2D_interleave(1, ncol, v, out_data);

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
  const int size_ratio = sizeof(Real) / sizeof(int);

  if (is_2kplus1(nrow) && is_2kplus1(ncol)) // input is (2^q + 1) x (2^p + 1)
  {
    int ncol_new = ncol;
    int nrow_new = nrow;

    int nlevel_new;
    set_number_of_levels(nrow_new, ncol_new, nlevel_new);
    const int l_target = nlevel_new - 1;

    std::vector<int> out_data(nrow_new * ncol_new + size_ratio);

    mgard::decompress_memory_z(data, data_len, out_data.data(),
                               out_data.size() *
                                   sizeof(int)); // decompress input buffer

    Real *v = (Real *)malloc(nrow_new * ncol_new * sizeof(Real));

    mgard::dequantize_2D_interleave(nrow_new, ncol_new, v, out_data);
    out_data.clear();

    std::vector<Real> row_vec(ncol_new);
    std::vector<Real> col_vec(nrow_new);
    std::vector<Real> work(nrow_new * ncol_new);

    mgard::recompose(nrow_new, ncol_new, l_target, v, work, row_vec, col_vec);

    return v;

  } else {
    std::vector<Real> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);

    const Dimensions2kPlus1<2> dims({nrow, ncol});
    const int l_target = dims.nlevel - 1;

    std::vector<int> out_data(nrow * ncol + size_ratio);

    mgard::decompress_memory_z(data, data_len, out_data.data(),
                               out_data.size() *
                                   sizeof(int)); // decompress input buffer

    Real *v = (Real *)malloc(nrow * ncol * sizeof(Real));

    mgard::dequantize_2D_interleave(nrow, ncol, v, out_data);

    std::vector<Real> row_vec(ncol);
    std::vector<Real> col_vec(nrow);
    std::vector<Real> work(nrow * ncol);

    mgard_2d::mgard_gen::recompose_2D(
        dims.rnded[0], dims.rnded[1], dims.input[0], dims.input[1], l_target, v,
        work, coords_x, coords_y, row_vec, col_vec);

    mgard_2d::mgard_gen::postp_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                                  dims.input[1], l_target, v, work, coords_x,
                                  coords_y, row_vec, col_vec);

    return v;
  }
}

template <typename Real>
Real *recompose_udq_2D(int nrow, int ncol, std::vector<Real> &coords_x,
                       std::vector<Real> &coords_y, unsigned char *data,
                       int data_len) {
  const int size_ratio = sizeof(Real) / sizeof(int);

  const Dimensions2kPlus1<2> dims({nrow, ncol});
  const int l_target = dims.nlevel - 1;

  std::vector<int> out_data(nrow * ncol + size_ratio);

  mgard::decompress_memory_z(data, data_len, out_data.data(),
                             out_data.size() *
                                 sizeof(int)); // decompress input buffer

  Real *v = (Real *)malloc(nrow * ncol * sizeof(Real));

  mgard::dequantize_2D_interleave(nrow, ncol, v, out_data);

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
  const int size_ratio = sizeof(Real) / sizeof(int);

  if (is_2kplus1(nrow) && is_2kplus1(ncol)) // input is (2^q + 1) x (2^p + 1)
  {
    int ncol_new = ncol;
    int nrow_new = nrow;

    int nlevel_new;
    set_number_of_levels(nrow_new, ncol_new, nlevel_new);
    const int l_target = nlevel_new - 1;

    std::vector<Real> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);

    std::vector<int> out_data(nrow_new * ncol_new + size_ratio);

    mgard::decompress_memory_z(data, data_len, out_data.data(),
                               out_data.size() *
                                   sizeof(int)); // decompress input buffer

    Real *v = (Real *)malloc(nrow_new * ncol_new * sizeof(Real));

    //      mgard::dequantize_2D_interleave(nrow_new, ncol_new, v, out_data) ;
    mgard_gen::dequantize_2D(nrow, ncol, nrow, ncol, nlevel_new, v, out_data,
                             coords_x, coords_y, s);
    out_data.clear();

    std::vector<Real> row_vec(ncol_new);
    std::vector<Real> col_vec(nrow_new);
    std::vector<Real> work(nrow_new * ncol_new);

    mgard::recompose(nrow_new, ncol_new, l_target, v, work, row_vec, col_vec);

    return v;

  } else {
    std::vector<Real> coords_x(ncol), coords_y(nrow);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);
    std::iota(std::begin(coords_y), std::end(coords_y), 0);

    const Dimensions2kPlus1<2> dims({nrow, ncol});
    const int l_target = dims.nlevel - 1;

    std::vector<int> out_data(nrow * ncol + size_ratio);

    mgard::decompress_memory_z(data, data_len, out_data.data(),
                               out_data.size() * sizeof(int));

    Real *v = (Real *)malloc(nrow * ncol * sizeof(Real));

    mgard_gen::dequantize_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                             dims.input[1], dims.nlevel, v, out_data, coords_x,
                             coords_y, s);

    std::vector<Real> row_vec(ncol);
    std::vector<Real> col_vec(nrow);
    std::vector<Real> work(nrow * ncol);

    mgard_2d::mgard_gen::recompose_2D(
        dims.rnded[0], dims.rnded[1], dims.input[0], dims.input[1], l_target, v,
        work, coords_x, coords_y, row_vec, col_vec);

    mgard_2d::mgard_gen::postp_2D(dims.rnded[0], dims.rnded[1], dims.input[0],
                                  dims.input[1], l_target, v, work, coords_x,
                                  coords_y, row_vec, col_vec);

    return v;
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

  int stride = std::pow(2, l);
  Real temp1, temp2;
  Real fac = 0.5;
  // Mass matrix times nodal value-vec
  temp1 = v.front(); // save u(0) for later use
  v.front() = fac * (2.0 * temp1 + v.at(stride));
  for (auto it = v.begin() + stride; it < v.end() - stride; it += stride) {
    temp2 = *it;
    *it = fac * (temp1 + 4 * temp2 + *(it + stride));
    temp1 = temp2; // save u(n) for later use
  }
  v.back() = fac * (2 * v.back() + temp1);
}

template <typename Real>
void solve_tridiag_M(const int l, std::vector<Real> &v) {

  //  int my_level = nlevel - l;
  int stride = std::pow(2, l); // current stride

  Real am, bm;

  am = 2.0; // first element of upper diagonal U.

  bm = 1.0 / am;

  int nlevel = nlevel_from_size(v.size());
  int n = std::pow(2, nlevel - l) + 1;
  std::vector<Real> coeff(n);
  int counter = 1;
  coeff.front() = am;

  // forward sweep
  for (auto it = std::begin(v) + stride; it < std::end(v) - stride;
       it += stride) {
    *(it) -= *(it - stride) / am;

    am = 4.0 - bm;
    bm = 1.0 / am;

    coeff.at(counter) = am;
    ++counter;
  }
  am = 2.0 - bm; // a_n = 2 - b_(n-1)

  auto it = v.end() - stride - 1;
  v.back() -= (*it) * bm; // last element

  coeff.at(counter) = am;

  // backward sweep

  v.back() /= am;
  --counter;

  for (auto it = v.rbegin() + stride; it <= v.rend(); it += stride) {

    *(it) = (*(it) - *(it - stride)) / coeff.at(counter);
    --counter;
    bm = 4.0 - am; // maybe assign 1/am -> bm?
    am = 1.0 / bm;
  }
}

template <typename Real> void restriction(const int l, std::vector<Real> &v) {
  int stride = std::pow(2, l);
  int Pstride = stride / 2;

  // calculate the result of restrictionion
  auto it = v.begin() + Pstride;
  v.front() += 0.5 * (*it); // first element
  for (auto it = std::begin(v) + stride; it <= std::end(v) - stride;
       it += stride) {
    *(it) += 0.5 * (*(it - Pstride) + *(it + Pstride));
  }
  it = v.end() - Pstride - 1;
  v.back() += 0.5 * (*it); // last element
}

template <typename Real>
void interpolate_from_level_nMl(const int l, std::vector<Real> &v) {

  int stride = std::pow(2, l);
  int Pstride = stride / 2;

  for (auto it = std::begin(v) + stride; it < std::end(v); it += stride) {
    *(it - Pstride) = 0.5 * (*(it - stride) + *it);
  }
}

template <typename Real>
void print_level_2D(const int nrow, const int ncol, const int l, Real *v) {

  int stride = std::pow(2, l);

  for (int irow = 0; irow < nrow; irow += stride) {
    // std::cout  << "\n";
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      // std::cout  << v[get_index (ncol, irow, jcol)] << "\t";
    }
    // std::cout  << "\n";
  }
}

template <typename Real>
void write_level_2D(const int nrow, const int ncol, const int l, Real *v,
                    std::ofstream &outfile) {
  int stride = std::pow(2, l);
  //  int nrow = std::pow(2, nlevel_row) + 1;
  // int ncol = std::pow(2, nlevel_col) + 1;

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      outfile.write(reinterpret_cast<char *>(&v[get_index(ncol, irow, jcol)]),
                    sizeof(Real));
    }
  }
}

template <typename Real>
void write_level_2D_exc(const int nrow, const int ncol, const int l, Real *v,
                        std::ofstream &outfile) {
  // Write P_l\P_{l-1}

  int stride = std::pow(2, l);
  int Cstride = stride * 2;

  int row_counter = 0;

  for (int irow = 0; irow < nrow; irow += stride) {
    if (row_counter % 2 == 0) {
      for (int jcol = Cstride; jcol < ncol; jcol += Cstride) {
        outfile.write(
            reinterpret_cast<char *>(&v[get_index(ncol, irow, jcol - stride)]),
            sizeof(Real));
      }
    } else {
      for (int jcol = 0; jcol < ncol; jcol += stride) {
        outfile.write(reinterpret_cast<char *>(&v[get_index(ncol, irow, jcol)]),
                      sizeof(Real));
      }
    }
    ++row_counter;
  }
}

template <typename Real> void pi_lminus1(const int l, std::vector<Real> &v0) {
  int nlevel = nlevel_from_size(v0.size());
  int my_level = nlevel - l;
  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  if (my_level != 0) {
    for (auto it0 = v0.begin() + Cstride; it0 < v0.end(); it0 += Cstride) {
      *(it0 - stride) -= 0.5 * (*it0 + *(it0 - Cstride));
    }
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
void pi_Ql(const int nrow, const int ncol, const int l, Real *v,
           std::vector<Real> &row_vec, std::vector<Real> &col_vec) {
  // Restrict data to coarser level

  int stride = std::pow(2, l); // current stride
  //  int Pstride = stride/2; //finer stride
  int Cstride = stride * 2; // coarser stride

  //  std::vector<Real> row_vec(ncol), col_vec(nrow)   ;

  for (int irow = 0; irow < nrow;
       irow += Cstride) // Do the rows existing  in the coarser level
  {
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = v[get_index(ncol, irow, jcol)];
    }

    pi_lminus1(l, row_vec);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      v[get_index(ncol, irow, jcol)] = row_vec[jcol];
    }
  }

  if (nrow > 1) {
    for (int jcol = 0; jcol < ncol;
         jcol += Cstride) // Do the columns existing  in the coarser level
    {
      for (int irow = 0; irow < nrow; ++irow) {
        col_vec[irow] = v[get_index(ncol, irow, jcol)];
      }

      pi_lminus1(l, col_vec);

      for (int irow = 0; irow < nrow; ++irow) {
        v[get_index(ncol, irow, jcol)] = col_vec[irow];
      }
    }

    // Now the new-new stuff

    for (int irow = Cstride; irow <= nrow - 1 - Cstride; irow += 2 * Cstride) {
      for (int jcol = Cstride; jcol <= ncol - 1 - Cstride;
           jcol += 2 * Cstride) {
        v[get_index(ncol, irow - stride, jcol - stride)] -=
            0.25 * (v[get_index(ncol, irow - Cstride, jcol - Cstride)] +
                    v[get_index(ncol, irow - Cstride, jcol)] +
                    v[get_index(ncol, irow, jcol)] +
                    v[get_index(ncol, irow, jcol - Cstride)]);

        v[get_index(ncol, irow - stride, jcol + stride)] -=
            0.25 * (v[get_index(ncol, irow - Cstride, jcol)] +
                    v[get_index(ncol, irow - Cstride, jcol + Cstride)] +
                    v[get_index(ncol, irow, jcol + Cstride)] +
                    v[get_index(ncol, irow, jcol)]);

        v[get_index(ncol, irow + stride, jcol + stride)] -=
            0.25 * (v[get_index(ncol, irow, jcol)] +
                    v[get_index(ncol, irow, jcol + Cstride)] +
                    v[get_index(ncol, irow + Cstride, jcol + Cstride)] +
                    v[get_index(ncol, irow + Cstride, jcol)]);

        v[get_index(ncol, irow + stride, jcol - stride)] -=
            0.25 * (v[get_index(ncol, irow, jcol - Cstride)] +
                    v[get_index(ncol, irow, jcol)] +
                    v[get_index(ncol, irow + Cstride, jcol)] +
                    v[get_index(ncol, irow + Cstride, jcol - Cstride)]);
      }
    }
  }
}

template <typename Real>
void assign_num_level(const int nrow, const int ncol, const int l, Real *v,
                      Real num) {
  // set the value of nodal values at level l to number num

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      v[get_index(ncol, irow, jcol)] = num;
    }
  }
}

template <typename Real>
void copy_level(const int nrow, const int ncol, const int l, Real *v,
                std::vector<Real> &work) {

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      work[get_index(ncol, irow, jcol)] = v[get_index(ncol, irow, jcol)];
    }
  }
}

template <typename Real>
void add_level(const int nrow, const int ncol, const int l, Real *v,
               Real *work) {
  // v += work at level l

  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      v[get_index(ncol, irow, jcol)] += work[get_index(ncol, irow, jcol)];
    }
  }
}

template <typename Real>
void subtract_level(const int nrow, const int ncol, const int l, Real *v,
                    Real *work) {
  // v += work at level l
  int stride = std::pow(2, l); // current stride

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      v[get_index(ncol, irow, jcol)] -= work[get_index(ncol, irow, jcol)];
    }
  }
}

template <typename Real>
void compute_correction_loadv(const int l, std::vector<Real> &v) {
  int stride = std::pow(2, l); // current stride
  int Pstride = stride / 2;    // finer stride

  auto it = v.begin() + Pstride;
  v.front() += 0.25 * (*it); // first element
  for (auto it = std::begin(v) + stride; it <= std::end(v) - stride;
       it += stride) {
    *(it) += 0.25 * (*(it - Pstride) + *(it + Pstride));
  }
  it = v.end() - Pstride - 1;
  v.back() += 0.25 * (*it); // last element
}

template <typename Real>
void qwrite_level_2D(const int nrow, const int ncol, const int nlevel,
                     const int l, Real *v, const Real tol,
                     const std::string outfile) {

  int stride = std::pow(2, l);

  Real norm = 0;

  for (int irow = 0; irow < nrow; irow += stride) {
    for (int jcol = 0; jcol < ncol; jcol += stride) {
      Real ntest = std::abs(v[get_index(ncol, irow, jcol)]);
      if (ntest > norm)
        norm = ntest;
    }
  }

  const mgard::LinearQuantizer<Real, int> quantizer(norm * tol / (nlevel + 1));

  gzFile out_file = gzopen(outfile.c_str(), "w9b");
  gzwrite(out_file, &quantizer.quantum, sizeof(Real));

  int prune_count = 0;

  for (int l = 0; l <= nlevel; l++) {
    int stride = std::pow(2, l);
    int Cstride = stride * 2;
    int row_counter = 0;

    for (int irow = 0; irow < nrow; irow += stride) {
      if (row_counter % 2 == 0 && l != nlevel) {
        for (int jcol = Cstride; jcol < ncol; jcol += Cstride) {
          const int n = quantizer(v[get_index(ncol, irow, jcol - stride)]);
          if (n == 0)
            ++prune_count;
          gzwrite(out_file, &n, sizeof(int));
        }
      } else {
        for (int jcol = 0; jcol < ncol; jcol += stride) {
          const int n = quantizer(v[get_index(ncol, irow, jcol)]);
          if (n == 0)
            ++prune_count;
          gzwrite(out_file, &n, sizeof(int));
        }
      }
      ++row_counter;
    }
  }

  // std::cout  << "Pruned : " << prune_count << " Reduction : "
  //            << (Real)nrow * ncol / (nrow * ncol - prune_count) << "\n";
  gzclose(out_file);
}

template <typename Real>
void quantize_2D_interleave(const int nrow, const int ncol, Real *v,
                            std::vector<int> &work, const Real norm,
                            const Real tol) {
  const int size_ratio = sizeof(Real) / sizeof(int);

  //    Real quantizer = 2.0*norm * tol;
  const mgard::LinearQuantizer<Real, int> quantizer(norm * tol);
  ////std::cout  << "Quantization factor: " << quantizer << "\n";
  std::memcpy(work.data(), &quantizer.quantum, sizeof(Real));

  int prune_count = 0;

  for (int index = 0; index < ncol * nrow; ++index) {
    const int n = quantizer(v[index]);
    work[index + size_ratio] = n;
    if (n == 0)
      ++prune_count;
  }

  ////std::cout  << "Pruned : " << prune_count << " Reduction : "
  //          << (Real)2 * nrow * ncol / (nrow * ncol - prune_count) << "\n";
}

template <typename Real>
void dequantize_2D_interleave(const int nrow, const int ncol, Real *v,
                              const std::vector<int> &work) {
  const int size_ratio = sizeof(Real) / sizeof(int);

  Real quantum;
  std::memcpy(&quantum, work.data(), sizeof(Real));
  const mgard::LinearDequantizer<int, Real> quantizer(quantum);

  for (int index = 0; index < nrow * ncol; ++index) {
    v[index] = quantizer(work[index + size_ratio]);
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

template <typename Real>
void qread_level_2D(const int nrow, const int ncol, const int nlevel, Real *v,
                    std::string infile) {
  int buff_size = 128 * 1024;
  unsigned char unzip_buffer[buff_size];
  int int_buffer[buff_size / sizeof(int)];
  unsigned int unzipped_bytes, total_bytes = 0;

  Real quantum;
  gzFile in_file_z = gzopen(infile.c_str(), "r");
  // std::cout  << in_file_z << "\n";

  unzipped_bytes = gzread(in_file_z, unzip_buffer,
                          sizeof(Real)); // read the quantization constant
  std::memcpy(&quantum, &unzip_buffer, unzipped_bytes);
  const mgard::LinearDequantizer<int, Real> dequantizer(quantum);

  int last = 0;
  while (true) {
    unzipped_bytes = gzread(in_file_z, unzip_buffer, buff_size);
    // std::cout  << unzipped_bytes << "\n";
    if (unzipped_bytes > 0) {
      total_bytes += unzipped_bytes;
      int num_int = unzipped_bytes / sizeof(int);

      std::memcpy(&int_buffer, &unzip_buffer, unzipped_bytes);
      for (int i = 0; i < num_int; ++i) {
        v[last] = dequantizer(int_buffer[i]);
        ++last;
      }
    } else {
      break;
    }
  }

  gzclose(in_file_z);
}

//       unzippedBytes = gzread(inFileZ, unzipBuffer, buff_size);
//       //std::cout  << "Read: "<< unzippedBytes <<"\n";
//       std::memcpy(&v[irow][0], &unzipBuffer, unzippedBytes);
//     }

//   gzclose(inFileZ);
// }
// Gary New
template <typename Real>
void refactor_1D(const int ncol, const int l_target, Real *v,
                 std::vector<Real> &work, std::vector<Real> &row_vec) {
  for (int l = 0; l < l_target; ++l) {

    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride
#if 1
    pi_Ql(ncol, l, v, row_vec); // rename!. v@l has I-\Pi_l Q_l+1 u
#endif
    copy_level(1, ncol, l, v,
               work); // copy the nodal values of v on l  to matrix work

    assign_num_level(1, ncol, l + 1, work.data(), static_cast<Real>(0.0));

    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[jcol];
    }

    mass_matrix_multiply(l, row_vec);

    restriction(l + 1, row_vec);

    solve_tridiag_M(l + 1, row_vec);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[jcol] = row_vec[jcol];
    }

    add_level(1, ncol, l + 1, v, work.data()); // Qu_l = \Pi_l Q_{l+1}u + z_l
  }
}

template <typename Real>
void refactor(const int nrow, const int ncol, const int l_target, Real *v,
              std::vector<Real> &work, std::vector<Real> &row_vec,
              std::vector<Real> &col_vec) {
  // refactor
  //  //std::cout  << "refactoring" << "\n";

  for (int l = 0; l < l_target; ++l) {

    int stride = std::pow(2, l); // current stride
    int Cstride = stride * 2;    // coarser stride

    pi_Ql(nrow, ncol, l, v, row_vec,
          col_vec); // rename!. v@l has I-\Pi_l Q_l+1 u
    copy_level(nrow, ncol, l, v,
               work); // copy the nodal values of v on l  to matrix work

    assign_num_level(nrow, ncol, l + 1, work.data(), static_cast<Real>(0.0));

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

    add_level(nrow, ncol, l + 1, v,
              work.data()); // Qu_l = \Pi_l Q_{l+1}u + z_l
  }
}

// Gary New
template <typename Real>
void recompose_1D(const int ncol, const int l_target, Real *v,
                  std::vector<Real> &work, std::vector<Real> &row_vec) {

  // recompose

  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;

    copy_level(1, ncol, l - 1, v, work); // copy the nodal values of cl
                                         // on l-1 (finer level)  to
                                         // matrix work
    // zero out nodes of l on cl
    assign_num_level(1, ncol, l, work.data(), static_cast<Real>(0.0));

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

    subtract_level(1, ncol, l, work.data(), v); // do -(Qu - zl)

    // row-sweep
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[jcol];
    }

    interpolate_from_level_nMl(l, row_vec);

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[jcol] = row_vec[jcol];
    }

    // zero out nodes of l on cl
    assign_num_level(1, ncol, l, v, static_cast<Real>(0.0));
    subtract_level(1, ncol, l - 1, v, work.data());
  }
}

template <typename Real>
void recompose(const int nrow, const int ncol, const int l_target, Real *v,
               std::vector<Real> &work, std::vector<Real> &row_vec,
               std::vector<Real> &col_vec) {

  // recompose

  for (int l = l_target; l > 0; --l) {

    int stride = std::pow(2, l); // current stride
    int Pstride = stride / 2;

    copy_level(nrow, ncol, l - 1, v, work); // copy the nodal values of cl
                                            // on l-1 (finer level)  to
                                            // matrix work
    // zero out nodes of l on cl
    assign_num_level(nrow, ncol, l, work.data(), static_cast<Real>(0.0));

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
    subtract_level(nrow, ncol, l, work.data(), v); // do -(Qu - zl)

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
    assign_num_level(nrow, ncol, l, v, static_cast<Real>(0.0));
    subtract_level(nrow, ncol, l - 1, v, work.data());
  }
}

template <typename Real>
Real interp_2d(Real q11, Real q12, Real q21, Real q22, Real x1, Real x2,
               Real y1, Real y2, Real x, Real y) {
  Real x2x1, y2y1, x2x, y2y, yy1, xx1;
  x2x1 = x2 - x1;
  y2y1 = y2 - y1;
  x2x = x2 - x;
  y2y = y2 - y;
  yy1 = y - y1;
  xx1 = x - x1;
  return 1.0 / (x2x1 * y2y1) *
         (q11 * x2x * y2y + q21 * xx1 * y2y + q12 * x2x * yy1 +
          q22 * xx1 * yy1);
}

template <typename Real>
Real interp_0d(const Real x1, const Real x2, const Real y1, const Real y2,
               const Real x) {
  // do a linear interpolation between (x1, y1) and (x2, y2)
  return (((x2 - x) * y1 + (x - x1) * y2) / (x2 - x1));
}

template <typename Real>
void resample_1d(const Real *inbuf, Real *outbuf, const int ncol,
                 const int ncol_new) {
  Real hx_o = 1.0 / Real(ncol - 1);
  Real hx = 1.0 / Real(ncol_new - 1); // x-spacing
  Real hx_ratio = (hx_o / hx);        // ratio of x-spacing resampled/orig

  for (int icol = 0; icol < ncol_new - 1; ++icol) {
    int i_left = floor(icol / hx_ratio);
    int i_right = i_left + 1;

    Real x1 = Real(i_left) * hx_o;
    Real x2 = Real(i_right) * hx_o;

    Real y1 = inbuf[i_left];
    Real y2 = inbuf[i_right];
    Real x = Real(icol) * hx;
    //      //std::cout  <<  x1 << "\t" << x2 << "\t" << x << "\t"<< "\n";
    // //std::cout  <<  y1 << "\t" << y2 << "\t" << "\n";

    outbuf[icol] = interp_0d(x1, x2, y1, y2, x);
    //      std:: cout << mgard_interp_0d( x1,  x2,  y1,  y2,  x) << "\n";
  }

  outbuf[ncol_new - 1] = inbuf[ncol - 1];
}

template <typename Real>
void resample_1d_inv2(const Real *inbuf, Real *outbuf, const int ncol,
                      const int ncol_new) {
  Real hx_o = 1.0 / Real(ncol - 1);
  Real hx = 1.0 / Real(ncol_new - 1); // x-spacing
  Real hx_ratio = (hx_o / hx);        // ratio of x-spacing resampled/orig

  for (int icol = 0; icol < ncol_new - 1; ++icol) {
    int i_left = floor(icol / hx_ratio);
    int i_right = i_left + 1;

    Real x1 = Real(i_left) * hx_o;
    Real x2 = Real(i_right) * hx_o;

    Real y1 = inbuf[i_left];
    Real y2 = inbuf[i_right];
    Real x = Real(icol) * hx;

    Real d1 = std::pow(x1 - x, 4.0);
    Real d2 = std::pow(x2 - x, 4.0);

    if (d1 == 0) {
      outbuf[icol] = y1;
    } else if (d2 == 0) {
      outbuf[icol] = y2;
    } else {
      Real dsum = 1.0 / d1 + 1.0 / d2;
      outbuf[icol] = (y1 / d1 + y2 / d2) / dsum;
    }
  }

  outbuf[ncol_new - 1] = inbuf[ncol - 1];
}

template <typename Real>
void resample_2d(const Real *inbuf, Real *outbuf, const int nrow,
                 const int ncol, const int nrow_new, const int ncol_new) {
  Real hx_o = 1.0 / Real(ncol - 1);
  Real hx = 1.0 / Real(ncol_new - 1); // x-spacing
  Real hx_ratio = (hx_o / hx);        // ratio of x-spacing resampled/orig

  Real hy_o = 1.0 / Real(nrow - 1);
  Real hy = 1.0 / Real(nrow_new - 1); // x-spacing
  Real hy_ratio = (hy_o / hy);        // ratio of x-spacing resampled/orig

  for (int irow = 0; irow < nrow_new - 1; ++irow) {
    int i_bot = floor(irow / hy_ratio);
    int i_top = i_bot + 1;

    Real y = Real(irow) * hy;
    Real y1 = Real(i_bot) * hy_o;
    Real y2 = Real(i_top) * hy_o;

    for (int jcol = 0; jcol < ncol_new - 1; ++jcol) {
      int j_left = floor(jcol / hx_ratio);
      int j_right = j_left + 1;

      Real x = Real(jcol) * hx;
      Real x1 = Real(j_left) * hx_o;
      Real x2 = Real(j_right) * hx_o;

      Real q11 = inbuf[get_index(ncol, i_bot, j_left)];
      Real q12 = inbuf[get_index(ncol, i_top, j_left)];
      Real q21 = inbuf[get_index(ncol, i_bot, j_right)];
      Real q22 = inbuf[get_index(ncol, i_top, j_right)];

      outbuf[get_index(ncol_new, irow, jcol)] =
          interp_2d(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
    }

    // last column
    Real q1 = inbuf[get_index(ncol, i_bot, ncol - 1)];
    Real q2 = inbuf[get_index(ncol, i_top, ncol - 1)];
    outbuf[get_index(ncol_new, irow, ncol_new - 1)] =
        interp_0d(y1, y2, q1, q2, y);
  }

  // last-row
  resample_1d(&inbuf[get_index(ncol, nrow - 1, 0)],
              &outbuf[get_index(ncol_new, nrow_new - 1, 0)], ncol, ncol_new);
}

template <typename Real>
void resample_2d_inv2(const Real *inbuf, Real *outbuf, const int nrow,
                      const int ncol, const int nrow_new, const int ncol_new) {
  Real hx_o = 1.0 / Real(ncol - 1);
  Real hx = 1.0 / Real(ncol_new - 1); // x-spacing
  Real hx_ratio = (hx_o / hx);        // ratio of x-spacing resampled/orig

  Real hy_o = 1.0 / Real(nrow - 1);
  Real hy = 1.0 / Real(nrow_new - 1); // x-spacing
  Real hy_ratio = (hy_o / hy);        // ratio of x-spacing resampled/orig

  for (int irow = 0; irow < nrow_new - 1; ++irow) {
    int i_bot = floor(irow / hy_ratio);
    int i_top = i_bot + 1;

    Real y = Real(irow) * hy;
    Real y1 = Real(i_bot) * hy_o;
    Real y2 = Real(i_top) * hy_o;

    for (int jcol = 0; jcol < ncol_new - 1; ++jcol) {
      int j_left = floor(jcol / hx_ratio);
      int j_right = j_left + 1;

      Real x = Real(jcol) * hx;
      Real x1 = Real(j_left) * hx_o;
      Real x2 = Real(j_right) * hx_o;

      Real q11 = inbuf[get_index(ncol, i_bot, j_left)];
      Real q12 = inbuf[get_index(ncol, i_top, j_left)];
      Real q21 = inbuf[get_index(ncol, i_bot, j_right)];
      Real q22 = inbuf[get_index(ncol, i_top, j_right)];

      Real d11 = (std::pow(x1 - x, 2.0) + std::pow(y1 - y, 2.0));
      Real d12 = (std::pow(x1 - x, 2.0) + std::pow(y2 - y, 2.0));
      Real d21 = (std::pow(x2 - x, 2.0) + std::pow(y1 - y, 2.0));
      Real d22 = (std::pow(x2 - x, 2.0) + std::pow(y2 - y, 2.0));

      if (d11 == 0) {
        outbuf[get_index(ncol_new, irow, jcol)] = q11;
      } else if (d12 == 0) {
        outbuf[get_index(ncol_new, irow, jcol)] = q12;
      } else if (d21 == 0) {
        outbuf[get_index(ncol_new, irow, jcol)] = q21;
      } else if (d22 == 0) {
        outbuf[get_index(ncol_new, irow, jcol)] = q22;
      } else {
        d11 = std::pow(d11, 1.5);
        d12 = std::pow(d12, 1.5);
        d21 = std::pow(d21, 1.5);
        d22 = std::pow(d22, 1.5);

        Real dsum = 1.0 / (d11) + 1.0 / (d12) + 1.0 / (d21) + 1.0 / (d22);
        ////std::cout  <<  (q11/d11 + q12/d12 + q21/d21 + q22/d22)/dsum << "\n";
        //              //std::cout  <<  dsum << "\n";

        outbuf[get_index(ncol_new, irow, jcol)] =
            (q11 / d11 + q12 / d12 + q21 / d21 + q22 / d22) / dsum;
      }
    }

    // last column
    Real q1 = inbuf[get_index(ncol, i_bot, ncol - 1)];
    Real q2 = inbuf[get_index(ncol, i_top, ncol - 1)];

    Real d1 = std::pow(y1 - y, 4.0);
    Real d2 = std::pow(y2 - y, 4.0);

    if (d1 == 0) {
      outbuf[get_index(ncol_new, irow, ncol_new - 1)] = q1;
    } else if (d2 == 0) {
      outbuf[get_index(ncol_new, irow, ncol_new - 1)] = q2;
    } else {
      Real dsum = 1.0 / d1 + 1.0 / d2;
      outbuf[get_index(ncol_new, irow, ncol_new - 1)] =
          (q1 / d1 + q2 / d2) / dsum;
    }
  }

  // last-row
  resample_1d_inv2(&inbuf[get_index(ncol, nrow - 1, 0)],
                   &outbuf[get_index(ncol_new, nrow_new - 1, 0)], ncol,
                   ncol_new);
}

} // end namespace mgard

#endif
