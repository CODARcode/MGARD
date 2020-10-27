// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney, Qing Liu
// Corresponding Author: Ben Whitney, Qing Liu
//
// version: 0.1.0
// See LICENSE for details.
#ifndef MGARD_TPP
#define MGARD_TPP

#include "mgard.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <zlib.h>

#include <bitset>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <iostream>
#ifdef MGARD_TIMING
#include <chrono>
#endif

#include "mgard_compress.hpp"
#include "mgard_mesh.hpp"
#include "mgard_nuni.h"

#include "LinearQuantizer.hpp"
#include "TensorMassMatrix.hpp"
#include "TensorProlongation.hpp"
#include "TensorRestriction.hpp"
#include "shuffle.hpp"

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
  quantize_interleave(hierarchy, v.data(), qv.data(), norm, tol);

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

  dequantize_interleave(hierarchy, v, out_data.data());

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
#ifdef MGARD_TIMING
    auto start = std::chrono::high_resolution_clock::now();
#endif
    mgard::refactor_1D(ncol, l_target, v.data(), work, row_vec);

    work.clear();
    row_vec.clear();

    int size_ratio = sizeof(Real) / sizeof(int);
    std::vector<int> qv(ncol + size_ratio);

    quantize_interleave(hierarchy, v.data(), qv.data(), norm, tol);

#ifdef MGARD_TIMING
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Refactor Time = " << (double)duration.count() / 1000000
              << "\n";
#endif
    std::vector<unsigned char> out_data;

    return mgard::compress_memory_huffman(qv, out_data, outsize);
  } else {
    std::vector<Real> coords_x(ncol);

    std::iota(std::begin(coords_x), std::end(coords_x), 0);

    tol /= dims.nlevel + 1;

    const int l_target = dims.nlevel - 1;
#ifdef MGARD_TIMING
    auto start = std::chrono::high_resolution_clock::now();
#endif
    mgard_2d::mgard_gen::prep_1D(dims.rnded[0], dims.input[0], l_target,
                                 v.data(), work, coords_x, row_vec);

    mgard_2d::mgard_gen::refactor_1D(dims.rnded[0], dims.input[0], l_target,
                                     v.data(), work, coords_x, row_vec);

    work.clear();
    row_vec.clear();

    const int size_ratio = sizeof(Real) / sizeof(int);
    std::vector<int> qv(ncol + size_ratio);

    quantize_interleave(hierarchy, v.data(), qv.data(), norm, tol);
#ifdef MGARD_TIMING
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "Refactor Time = " << duration.count() << "\n";
#endif
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

    quantize_interleave(hierarchy, v.data(), qv.data(), norm, tol);

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
  quantize_interleave(hierarchy, v.data(), qv.data(), norm, tol);

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

    dequantize_interleave(hierarchy, v, out_data.data());
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

    dequantize_interleave(hierarchy, v, out_data.data());

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

    mgard::dequantize_interleave(hierarchy, v, out_data.data());
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

    mgard::dequantize_interleave(hierarchy, v, out_data.data());

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

    dequantize_interleave(hierarchy, v, out_data.data());
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

  dequantize_interleave(hierarchy, v, out_data.data());

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

template <std::size_t N, typename Real>
void mass_matrix_multiply(const TensorMeshHierarchy<N, Real> &hierarchy,
                          const int index_difference,
                          const std::size_t dimension, Real *const v) {
  // TODO: This will be unnecessary once we change `index_difference`/`l` to be
  // an index instead of a difference of indices.
  const std::size_t l = hierarchy.l(index_difference);
  const std::size_t stride = hierarchy.stride(l, dimension);
  const std::size_t n = hierarchy.meshes.at(l).shape.at(dimension);
  // The entries of the mass matrix are scaled by `h / 6`. We assume that the
  // cells of the finest level have width `6`, so that the cells on this level
  // have width `6 * stride`. `factor` is then `h / 6`.
  const Real factor = stride;
  Real left, middle, right;
  Real *p = v;
  middle = *p;
  right = *(p + stride);
  *p = factor * (2 * middle + right);
  p += stride;
  for (std::size_t i = 1; i + 1 < n; ++i) {
    left = middle;
    middle = right;
    right = *(p + stride);
    *p = factor * (left + 4 * middle + right);
    p += stride;
  }
  left = middle;
  middle = right;
  *p = factor * (left + 2 * middle);
}

template <std::size_t N, typename Real>
void solve_tridiag_M(const TensorMeshHierarchy<N, Real> &hierarchy,
                     const int index_difference, const std::size_t dimension,
                     Real *const v) {
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
  const std::size_t l = hierarchy.l(index_difference);
  const std::size_t stride = hierarchy.stride(l, dimension);
  const std::size_t n = hierarchy.meshes.at(l).shape.at(dimension);
  // See the note in `mass_matrix_multiply`.
  const Real factor = stride;

  // The system size is `n`, but we don't need to store the final divisor. In
  // the notation above, this vector will be [b_1, …, b_{n - 1}]` (after the
  // modification in the forward sweep).
  Real *p = v;
  std::vector<Real> divisors(n - 1);
  typename std::vector<Real>::iterator d = divisors.begin();
  // This is `b_{i - 1}`.
  Real previous_divisor = *d = 2;
  // This is `d_{i - 1}`.
  Real previous_entry = *p;
  p += stride;
  // Forward sweep (except for last entry).
  for (std::size_t i = 1; i + 1 < n; ++i) {
    // The numerator is really `a`.
    const Real w = 1 / previous_divisor;
    // The last term is really `w * c`.
    previous_divisor = *++d = 4 - w;
    previous_entry = *p -= w * previous_entry;
    p += stride;
  }

  // Forward sweep (last entry) and start of backward sweep (first entry).
  {
    // The numerator is really `a`.
    const Real w = 1 / previous_divisor;
    // Don't need to update `previous_divisor` and `previous_entry` as we won't
    // be using them.
    Real &entry = *p;
    entry -= w * previous_entry;
    // Don't need need to write to `*d` or increment `d` as we're using the
    // divisor immediately.
    // The last term is really `w * c`.
    previous_entry = entry /= 2 - w;
    p -= stride;
  }

  // Backward sweep (remaining entries).
  for (std::size_t i = 1; i < n; ++i) {
    Real &entry = *p;
    // The subtrahend is really `c * previous_entry`.
    entry -= previous_entry;
    previous_entry = entry /= *d--;
    *(p + stride) /= factor;
    p -= stride;
  }
  *(p + stride) /= factor;
}

template <std::size_t N, typename Real>
void restriction(const TensorMeshHierarchy<N, Real> &hierarchy,
                 const int index_difference, const std::size_t dimension,
                 Real *const v) {
  if (!index_difference) {
    throw std::domain_error("cannot restrict from the finest level");
  }
  const std::size_t l = hierarchy.l(index_difference);
  // The capitalization here comes from the convention that lowercase names
  // correspond to coarser meshes and uppercase names correspond to finer
  // meshes. Confusingly, `STRIDE` is actually smaller than `stride`.
  const std::size_t stride = hierarchy.stride(l, dimension);
  const std::size_t STRIDE = hierarchy.stride(l + 1, dimension);
  const std::size_t n = hierarchy.meshes.at(l).shape.at(dimension);

  Real left, right;
  Real *p = v;
  right = *(p + STRIDE);
  *p += 0.5 * right;
  p += stride;
  for (std::size_t i = 1; i + 1 < n; ++i) {
    left = right;
    right = *(p + STRIDE);
    *p += 0.5 * (left + right);
    p += stride;
  }
  left = right;
  *p += 0.5 * left;
}

template <std::size_t N, typename Real>
void interpolate_old_to_new_and_overwrite(
    const TensorMeshHierarchy<N, Real> &hierarchy, const int index_difference,
    const std::size_t dimension, Real *const v) {
  if (!index_difference) {
    throw std::domain_error("cannot interpolate from the finest level");
  }
  const std::size_t l = hierarchy.l(index_difference);
  // See note on capitalization in `restriction`.
  const std::size_t stride = hierarchy.stride(l, dimension);
  const std::size_t STRIDE = hierarchy.stride(l + 1, dimension);
  const std::size_t n = hierarchy.meshes.at(l).shape.at(dimension);

  Real *q = v;
  Real left = *q;
  q += stride;
  Real *p = v + STRIDE;
  for (std::size_t i = 1; i < n; ++i) {
    const Real right = *q;
    *p = 0.5 * (left + right);
    left = right;
    q += stride;
    // Might be better to think of this as advancing by `STRIDE` twice.
    p += stride;
  }
}

template <std::size_t N, typename Real>
void interpolate_old_to_new_and_subtract(
    const TensorMeshHierarchy<N, Real> &hierarchy, const int index_difference,
    const std::size_t dimension, Real *const v) {
  const std::size_t l = hierarchy.l(index_difference);
  if (!l) {
    throw std::domain_error("cannot interpolate from the coarsest level");
  }
  const std::size_t STRIDE = hierarchy.stride(l, dimension);
  const std::size_t stride = hierarchy.stride(l - 1, dimension);
  const std::size_t n = hierarchy.meshes.at(l - 1).shape.at(dimension);

  Real *q = v;
  Real left = *q;
  q += stride;
  Real *p = v + STRIDE;
  for (std::size_t i = 1; i < n; ++i) {
    const Real right = *q;
    *p -= 0.5 * (left + right);
    left = right;
    q += stride;
    p += stride;
  }
}

template <std::size_t N, typename Real>
void interpolate_old_to_new_and_subtract(
    const TensorMeshHierarchy<N, Real> &hierarchy, const int index_difference,
    Real *const v) {
  const std::size_t l = hierarchy.l(index_difference);
  if (!l) {
    throw std::domain_error("cannot interpolate from the coarsest level");
  }
  const std::size_t STRIDE = stride_from_index_difference(index_difference);
  const std::size_t stride = stride_from_index_difference(index_difference + 1);

  const std::array<std::size_t, N> &shape = hierarchy.meshes.back().shape;
  const Dimensions2kPlus1<N> dims(shape);
  if (!dims.is_2kplus1()) {
    throw std::domain_error("dimensions must all be of the form `2^k + 1`");
  }
  // It'd be nice to somehow use `LevelValues` here. We might like to write
  // something like `SituatedCoefficientRange` so we get the multiindices along
  // with the values. For now we'll iterate over the multiindices and fetch the
  // values ourselves.
  // Now that `LevelValues` has been replaced by `TensorLevelValues`, it is now
  // possible to do the above. Holding off as I expect to delete this function.
  // Still holding off now as `TensorLevelValues` itself is being replaced.
  const MultiindexRectangle<N> rectangle(shape);

  // We're splitting the grid into 'boxes' with 'lower left' corner `alpha` and
  // side length `stride` (so containing (in each dimension) `stride + 1` points
  // – the second parameter to the `MultiindexRectangle` constructor is an array
  // of sizes (rather than 'lengths'), so `stride + 1` is what we use) (except
  // for the boxes at the far boundaries, which will be smaller). We iterate
  // over the corners of each box (`beta`) to calculate the interpolant at the
  // interior points (`BETA`). We only want to adjust the values on the 'new'
  // nodes, so we need to skip any `BETA` that is also a `beta`. Additionally,
  // some `BETA`s straddle multiple boxes, and we must take care to adjust the
  // values at those multiindices only once. We do this dealing with `BETA` at
  // the 'minimum' (elementwise – think 'lower left') `alpha`.
  for (const std::array<std::size_t, N> alpha : rectangle.indices(stride)) {
    // Shape to use in iterating over `beta`s. In particular, we need to include
    // the 'far' corners (in 2D, the 'upper right' corner).
    std::array<std::size_t, N> minishape;
    // Shape to use in iterating over `BETA`s. We only include the 'far' corners
    // if we've reached the 'far' edge of `rectangle`.
    std::array<std::size_t, N> MINISHAPE;
    for (std::size_t i = 0; i < N; ++i) {
      std::size_t &m = minishape.at(i);
      std::size_t &M = MINISHAPE.at(i);
      if (alpha.at(i) + stride <= shape.at(i)) {
        // Do include the 'far' 'old' nodes.
        m = stride + 1;
        // Do not include the 'far' 'new' nodes.
        M = stride;
      } else {
        // This relies on there never being more than `1` but fewer than `stride
        // + 1` nodes, which is a result of the dimensions being of the form
        // `2^k + 1`.
        m = 1;
        M = 1;
      }
    }
    const MultiindexRectangle<N> minirectangle(alpha, minishape);
    const MultiindexRectangle<N> MINIRECTANGLE(alpha, MINISHAPE);
    for (const std::array<std::size_t, N> BETA :
         MINIRECTANGLE.indices(STRIDE)) {
      // Check that `BETA` is the multiindex of a 'new' node.
      bool BETA_is_new = false;
      for (std::size_t i = 0; i < N; ++i) {
        if (BETA.at(i) == alpha.at(i) + STRIDE) {
          BETA_is_new = true;
          break;
        }
      }
      if (!BETA_is_new) {
        continue;
      }
      Real interpolant = 0;
      for (const std::array<std::size_t, N> beta :
           minirectangle.indices(stride)) {
        Real weight = 1;
        for (std::size_t i = 0; i < N; ++i) {
          Real factor;
          const std::size_t B = BETA.at(i);
          if (B == alpha.at(i) + STRIDE) {
            factor = 0.5;
          } else if (B == beta.at(i)) {
            factor = 1;
          } else {
            factor = 0;
          }
          weight *= factor;
        }
        interpolant += weight * hierarchy.at(v, beta);
      }
      hierarchy.at(v, BETA) -= interpolant;
    }
  }
}

template <std::size_t N, typename Real>
void assign_num_level(const TensorMeshHierarchy<N, Real> &hierarchy,
                      const int l, Real *const v, const Real num) {
  for (const mgard::TensorNode<N> node : hierarchy.nodes(hierarchy.L - l)) {
    hierarchy.at(v, node.multiindex) = num;
  }
}

template <std::size_t N, typename Real>
void copy_level(const TensorMeshHierarchy<N, Real> &hierarchy, const int l,
                Real const *const v, Real *const work) {
  for (const mgard::TensorNode<N> node : hierarchy.nodes(hierarchy.L - l)) {
    hierarchy.at(work, node.multiindex) = hierarchy.at(v, node.multiindex);
  }
}

template <std::size_t N, typename Real>
void add_level(const TensorMeshHierarchy<N, Real> &hierarchy, const int l,
               Real *const v, Real const *const work) {
  for (const mgard::TensorNode<N> node : hierarchy.nodes(hierarchy.L - l)) {
    hierarchy.at(v, node.multiindex) += hierarchy.at(work, node.multiindex);
  }
}

template <std::size_t N, typename Real>
void subtract_level(const TensorMeshHierarchy<N, Real> &hierarchy, const int l,
                    Real *const v, Real const *const work) {
  for (const mgard::TensorNode<N> node : hierarchy.nodes(hierarchy.L - l)) {
    hierarchy.at(v, node.multiindex) -= hierarchy.at(work, node.multiindex);
  }
}

template <std::size_t N, typename Real>
void quantize_interleave(const TensorMeshHierarchy<N, Real> &hierarchy,
                         Real const *const v, int *const work, const Real norm,
                         const Real tol) {
  static_assert(sizeof(Real) % sizeof(int) == 0,
                "`int` size does not divide `Real` size");
  const std::size_t size_ratio = sizeof(Real) / sizeof(int);
  const mgard::LinearQuantizer<Real, int> quantizer(norm * tol);
  std::memcpy(work, &quantizer.quantum, sizeof(Real));
  for (std::size_t index = 0; index < hierarchy.ndof(); ++index) {
    work[size_ratio + index] = quantizer(v[index]);
  }
}

template <std::size_t N, typename Real>
void dequantize_interleave(const TensorMeshHierarchy<N, Real> &hierarchy,
                           Real *const v, int const *const work) {
  static_assert(sizeof(Real) % sizeof(int) == 0,
                "`int` size does not divide `Real` size");
  const std::size_t size_ratio = sizeof(Real) / sizeof(int);

  Real quantum;
  std::memcpy(&quantum, work, sizeof(Real));
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
    interpolate_old_to_new_and_subtract(hierarchy, l, 0, v);
#endif
    // copy the nodal values of v on l  to matrix work
    copy_level(hierarchy, l, v, work.data());

    assign_num_level(hierarchy, l + 1, work.data(), static_cast<Real>(0.0));

    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[jcol];
    }

    mass_matrix_multiply(hierarchy, l, 0, row_vec.data());

    restriction(hierarchy, l + 1, 0, row_vec.data());

    solve_tridiag_M(hierarchy, l + 1, 0, row_vec.data());

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

    interpolate_old_to_new_and_subtract(hierarchy, l, v);
    // copy the nodal values of v on l  to matrix work
    copy_level(hierarchy, l, v, work.data());

    assign_num_level(hierarchy, l + 1, work.data(), static_cast<Real>(0.0));

    // row-sweep
    for (int irow = 0; irow < nrow; ++irow) {
      for (int jcol = 0; jcol < ncol; ++jcol) {
        row_vec[jcol] = work[get_index(ncol, irow, jcol)];
      }

      mass_matrix_multiply(hierarchy, l, 1, row_vec.data());

      restriction(hierarchy, l + 1, 1, row_vec.data());

      solve_tridiag_M(hierarchy, l + 1, 1, row_vec.data());

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

        mass_matrix_multiply(hierarchy, l, 0, col_vec.data());

        restriction(hierarchy, l + 1, 0, col_vec.data());
        solve_tridiag_M(hierarchy, l + 1, 0, col_vec.data());

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

    mass_matrix_multiply(hierarchy, l - 1, 0, row_vec.data());

    restriction(hierarchy, l, 0, row_vec.data());
    solve_tridiag_M(hierarchy, l, 0, row_vec.data());

    for (int jcol = 0; jcol < ncol; ++jcol) {
      work[jcol] = row_vec[jcol];
    }

    subtract_level(hierarchy, l, work.data(), v); // do -(Qu - zl)

    // row-sweep
    for (int jcol = 0; jcol < ncol; ++jcol) {
      row_vec[jcol] = work[jcol];
    }

    interpolate_old_to_new_and_overwrite(hierarchy, l, 0, row_vec.data());

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

      mass_matrix_multiply(hierarchy, l - 1, 1, row_vec.data());

      restriction(hierarchy, l, 1, row_vec.data());
      solve_tridiag_M(hierarchy, l, 1, row_vec.data());

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

        mass_matrix_multiply(hierarchy, l - 1, 0, col_vec.data());

        restriction(hierarchy, l, 0, col_vec.data());
        solve_tridiag_M(hierarchy, l, 0, col_vec.data());

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

      interpolate_old_to_new_and_overwrite(hierarchy, l, 0, row_vec.data());

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

        interpolate_old_to_new_and_overwrite(hierarchy, l, 0, col_vec.data());

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

namespace {

// Not documenting the parameters here. I think it'll be easiest to understand
// by reading the code.

template <std::size_t N, typename Real>
void add_on_old_add_on_new(const TensorMeshHierarchy<N, Real> &hierarchy,
                           Real const *const src, Real *const dst,
                           const std::size_t l) {
  for (const TensorNode<N> node : hierarchy.nodes(l)) {
    hierarchy.at(dst, node.multiindex) += hierarchy.at(src, node.multiindex);
  }
}

template <std::size_t N, typename Real>
void subtract_on_old_zero_on_new(const TensorMeshHierarchy<N, Real> &hierarchy,
                                 Real const *const src, Real *const dst,
                                 const std::size_t l) {
  for (const TensorNode<N> node : hierarchy.nodes(l)) {
    Real &out = hierarchy.at(dst, node.multiindex);
    out = hierarchy.date_of_birth(node.multiindex) == l
              ? 0
              : out - hierarchy.at(src, node.multiindex);
  }
}

template <std::size_t N, typename Real>
void copy_negation_on_old_subtract_on_new(
    const TensorMeshHierarchy<N, Real> &hierarchy, Real const *const src,
    Real *const dst, const std::size_t l) {
  for (const TensorNode<N> node : hierarchy.nodes(l)) {
    const Real in = hierarchy.at(src, node.multiindex);
    Real &out = hierarchy.at(dst, node.multiindex);
    out = hierarchy.date_of_birth(node.multiindex) == l ? out - in : -in;
  }
}

template <std::size_t N, typename Real>
void copy_on_old_zero_on_new(const TensorMeshHierarchy<N, Real> &hierarchy,
                             Real const *const src, Real *const dst,
                             const std::size_t l) {
  // `l` shouldn't be zero, but in that case we'll do the expected thing: zero
  // every value.
  for (const TensorNode<N> node : hierarchy.nodes(l)) {
    hierarchy.at(dst, node.multiindex) =
        hierarchy.date_of_birth(node.multiindex) == l
            ? 0
            : hierarchy.at(src, node.multiindex);
  }
}

template <std::size_t N, typename Real>
void zero_on_old_copy_on_new(const TensorMeshHierarchy<N, Real> &hierarchy,
                             Real const *const src, Real *const dst,
                             const std::size_t l) {
  for (const TensorNode<N> node : hierarchy.nodes(l)) {
    hierarchy.at(dst, node.multiindex) =
        hierarchy.date_of_birth(node.multiindex) == l
            ? hierarchy.at(src, node.multiindex)
            : 0;
  }
}

template <std::size_t N, typename Real>
void zero_on_old_subtract_and_copy_back_on_new(
    const TensorMeshHierarchy<N, Real> &hierarchy, Real *const subtrahend,
    Real *const minuend, const std::size_t l) {
  for (const TensorNode<N> node : hierarchy.nodes(l)) {
    // Note that we may also write to a value of `subtrahend`.
    Real &out = hierarchy.at(minuend, node.multiindex);
    out = hierarchy.date_of_birth(node.multiindex) == l
              ? (hierarchy.at(subtrahend, node.multiindex) -= out)
              : 0;
  }
}

} // namespace

template <std::size_t N, typename Real>
void decompose(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v) {
  const std::size_t ndof = hierarchy.ndof();
  Real *const buffer = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  for (std::size_t l = hierarchy.L; l > 0; --l) {
    // We start with `Q_{l}u` on `nodes(l)` of `v`. First we copy the values on
    // `old_nodes(l)` to `buffer`. At the same time, we zero the values on
    // `new_nodes(l)` of `buffer` in preparation for the interpolation routine.
    copy_on_old_zero_on_new(hierarchy, v, buffer, l);
    // Now we have `Π_{l - 1}Q_{l}u` on `old_nodes(l)` of `buffer` and zeros on
    // `new_nodes(l)` of `buffer`. Time to interpolate.
    {
      const TensorProlongationAddition<N, Real> PA(hierarchy, l);
      PA(buffer);
    }
    // Now we have `Π_{l - 1}Q_{l}u` on `nodes(l)` (that is, on both
    // `old_nodes(l)` and `new_nodes(l)`) of `buffer`. `Q_{l}u` is still on
    // `nodes(l)` of `v`. We want to end up with
    //     1. `(I - Π_{l - 1})Q_{l}u` on `new_nodes(l)` of `v` and
    //     2. `(I - Π_{l - 1})Q_{l}u` on `nodes(l)` of `v`.
    // So, we will subtract the values on `new_nodes(l)` of `buffer` from the
    // values on `new_nodes(l)` of `v`, store the difference in both `buffer`
    // and `v`, and also zero the values on `old_nodes(l)` of `buffer`.
    zero_on_old_subtract_and_copy_back_on_new(hierarchy, v, buffer, l);
    // Now we have `(I - Π_{l - 1})Q_{l}u` on `nodes(l)` of `buffer`. Time to
    // project.
    {
      const TensorMassMatrix<N, Real> M(hierarchy, l);
      const TensorRestriction<N, Real> R(hierarchy, l);
      const TensorMassMatrixInverse<N, Real> m_inv(hierarchy, l - 1);
      M(buffer);
      R(buffer);
      m_inv(buffer);
    }
    // Now we have `Q_{l - 1}u - Π_{l - 1}Q_{l}u` on `old_nodes(l)` of `buffer`.
    // Time to correct `Π_{l - 1}Q_{l}u` on `old_nodes(l)` of `v`.
    add_on_old_add_on_new(hierarchy, buffer, v, l - 1);
    // Now we have `(I - Π_{l - 1})Q_{l}u` on `new_nodes(l)` of `v` and
    // `Q_{l - 1}u` on `old_nodes(l)` of `v`.
  }
  std::free(buffer);
}

template <std::size_t N, typename Real>
void recompose(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v) {
  const std::size_t ndof = hierarchy.ndof();
  Real *const buffer = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  for (std::size_t l = 1; l <= hierarchy.L; ++l) {
    // We start with `Q_{l - 1}u` on `old_nodes(l)` of `v` and
    // `(I - Π_{l - 1})Q_{l}u` on `new_nodes(l)` of `v`. We begin by copying
    // `(I - Π_{l - 1})Q_{l}u` to `buffer`.
    // I think we could instead copy all of `v` to `buffer` at the beginning and
    // then just zero `old_nodes(l)` of `buffer` here.
    zero_on_old_copy_on_new(hierarchy, v, buffer, l);
    // Now we have `(I - Π_{l - 1})Q_{l}u` on `nodes(l)` of `buffer`. Time to
    // project.
    {
      const TensorMassMatrix<N, Real> M(hierarchy, l);
      const TensorRestriction<N, Real> R(hierarchy, l);
      const TensorMassMatrixInverse<N, Real> m_inv(hierarchy, l - 1);
      M(buffer);
      R(buffer);
      m_inv(buffer);
    }
    // Now we have `Q_{l - 1}u - Π_{l - 1}Q_{l}u` on `old_nodes(l)` of `buffer`.
    // We can subtract `Q_{l - 1}u` (on `old_nodes(l)` of `v`) to obtain
    // `-Π_{l - 1}Q_{l}u`.
    subtract_on_old_zero_on_new(hierarchy, v, buffer, l);
    // Now we have `-Π_{l - 1}Q_{l}u` on `old_nodes(l)` of buffer. In addition,
    // we have zeros on `new_nodes(l)` of buffer, so we're ready to use
    // `TensorProlongationAddition`.
    {
      const TensorProlongationAddition<N, Real> PA(hierarchy, l);
      PA(buffer);
    }
    // Now we have `-Π_{l - 1}Q_{l}u` on `nodes(l)` of `buffer`. Subtracting
    // from `(I - Π_{l - 1})Q_{l}u`, we'll recover the projection.
    copy_negation_on_old_subtract_on_new(hierarchy, buffer, v, l);
    // Now we have `Q_{l}u` on `nodes(l)` of `v`.
  }
  std::free(buffer);
}

} // end namespace mgard

#endif
