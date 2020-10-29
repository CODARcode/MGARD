// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ben Whitney, Qing Liu
//
// version: 0.1.0
// See LICENSE for details.
#ifndef MGARD_API_TPP
#define MGARD_API_TPP

#include <cassert>
#include <cstddef>

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#include "TensorMultilevelCoefficientQuantizer.hpp"
#include "TensorNorms.hpp"
#include "mgard.hpp"
#include "mgard_compress.hpp"
#include "mgard_nuni.h"
#include "shuffle.hpp"

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

namespace mgard {

template <std::size_t N, typename Real>
CompressedDataset<N, Real>::CompressedDataset(
    const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
    const Real tolerance, void const *const data, const std::size_t size)
    : hierarchy(hierarchy), s(s), tolerance(tolerance),
      data_(static_cast<unsigned char const *>(data)), size_(size) {}

template <std::size_t N, typename Real>
void const *CompressedDataset<N, Real>::data() const {
  return data_.get();
}

template <std::size_t N, typename Real>
std::size_t CompressedDataset<N, Real>::size() const {
  return size_;
}

template <std::size_t N, typename Real>
DecompressedDataset<N, Real>::DecompressedDataset(
    const CompressedDataset<N, Real> &compressed, Real const *const data)
    : hierarchy(compressed.hierarchy), s(compressed.s),
      tolerance(compressed.tolerance), data_(data) {}

template <std::size_t N, typename Real>
Real const *DecompressedDataset<N, Real>::data() const {
  return data_.get();
}

using DEFAULT_INT_T = long int;

template <std::size_t N, typename Real>
CompressedDataset<N, Real>
compress(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v,
         const Real s, const Real tolerance) {
  const std::size_t ndof = hierarchy.ndof();
  // TODO: Can be smarter about copies later.
  Real *const u = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  shuffle(hierarchy, v, u);
  decompose(hierarchy, u);

  using Qntzr = TensorMultilevelCoefficientQuantizer<N, Real, DEFAULT_INT_T>;
  const Qntzr quantizer(hierarchy, s, tolerance);
  using It = typename Qntzr::iterator;
  const RangeSlice<It> quantized_range = quantizer(u);
  const std::vector<DEFAULT_INT_T> quantized(quantized_range.begin(),
                                             quantized_range.end());
  std::free(u);

  std::vector<std::uint8_t> z_output;
  // TODO: Check whether `compress_memory_z` changes its input.
  compress_memory_z(
      const_cast<void *>(static_cast<void const *>(quantized.data())),
      sizeof(DEFAULT_INT_T) * hierarchy.ndof(), z_output);
  // Possibly we should check that `sizeof(std::uint8_t)` is `1`.
  const std::size_t size = z_output.size();

  void *const buffer = new unsigned char[size];
  std::copy(z_output.begin(), z_output.end(),
            static_cast<unsigned char *>(buffer));
  return CompressedDataset<N, Real>(hierarchy, s, tolerance, buffer, size);
}

template <std::size_t N, typename Real>
DecompressedDataset<N, Real>
decompress(const CompressedDataset<N, Real> &compressed) {
  const std::size_t ndof = compressed.hierarchy.ndof();
  DEFAULT_INT_T *const quantized =
      static_cast<DEFAULT_INT_T *>(std::malloc(ndof * sizeof(*quantized)));
  // TODO: Figure out all these casts here and above.
  decompress_memory_z(const_cast<void *>(compressed.data()), compressed.size(),
                      reinterpret_cast<int *>(quantized),
                      ndof * sizeof(*quantized));

  using Dqntzr = TensorMultilevelCoefficientDequantizer<N, DEFAULT_INT_T, Real>;
  const Dqntzr dequantizer(compressed.hierarchy, compressed.s,
                           compressed.tolerance);
  using It = typename Dqntzr::template iterator<DEFAULT_INT_T *>;
  const RangeSlice<It> dequantized_range =
      dequantizer(quantized, quantized + ndof);

  // TODO: Can be smarter about copies later.
  Real *const buffer = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  std::copy(dequantized_range.begin(), dequantized_range.end(), buffer);
  std::free(quantized);

  recompose(compressed.hierarchy, buffer);
  Real *const v = new Real[ndof];
  unshuffle(compressed.hierarchy, buffer, v);
  std::free(buffer);
  return DecompressedDataset<N, Real>(compressed, v);
}

} // namespace mgard

#endif
