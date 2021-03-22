// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ben Whitney, Qing Liu
//
// version: 0.1.0
// See LICENSE for details.
#ifndef MGARD_API_TPP
#define MGARD_API_TPP

#include <cstddef>

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include "TensorMultilevelCoefficientQuantizer.hpp"
#include "TensorNorms.hpp"
#include "mgard.hpp"
#include "shuffle.hpp"

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
#ifndef MGARD_ZSTD
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
#else
//! Compress an array of data using `zstd`.
  std::vector<unsigned char> zstd_output;
  int zstd_outsize;

  void *const buffer = compress_memory_huffman(quantized, zstd_output, zstd_outsize);

  const std::size_t size = zstd_outsize;
/*
  void *const buffer = new unsigned char[size];

  std::copy(zstd_output.begin(), zstd_output.end(),
            static_cast<unsigned char *>(buffer));
std::cout << "second value = " << * ((std::size_t *) buffer) << "\n";
*/
#endif
  return CompressedDataset<N, Real>(hierarchy, s, tolerance, buffer, size);
}

template <std::size_t N, typename Real>
DecompressedDataset<N, Real>
decompress(const CompressedDataset<N, Real> &compressed) {
  const std::size_t ndof = compressed.hierarchy.ndof();
  DEFAULT_INT_T *const quantized =
      static_cast<DEFAULT_INT_T *>(std::malloc(ndof * sizeof(*quantized)));
  // TODO: Figure out all these casts here and above.
#ifndef MGARD_ZSTD
  decompress_memory_z(const_cast<void *>(compressed.data()), compressed.size(),
                      reinterpret_cast<int *>(quantized),
                      ndof * sizeof(*quantized));
#else
  std::cout << "compressed.size() = " << compressed.size() << "\n";
  decompress_memory_huffman(reinterpret_cast<unsigned char *>(const_cast<void *>(compressed.data())),
                            compressed.size(), quantized,
                            ndof * sizeof(*quantized));
#endif
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
