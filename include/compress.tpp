// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ben Whitney, Qing Liu
//
// See LICENSE for details.
#ifndef COMPRESS_TPP
#define COMPRESS_TPP

#include <cstddef>

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include "TensorMultilevelCoefficientQuantizer.hpp"
#include "TensorNorms.hpp"
#include "decompose.hpp"
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

const std::string SIGNATURE_STR = "COMPRESSED_BY_MGARD";
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
  // Compress an array of data using `zstd`.
  std::vector<unsigned char> zstd_output;
  int zstd_outsize;

  void *const buffer_h =
      compress_memory_huffman(quantized, zstd_output, zstd_outsize);

  // Size of metadata: 4
  // Signature: 19
  // Version: 1 + 1 
  // Type: 1
  // Number of dims: 1
  // Tolerance: 8
  // S: 8
  // L-inf norm or s-norm: 8
  // Target level: 4
  uint32_t metadata_size = 4
                         + 19 
                         + 1 + 1 
                         + 1
                         + 1
                         + N * 8
                         + 8 + 8 + 8
                         + 4
                         + 1;
  // pack the minimal metadata for now, i.e., error tolerence and s
  unsigned char *const buffer = static_cast<unsigned char *>(std::malloc(zstd_outsize + metadata_size));
  unsigned char *b = buffer;

  *(uint32_t *)b = metadata_size;
  b += 4;

  std::memcpy(b, SIGNATURE_STR.c_str(), SIGNATURE_STR.size());
  b += SIGNATURE_STR.size();

  // Version major and minor
  *(uint8_t *)b = 0;
  b += 1;
  *(uint8_t *)b = 1;
  b += 1;

  // Type: double 0, float 1
  *(uint8_t *)b = 0;
  b += 1;

  // Number of dims
  *(uint8_t *)b = N;
  b += 1;

  for (uint8_t i = 0; i < N; i++) {
    *(uint64_t *)b = hierarchy.shapes.at(hierarchy.L).at(i);
    b += 8;
  }

  // Tolerance
  *(double *)b = tolerance;
  b += 8;

  // S
  *(double *)b = s;
  b += 8;

  // L-inf norm or s-norm. TODO
  *(double *)b = 0;
  b += 8;

  // Target level. TODO
  *(uint32_t *)b = hierarchy.L;
  b += 4;

  // Target level. TODO
  *(uint8_t *)b = 0;
  b += 1;

  std::memcpy(buffer + metadata_size, buffer_h, zstd_outsize);
  std::free(buffer_h); 

  const std::size_t size = zstd_outsize + metadata_size;
#endif
  return CompressedDataset<N, Real>(hierarchy, s, tolerance, buffer, size);
}

void const * mgard_decompress(void const *const compressed_buffer, std::size_t compressed_size) {
  unsigned char *b = (unsigned char *)compressed_buffer;

  uint32_t metadata_size = *(uint32_t *)b;
  b += 4;
 
  char sig_str[SIGNATURE_STR.size() + 1];
  std::memcpy(sig_str, b, SIGNATURE_STR.size());
  b += SIGNATURE_STR.size();
  sig_str[SIGNATURE_STR.size()] = '\0';
  if (strcmp(sig_str, SIGNATURE_STR.c_str()) != 0) {
    throw std::invalid_argument("Data was not compressed by MGARD.");
  }

  // Version major and minor
  uint8_t version_major = *(uint8_t *)b;
  b += 1;

  uint8_t version_minor = *(uint8_t *)b;
  b += 1;

  // Type
  uint8_t type = *(uint8_t *)b;
  b += 1;

  // Number of dims 
  uint8_t ndims = *(uint8_t *)b;
  b += 1;

  uint64_t shape[ndims];
  for (uint8_t i = 0; i < ndims; i++) {
    shape[i] = *(uint64_t *)b;
    b += 8;
  }

  // Tolerance
  double tol = *(double *)b;
  b += 8;

  // S
  double s = *(double *)b;
  b += 8;

  // L-inf norm or S-norm
  double norm = *(double *)b;
  b += 8;

  // L-inf norm or S-norm
  uint32_t target_level = *(uint32_t *)b;
  b += 4;

  // Grid type
  uint32_t grid_type = *(uint8_t *)b;
  b += 1;

  std::cout << "ndims = " << (unsigned)ndims << " tol = " << tol << " s = " << s << " target_level = "<< target_level << " grid_type = "<<grid_type<<"\n";

  unsigned char *cb_copy = (unsigned char*)std::malloc(compressed_size - metadata_size);
  std::memcpy(cb_copy, compressed_buffer + metadata_size, compressed_size - metadata_size);

  void * decompressed_buffer = 0;

  switch(ndims) {
    case 1:
      if (type == 0) {
      } else if (type == 1) {
      }
      break;
    case 2:
      if (type == 0) {
        const std::array<std::size_t, 2> dims = {shape[0], shape[1]};
        TensorMeshHierarchy<2, double> hierarchy(dims);
        const mgard::CompressedDataset<2, double> compressed(hierarchy, s, tol,
                                                             cb_copy,
                                                             compressed_size - metadata_size);
        const mgard::DecompressedDataset<2, double> decompressed =
          mgard::decompress(compressed);
        decompressed_buffer = std::malloc(hierarchy.ndof() * 8);
        std::memcpy(decompressed_buffer, decompressed.data(), hierarchy.ndof() * 8);
      } else if (type == 1) {
      }
      break;
    case 3:
      if (type == 0) {
      } else if (type == 1) {
      }
      break;
  }

  return decompressed_buffer;
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
  decompress_memory_huffman(
      reinterpret_cast<unsigned char *>(const_cast<void *>(compressed.data())),
      compressed.size(), quantized, ndof * sizeof(*quantized));
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
