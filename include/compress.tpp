// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ben Whitney, Qing Liu
//
// See LICENSE for details.
#ifndef COMPRESS_TPP
#define COMPRESS_TPP

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef MGARD_PROTOBUF
#include <type_traits>
#endif

#include "MGARDConfig.hpp"
#include "TensorMultilevelCoefficientQuantizer.hpp"
#include "TensorNorms.hpp"
#include "compressors.hpp"
#include "decompose.hpp"
#include "format.hpp"
#include "quantize.hpp"
#include "shuffle.hpp"

namespace mgard {

template <std::size_t N, typename Real>
CompressedDataset<N, Real>::CompressedDataset(
    const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
    const Real tolerance, void const *const data, const std::size_t size)
    : hierarchy(hierarchy), s(s), tolerance(tolerance),
      data_(static_cast<unsigned char const *>(data)), size_(size) {}

#ifdef MGARD_PROTOBUF
template <std::size_t N, typename Real>
CompressedDataset<N, Real>::CompressedDataset(
    const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
    const Real tolerance, void const *const data, const std::size_t size,
    const pb::Header &header)
    : CompressedDataset(hierarchy, s, tolerance, data, size) {
  // TODO: Fix this.
  header_ = header;
}
#endif

template <std::size_t N, typename Real>
void const *CompressedDataset<N, Real>::data() const {
  return data_.get();
}

template <std::size_t N, typename Real>
std::size_t CompressedDataset<N, Real>::size() const {
  return size_;
}

#ifdef MGARD_PROTOBUF
template <std::size_t N, typename Real>
pb::Header const *CompressedDataset<N, Real>::header() const {
  return &header_;
}

template <std::size_t N, typename Real>
void CompressedDataset<N, Real>::write(std::ostream &ostream) const {
  if (not header_.SerializeToOstream(&ostream)) {
    throw std::runtime_error("failed to serialize protocol buffer");
  }
  ostream.write(static_cast<char const *>(data()), size());
}
#endif

template <std::size_t N, typename Real>
DecompressedDataset<N, Real>::DecompressedDataset(
    const CompressedDataset<N, Real> &compressed, Real const *const data)
    : hierarchy(compressed.hierarchy), s(compressed.s),
      tolerance(compressed.tolerance), data_(data)
#ifdef MGARD_PROTOBUF
      ,
      header_(*compressed.header())
#endif
{
}

template <std::size_t N, typename Real>
Real const *DecompressedDataset<N, Real>::data() const {
  return data_.get();
}

using DEFAULT_INT_T = std::int64_t;

template <std::size_t N, typename Real>
CompressedDataset<N, Real>
compress(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v,
         const Real s, const Real tolerance) {
  const std::size_t ndof = hierarchy.ndof();
  Real *const u = new Real[ndof];
  shuffle(hierarchy, v, u);

#ifdef MGARD_PROTOBUF
  pb::Header header;
  populate_version_numbers(header);
  hierarchy.populate(header);
  decompose(hierarchy, u, header);
  header.mutable_quantization()->set_type(pb::Quantization::INT64_T);
  MemoryBuffer<unsigned char> quantized = quantization_buffer(ndof, header);
  quantize(hierarchy, s, tolerance, u, quantized.data.get(), header);
  MemoryBuffer<unsigned char> buffer =
      compress(quantized.data.get(), quantized.size, header);
#else
  decompose(hierarchy, u);
  using Qntzr = TensorMultilevelCoefficientQuantizer<N, Real, DEFAULT_INT_T>;
  const Qntzr quantizer(hierarchy, s, tolerance);
  using It = typename Qntzr::iterator;
  const RangeSlice<It> quantized_ = quantizer(u);
  DEFAULT_INT_T *const quantized = new quantized[ndof];
  std::copy(quantized_.begin(), quantized_.end(), quantized);
  MemoryBuffer<unsigned char> buffer =
#ifdef MGARD_ZSTD
      compress_memory_huffman(quantized, ndof)
#else
      compress_memory_z(quantized, ndof * sizeof(*quantized))
#endif
      ;
  delete[] quantized;
#endif
  delete[] u;
  return CompressedDataset<N, Real>(hierarchy, s, tolerance,
                                    buffer.data.release(), buffer.size
#ifdef MGARD_PROTOBUF
                                    ,
                                    header
#endif
  );
}

template <std::size_t N, typename Real>
DecompressedDataset<N, Real>
decompress(const CompressedDataset<N, Real> &compressed) {
  const std::size_t ndof = compressed.hierarchy.ndof();
  Real *const dequantized = new Real[ndof];
  Real *const v = new Real[ndof];
#ifdef MGARD_PROTOBUF
  const pb::Header &header = *compressed.header();
  MemoryBuffer<unsigned char> quantized = quantization_buffer(ndof, header);

  decompress(const_cast<void *>(compressed.data()), compressed.size(),
             quantized.data.get(), quantized.size, header);
  dequantize(compressed, quantized.data.get(), dequantized, header);

  recompose(compressed.hierarchy, dequantized, header);
#else
  DEFAULT_INT_T *const quantized = new DEFAULT_INT_T[ndof];
  const std::size_t quantizedLen = ndof * sizeof(*quantized);
#ifdef MGARD_ZSTD
  decompress_memory_huffman(const_cast<unsigned char *>(compressed.data()),
                            compressed.size(), quantized, quantizedLen);
#else
  decompress_memory_z(compressed.data(), compressed.size(),
                      reinterpret_cast<unsigned char *>(quantized),
                      quantizedLen);
#endif
  using Dqntzr = TensorMultilevelCoefficientDequantizer<N, DEFAULT_INT_T, Real>;
  const Dqntzr dequantizer(compressed.hierarchy, compressed.s,
                           compressed.tolerance);
  using It = typename Dqntzr::template iterator<DEFAULT_INT_T *>;
  const RangeSlice<It> dequantized_ = dequantizer(quantized, quantized + ndof);
  std::copy(dequantized_.begin(), dequantized_.end(), dequantized);
  delete[] quantized;

  recompose(compressed.hierarchy, dequantized);
#endif
  unshuffle(compressed.hierarchy, dequantized, v);
  delete[] dequantized;
  return DecompressedDataset<N, Real>(compressed, v);
}

} // namespace mgard

#endif
