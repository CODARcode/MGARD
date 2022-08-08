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
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

#include "MGARDConfig.hpp"
#include "TensorMultilevelCoefficientQuantizer.hpp"
#include "TensorNorms.hpp"
#include "decompose.hpp"
#include "format.hpp"
#include "lossless.hpp"
#include "quantize.hpp"
#include "shuffle.hpp"

namespace mgard {

template <std::size_t N, typename Real>
CompressedDataset<N, Real>
compress(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v,
         const Real s, const Real tolerance) {
  const std::size_t ndof = hierarchy.ndof();
  Real *const u = new Real[ndof];
  shuffle(hierarchy, v, u);
  pb::Header header;
  populate_defaults(header);
  hierarchy.populate(header);
  decompose(hierarchy, header, u);
  {
    pb::ErrorControl &e = *header.mutable_error_control();
    e.set_mode(pb::ErrorControl::ABSOLUTE);
    if (s == std::numeric_limits<Real>::infinity()) {
      e.set_norm(pb::ErrorControl::L_INFINITY);
    } else {
      e.set_norm(pb::ErrorControl::S_NORM);
      e.set_s(s);
    }
    e.set_tolerance(tolerance);
  }
  MemoryBuffer<unsigned char> quantized = quantization_buffer(header, ndof);
  quantize(hierarchy, header, s, tolerance, u, quantized.data.get());
  MemoryBuffer<unsigned char> buffer =
      compress(header, quantized.data.get(), quantized.size);
  delete[] u;
  return CompressedDataset<N, Real>(hierarchy, header, s, tolerance,
                                    buffer.data.release(), buffer.size);
}

template <std::size_t N, typename Real>
DecompressedDataset<N, Real>
decompress(const CompressedDataset<N, Real> &compressed) {
  const std::size_t ndof = compressed.hierarchy.ndof();
  Real *const dequantized = new Real[ndof];
  Real *const v = new Real[ndof];
  MemoryBuffer<unsigned char> quantized =
      quantization_buffer(compressed.header, ndof);

  decompress(compressed.header, const_cast<void *>(compressed.data()),
             compressed.size(), quantized.data.get(), quantized.size);
  dequantize(compressed, quantized.data.get(), dequantized);

  recompose(compressed.hierarchy, compressed.header, dequantized);
  unshuffle(compressed.hierarchy, dequantized, v);
  delete[] dequantized;
  return DecompressedDataset<N, Real>(compressed, v);
}

} // namespace mgard

#endif
