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
#include "decompose.hpp"
#include "shuffle.hpp"

namespace mgard {

enum GridType { UNIFORM = 0, NONUNIFORM };

enum GridType GetGridType(uint8_t g) {
  switch (g) {
  case 0:
    return UNIFORM;
  case 1:
    return NONUNIFORM;
  default:
    throw std::runtime_error("U.");
  }
}

template <std::size_t N, typename Real>
CompressedDataset<N, Real>::CompressedDataset(
    const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
    const Real tolerance, void const *const data, const std::size_t size)
    : hierarchy(hierarchy), s(s), tolerance(tolerance),
      data_(static_cast<unsigned char const *>(data)), size_(size) {
  {
    pb::VersionNumber *const mgard_version = new pb::VersionNumber;
    mgard_version->set_major_(MGARD_VERSION_MAJOR);
    mgard_version->set_minor_(MGARD_VERSION_MINOR);
    mgard_version->set_patch_(MGARD_VERSION_PATCH);
    protocol_buffer.set_allocated_mgard_version(mgard_version);
  }
  {
    pb::VersionNumber *const file_format_version = new pb::VersionNumber;
    file_format_version->set_major_(MGARD_FILE_VERSION_MAJOR);
    file_format_version->set_minor_(MGARD_FILE_VERSION_MINOR);
    file_format_version->set_patch_(MGARD_FILE_VERSION_PATCH);
    protocol_buffer.set_allocated_file_format_version(file_format_version);
  }
  {
    pb::Domain *const domain = new pb::Domain;
    domain->set_topology(pb::Domain::CARTESIAN_GRID);
    {
      pb::CartesianGridTopology *const cartesian_grid_topology =
          new pb::CartesianGridTopology;
      cartesian_grid_topology->set_dimension(N);
      const std::array<std::size_t, N> &SHAPE = hierarchy.shapes.back();
      for (const std::size_t n : SHAPE) {
        cartesian_grid_topology->add_shape(n);
      }
      domain->set_allocated_cartesian_grid_topology(cartesian_grid_topology);
    }
    pb::Domain::Geometry geometry;
    if (hierarchy.uniform) {
      geometry = pb::Domain::UNIT_CUBE;
    } else {
      geometry = pb::Domain::EXPLICIT_CUBE;
      {
        pb::ExplicitCubeGeometry *const explicit_cube_geometry =
            new pb::ExplicitCubeGeometry;
        for (const std::vector<Real> &coords : hierarchy.coordinates) {
          for (const Real coord : coords) {
            explicit_cube_geometry->add_coordinates(static_cast<double>(coord));
          }
        }
        domain->set_allocated_explicit_cube_geometry(explicit_cube_geometry);
      }
    }
    domain->set_geometry(geometry);
    protocol_buffer.set_allocated_domain(domain);
  }
  {
    pb::Dataset *const dataset = new pb::Dataset;
    pb::Dataset::Type dataset_type;
    if (std::is_same<Real, float>::value) {
      dataset_type = pb::Dataset::FLOAT;
    } else if (std::is_same<Real, double>::value) {
      dataset_type = pb::Dataset::DOUBLE;
    } else {
      // It would be better to catch this at compile time.
      throw std::runtime_error("unsupported dataset type");
    }
    dataset->set_type(dataset_type);
    dataset->set_dimension(1);
    protocol_buffer.set_allocated_dataset(dataset);
  }
  {
    pb::ErrorControl *const error_control = new pb::ErrorControl;
    error_control->set_mode(pb::ErrorControl::ABSOLUTE);
    pb::ErrorControl::Norm norm;
    if (s == std::numeric_limits<Real>::infinity()) {
      norm = pb::ErrorControl::L_INFINITY;
    } else {
      norm = pb::ErrorControl::S_NORM;
      error_control->set_s(s);
    }
    error_control->set_norm(norm);
    error_control->set_tolerance(tolerance);
    protocol_buffer.set_allocated_error_control(error_control);
  }
  {
    pb::Decomposition *const decomposition = new pb::Decomposition;
    decomposition->set_transform(pb::Decomposition::MULTILEVEL_COEFFICIENTS);
    decomposition->set_hierarchy(pb::Decomposition::POWER_OF_TWO_PLUS_ONE);
    protocol_buffer.set_allocated_decomposition(decomposition);
  }
  {
    pb::Quantization *const quantization = new pb::Quantization;
    quantization->set_method(pb::Quantization::COEFFICIENTWISE_LINEAR);
    quantization->set_bin_widths(pb::Quantization::PER_COEFFICIENT);
    quantization->set_type(pb::Quantization::INT64_T);
    {
      const std::int64_t a = 1;
      quantization->set_big_endian(
          not*reinterpret_cast<std::int8_t const *>(&a));
    }
    protocol_buffer.set_allocated_quantization(quantization);
  }
  {
    pb::Encoding *const encoding = new pb::Encoding;
    encoding->set_preprocessor(pb::Encoding::SHUFFLE);
    encoding->set_compressor(
#if defined(MGARD_ZSTD)
        pb::Encoding::CPU_HUFFMAN_ZSTD
#elif defined(MGARD_ZLIB)
        pb::Encoding::CPU_HUFFMAN_ZLIB
#else
        pb::Encoding::NOOP
#endif
    );
    protocol_buffer.set_allocated_encoding(encoding);
  }
  {
    pb::Buffer *const buffer = new pb::Buffer;
    buffer->set_interpretation(pb::Buffer::SINGLE_DATASET_AFTER);
    protocol_buffer.set_allocated_buffer(buffer);
  }
}

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
void CompressedDataset<N, Real>::write(std::ostream &ostream) const {
  if (not protocol_buffer.SerializeToOstream(&ostream)) {
    throw std::runtime_error("failed to serialize protocol buffer");
  }
  ostream.write(static_cast<char const *>(data()), size());
}
#endif

template <std::size_t N, typename Real>
DecompressedDataset<N, Real>::DecompressedDataset(
    const CompressedDataset<N, Real> &compressed, Real const *const data)
    : hierarchy(compressed.hierarchy), s(compressed.s),
      tolerance(compressed.tolerance), data_(data) {}

template <std::size_t N, typename Real>
Real const *DecompressedDataset<N, Real>::data() const {
  return data_.get();
}

const std::string SIGNATURE_STR = "MGARD";
using DEFAULT_INT_T = std::int64_t;

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
  std::size_t size;
  void *const buffer_h = compress_memory_huffman(quantized, size);
  void *const buffer = new unsigned char[size];
  std::memcpy(static_cast<unsigned char *>(buffer), buffer_h, size);
  std::free(buffer_h);
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
                      reinterpret_cast<unsigned char *>(quantized),
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
