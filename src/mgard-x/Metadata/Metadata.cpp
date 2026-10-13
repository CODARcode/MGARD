/*
 * Copyright 2026, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Kenneth Moreland (morelandkd@ornl.gov)
 * Date: January 5, 2026
 */

#include "mgard-x/Metadata/Metadata.hpp"
#include "MGARDConfig.hpp"
#include "mgard-x/RuntimeX/Utilities/Exceptions.h"
#include "proto/mgard.pb.h"

#include <google/protobuf/io/coded_stream.h>

#include <limits>

using mgard_x::SERIALIZED_TYPE;

using SerializedIter = std::vector<SERIALIZED_TYPE>::const_iterator;

namespace {

template <typename Int> bool big_endian() {
  static_assert(std::is_integral<Int>::value,
                "can only check endianness of integral types");
  const Int n = 1;
  return not *reinterpret_cast<unsigned char const *>(&n);
}

// The width of the quantized-coefficient type is a build-time choice
// (QUANTIZED_INT in DataTypes.h) that the reader must agree with, so it is
// recorded in the header rather than assumed. Writing it means the width can
// be changed later without turning existing files into garbage: a mismatch is
// then a clean error instead of a misparse.
mgard::pb::Quantization::Type QuantizationTypeForWidth(std::size_t width) {
  switch (width) {
  case 1:
    return mgard::pb::Quantization::INT8_T;
  case 2:
    return mgard::pb::Quantization::INT16_T;
  case 4:
    return mgard::pb::Quantization::INT32_T;
  case 8:
    return mgard::pb::Quantization::INT64_T;
  default:
    throw mgard_x::InvalidDataException("unsupported quantized integer width.");
  }
}

std::size_t WidthForQuantizationType(mgard::pb::Quantization::Type type) {
  switch (type) {
  case mgard::pb::Quantization::INT8_T:
    return 1;
  case mgard::pb::Quantization::INT16_T:
    return 2;
  case mgard::pb::Quantization::INT32_T:
    return 4;
  case mgard::pb::Quantization::INT64_T:
    return 8;
  default:
    throw mgard_x::InvalidDataException("unrecognized quantization type.");
  }
}

uint32_t ComputeCRC32(const std::vector<SERIALIZED_TYPE> &data,
                      std::size_t start = 0) {
  // `crc32_z` takes a `z_size_t`.
  if (data.size() - start > std::numeric_limits<z_size_t>::max()) {
    std::cout << mgard_x::log::log_err
              << "buffer is too large (size would overflow.\n";
  }
  uLong crc32_ = crc32_z(0, Z_NULL, 0);
  crc32_ = crc32_z(crc32_, static_cast<const Bytef *>(data.data() + start),
                   data.size() - start);
  return crc32_;
}

std::vector<SERIALIZED_TYPE> SerializeProtoBuf(mgard::pb::Header &header) {
  size_t header_size = header.ByteSize();
  std::vector<SERIALIZED_TYPE> header_bytes(header_size);
  header.SerializeToArray(header_bytes.data(), header_size);
  return header_bytes;
}

mgard::pb::Header
DeserializeProtoBuf(const std::vector<SERIALIZED_TYPE> header_bytes,
                    uint64_t offset) {
  uint64_t header_size = header_bytes.size() - offset;

  // The `CodedInputStream` constructor takes an `int`.
  if (header_size > std::numeric_limits<int>::max()) {
    std::cout << mgard_x::log::log_err
              << "header is too large (size would overflow).\n";
  }
  mgard::pb::Header header;
  google::protobuf::io::CodedInputStream stream(
      static_cast<google::protobuf::uint8 const *>(header_bytes.data() +
                                                   offset),
      header_size);
  if (not header.ParseFromCodedStream(&stream)) {
    throw std::runtime_error("header parsing encountered read or format error");
  }
  if (not stream.ConsumedEntireMessage()) {
    throw std::runtime_error("part of header left unparsed");
  }
  return header;
}

template <typename T>
void Serialize(const T &item, std::vector<SERIALIZED_TYPE> &vec) {
  if constexpr (std::is_integral<T>::value) {
    T in = item;
    for (int i = 0; i < sizeof(T); i++) {
      vec.push_back(in);
      in = in >> 8;
    }
  } else {
    const SERIALIZED_TYPE *in =
        reinterpret_cast<const SERIALIZED_TYPE *>(&item);
    vec.insert(vec.end(), in, in + sizeof(item));
  }
}

template <typename T> void Deserialize(T &item, SerializedIter &iter) {
  if constexpr (std::is_integral<T>::value) {
    T out = 0;
    for (int i = sizeof(T) - 1; i >= 0; i--) {
      out = out << 8;
      out = out + *(iter + i);
    }
    item = out;
  } else {
    SERIALIZED_TYPE *out = reinterpret_cast<SERIALIZED_TYPE *>(&item);
    std::copy(iter, iter + sizeof(T), out);
  }
  iter += sizeof(T);
}

void SerializeSignature(std::vector<SERIALIZED_TYPE> &vec) {
  for (char c : mgard::SIGNATURE) {
    Serialize(c, vec);
  }
}

// This function does not assign to a signature data member. Instead, it just
// checks that the deserialized signature matches `mgard::SIGNATURE`.
void DeserializeSignature(SerializedIter &iter) {
  for (const char c : mgard::SIGNATURE) {
    char c_;
    Deserialize(c_, iter);
    if (c_ != c) {
      throw mgard_x::InvalidDataException("signature mismatch.");
    }
  }
}

void SerializeBytes(const std::vector<SERIALIZED_TYPE> &data,
                    std::vector<SERIALIZED_TYPE> &vec) {
  vec.insert(vec.end(), data.begin(), data.end());
}

// Write `tolerances` into `roi` using whichever of the two encodings is
// smaller. ROI maps are piecewise constant in practice (a few distinct
// tolerances over long runs of blocks), where run-length is dramatically
// smaller; a fully irregular map is bigger under run-length, so both are
// built and the smaller one wins. Both decode to the same vector.
void SerializeROITolerances(const std::vector<double> &tolerances,
                            mgard::pb::RegionOfInterest &roi) {
  mgard::pb::PackedTolerances packed;
  google::protobuf::RepeatedField<double> &packed_values =
      *packed.mutable_tolerances();
  packed_values.Reserve(tolerances.size());
  for (double t : tolerances) {
    packed_values.Add(t);
  }

  mgard::pb::RunLengthTolerances run_length;
  google::protobuf::RepeatedField<double> &rl_values =
      *run_length.mutable_values();
  google::protobuf::RepeatedField<google::protobuf::uint64> &rl_lengths =
      *run_length.mutable_run_lengths();
  for (std::size_t i = 0; i < tolerances.size();) {
    std::size_t j = i;
    // Bit-exact comparison: the decoded tolerances must reproduce the
    // compressor's quantization steps exactly, so runs may only merge values
    // that are identical, not merely close.
    while (j < tolerances.size() && tolerances[j] == tolerances[i]) {
      j++;
    }
    rl_values.Add(tolerances[i]);
    rl_lengths.Add(static_cast<google::protobuf::uint64>(j - i));
    i = j;
  }

  if (run_length.ByteSizeLong() < packed.ByteSizeLong()) {
    *roi.mutable_run_length() = run_length;
  } else {
    *roi.mutable_packed() = packed;
  }
}

// Inverse of SerializeROITolerances.
std::vector<double>
DeserializeROITolerances(const mgard::pb::RegionOfInterest &roi) {
  std::vector<double> tolerances;
  if (roi.has_packed()) {
    const google::protobuf::RepeatedField<double> &values =
        roi.packed().tolerances();
    tolerances.assign(values.begin(), values.end());
  } else if (roi.has_run_length()) {
    const mgard::pb::RunLengthTolerances &run_length = roi.run_length();
    if (run_length.values_size() != run_length.run_lengths_size()) {
      throw mgard_x::InvalidDataException(
          "ROI tolerance map has mismatched run values and run lengths.");
    }
    std::size_t total = 0;
    for (int i = 0; i < run_length.run_lengths_size(); i++) {
      total += run_length.run_lengths(i);
    }
    tolerances.reserve(total);
    for (int i = 0; i < run_length.values_size(); i++) {
      tolerances.insert(tolerances.end(), run_length.run_lengths(i),
                        run_length.values(i));
    }
  } else {
    throw mgard_x::InvalidDataException(
        "ROI tolerance map is present but carries no tolerance encoding.");
  }
  return tolerances;
}

} // anonymous namespace

namespace mgard_x {

void MetadataBase::InitializeConfig(Config &config) {
  config.decomposition = decomposition;
  config.lossless = ltype;
  // Only overwrite a backend's parameters when the metadata actually carries
  // them (non-zero). A file compressed with a non-Huffman backend leaves the
  // Huffman fields at 0; copying those zeros would make
  // HuffmanWorkspace::resize divide by a zero block size. Leaving config
  // defaults intact avoids that (the unused backend's workspace is harmlessly
  // sized with defaults).
  if (huff_dict_size != 0) {
    config.huff_dict_size = huff_dict_size;
  }
  if (huff_block_size != 0) {
    config.huff_block_size = huff_block_size;
  }
  if (block_delta_block_size != 0) {
    config.block_delta_block_size = block_delta_block_size;
  }
  // The decomposition has to be rebuilt with the levels it was made with.
  // Streams from before the level cap was recorded used all levels.
  config.num_global_refactoring_level = has_num_global_refactoring_level
                                            ? (SIZE)num_global_refactoring_level
                                            : std::numeric_limits<SIZE>::max();
  // The hybrid parameters are only meaningful for a hybrid file, and
  // Deserialize refuses to produce a hybrid file without them, so a Hybrid
  // decomposition here always carries a full set. For non-hybrid files leave
  // the caller's Config alone -- those fields are unused and overwriting them
  // with zeros would break a subsequent hybrid compression through the same
  // Config.
  if (decomposition == decomposition_type::Hybrid) {
    config.num_local_refactoring_level = (SIZE)hybrid_num_local_levels;
    config.num_global_refactoring_level = (SIZE)hybrid_num_global_levels;
    config.projection_mode = hybrid_projection_mode;
    config.enable_roi = hybrid_enable_roi;
    if (hybrid_enable_roi) {
      config.roi_tolerance_map = hybrid_roi_tolerance_map;
    }
  }
}

void MetadataBase::PrintSummary() {
  std::cout << "=======Metadata Summary=======\n";
  std::cout << "Signature: ";
  for (const char c : mgard::SIGNATURE)
    std::cout << c;
  std::cout << "\n";
  std::cout << "MGARD version: " << (int)software_version[0] << "."
            << (int)software_version[1] << "." << (int)software_version[2]
            << "\n";
  std::cout << "File format version: " << (int)file_version[0] << "."
            << (int)file_version[1] << "." << (int)file_version[2] << "\n";
  std::cout << "Metadata size: " << metadata_size << "\n";
  std::cout << "Metadata crc32: " << metadata_crc32 << "\n";
  std::cout << "Endiness: ";
  if (etype == endiness_type::Big_Endian) {
    std::cout << "Big Endian\n";
  } else {
    std::cout << "Little Endian\n";
  }
  std::cout << "Data type: ";
  if (dtype == data_type::Float) {
    std::cout << "Float\n";
  } else if (dtype == data_type::Double) {
    std::cout << "Double\n";
  }
  std::cout << "Topology: ";
  if (dstype == data_structure_type::Cartesian_Grid_Uniform) {
    std::cout << "Uniform Grid\n";
  } else if (dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {
    std::cout << "Non-uniform Grid\n";
  }
  std::cout << "Shape: ";
  for (uint64_t &c : shape)
    std::cout << c << " ";
  std::cout << "\n";
  std::cout << "Function Decomposition: ";
  if (decomposition == decomposition_type::MultiDim) {
    std::cout << "MultiDim\n";
  } else if (decomposition == decomposition_type::SingleDim) {
    std::cout << "SingleDim\n";
  } else if (decomposition == decomposition_type::Hybrid) {
    std::cout << "Hybrid\n";
    std::cout << "Local refactoring levels: " << hybrid_num_local_levels
              << "\n";
    std::cout << "Global refactoring levels: " << hybrid_num_global_levels
              << "\n";
    std::cout << "Local block size: " << hybrid_local_block_size << "\n";
    std::cout << "Projection: "
              << (hybrid_projection_mode ==
                          compression_projection_mode_type::Hierarchical
                      ? "hierarchical"
                      : "orthogonal")
              << "\n";
    std::cout << "ROI: ";
    if (hybrid_enable_roi) {
      std::cout << "enabled (" << hybrid_roi_tolerance_map.size()
                << " level-0 blocks";
      if (!hybrid_roi_block_dimensions.empty()) {
        std::cout << ", grid";
        for (uint64_t n : hybrid_roi_block_dimensions)
          std::cout << " " << n;
      }
      std::cout << ")\n";
    } else {
      std::cout << "disabled\n";
    }
  }
  std::cout << "Domain Decomposition: ";
  if (domain_decomposed) {
    std::cout << "blocks of ";
    for (size_t d = 0; d < domain_decomposition_block_shape.size(); d++) {
      std::cout << (d ? "x" : "") << domain_decomposition_block_shape[d];
    }
    std::cout << "\n";
  } else {
    std::cout << "No\n";
  }
  std::cout << "Error bound mode: ";
  if (ebtype == error_bound_type::REL) {
    std::cout << "REL\n";
  } else if (ebtype == error_bound_type::ABS) {
    std::cout << "ABS\n";
  }
  std::cout << "Norm type: ";
  if (ntype == norm_type::L_Inf) {
    std::cout << "L_Inf\n";
  } else if (ntype == norm_type::L_2) {
    std::cout << "L_2\n";
  }
  std::cout << "Norm: " << norm << "\n";
  std::cout << "tol: " << tol << "\n";
  std::cout << "s: " << s << "\n";

  std::cout << "Lossless:  ";
  if (ltype == mgard_x::lossless_type::Huffman) {
    std::cout << "Huffman\n";
    std::cout << "Huffman dictionary size: " << huff_dict_size << "\n";
    std::cout << "Huffman block size: " << huff_block_size << "\n";
  } else if (ltype == mgard_x::lossless_type::Huffman_LZ4) {
    std::cout << "Huffman_LZ4\n";
    std::cout << "Huffman dictionary size: " << huff_dict_size << "\n";
    std::cout << "Huffman block size: " << huff_block_size << "\n";
  } else if (ltype == mgard_x::lossless_type::LZ4) {
    std::cout << "LZ4\n";
  } else if (ltype == mgard_x::lossless_type::Huffman_Zstd) {
    std::cout << "Huffman_Zstd\n";
    std::cout << "Huffman dictionary size: " << huff_dict_size << "\n";
    std::cout << "Huffman block size: " << huff_block_size << "\n";
  } else if (ltype == mgard_x::lossless_type::CPU_Lossless) {
    std::cout << "CPU_Lossless\n";
  } else if (ltype == mgard_x::lossless_type::BlockDelta) {
    std::cout << "BlockDelta\n";
    std::cout << "BlockDelta block size: " << block_delta_block_size << "\n";
  }

  std::cout << "Backend:  ";
  if (ptype == processor_type::X_SERIAL) {
    std::cout << "X_SERIAL\n";
  } else if (ptype == processor_type::X_CUDA) {
    std::cout << "X_OPENMP\n";
  } else if (ptype == processor_type::X_OPENMP) {
    std::cout << "X_CUDA\n";
  } else if (ptype == processor_type::X_HIP) {
    std::cout << "X_HIP\n";
  } else if (ptype == processor_type::X_SYCL) {
    std::cout << "X_SYCL\n";
  }
}

std::uint64_t MetadataBase::SerializePreambleSize() {
  std::uint64_t preamble_size = 0;
  preamble_size += mgard::SIGNATURE.size();
  preamble_size += sizeof(uint64_t); // header size
  preamble_size += sizeof(uint32_t); // crc32 size
  return preamble_size;
}

std::vector<SERIALIZED_TYPE> MetadataBase::Serialize() {
  mgard::pb::Header header;

  { // Version Number
    software_version[0] = MGARD_VERSION_MAJOR;
    software_version[1] = MGARD_VERSION_MINOR;
    software_version[2] = MGARD_VERSION_PATCH;

    mgard::pb::VersionNumber &mgard_version_number =
        *header.mutable_mgard_version();
    mgard_version_number.set_major_(MGARD_VERSION_MAJOR);
    mgard_version_number.set_minor_(MGARD_VERSION_MINOR);
    mgard_version_number.set_patch_(MGARD_VERSION_PATCH);

    file_version[0] = MGARD_FILE_VERSION_MAJOR;
    file_version[1] = MGARD_FILE_VERSION_MINOR;
    file_version[2] = MGARD_FILE_VERSION_PATCH;

    mgard::pb::VersionNumber &format_version_number =
        *header.mutable_file_format_version();
    mgard_version_number.set_major_(MGARD_FILE_VERSION_MAJOR);
    mgard_version_number.set_minor_(MGARD_FILE_VERSION_MINOR);
    mgard_version_number.set_patch_(MGARD_FILE_VERSION_PATCH);
  }

  { // Domain
    mgard::pb::Domain &domain = *header.mutable_domain();
    domain.set_topology(mgard::pb::Domain::CARTESIAN_GRID);
    mgard::pb::CartesianGridTopology &cartesian_grid_topology =
        *domain.mutable_cartesian_grid_topology();
    cartesian_grid_topology.set_dimension(total_dims);
    google::protobuf::RepeatedField<google::protobuf::uint64> &shape_ =
        *cartesian_grid_topology.mutable_shape();
    shape_.Resize(total_dims, 0);
    std::copy(shape.begin(), shape.end(), shape_.mutable_data());
    mgard::pb::Domain::Geometry geometry;
    if (dstype == data_structure_type::Cartesian_Grid_Uniform) {
      geometry = mgard::pb::Domain::UNIT_CUBE;
    } else {
      geometry = mgard::pb::Domain::EXPLICIT_CUBE;
      mgard::pb::ExplicitCubeGeometry &explicit_cube_geometry =
          *domain.mutable_explicit_cube_geometry();
      google::protobuf::RepeatedField<double> &coordinates_ =
          *explicit_cube_geometry.mutable_coordinates();

      uint64_t totel_len = 0;
      for (DIM d = 0; d < total_dims; d++)
        totel_len += shape[d];
      coordinates_.Resize(totel_len, 0);
      double *p = coordinates_.mutable_data();
      for (DIM d = 0; d < total_dims; d++) {
        std::copy(coords[d].begin(), coords[d].end(), p);
        p += shape[d];
      }
    }
    domain.set_geometry(geometry);
  }

  { // Dataset
    mgard::pb::Dataset &dataset = *header.mutable_dataset();
    if (dtype == data_type::Float) {
      dataset.set_type(mgard::pb::Dataset::FLOAT);
    } else if (dtype == data_type::Double) {
      dataset.set_type(mgard::pb::Dataset::DOUBLE);
    }
    dataset.set_dimension(1);
  }

  { // Error control
    mgard::pb::ErrorControl &error = *header.mutable_error_control();
    if (ebtype == error_bound_type::ABS) {
      error.set_mode(mgard::pb::ErrorControl::ABSOLUTE);
    } else if (ebtype == error_bound_type::REL) {
      error.set_mode(mgard::pb::ErrorControl::RELATIVE);
      error.set_norm_of_original_data(norm);
    }
    if (ntype == norm_type::L_Inf) {
      error.set_norm(mgard::pb::ErrorControl::L_INFINITY);
      error.set_s(s);
    } else {
      error.set_norm(mgard::pb::ErrorControl::S_NORM);
      error.set_s(s);
    }
    error.set_tolerance(tol);
  }

  { // Domain Decomposition
    mgard::pb::DomainDecomposition &domainDecomposition =
        *header.mutable_domain_decomposition();
    if (domain_decomposed) {
      domainDecomposition.set_method(mgard::pb::DomainDecomposition::BLOCK);
      for (uint64_t extent : domain_decomposition_block_shape) {
        domainDecomposition.add_block_shape(extent);
      }
    } else {
      domainDecomposition.set_method(
          mgard::pb::DomainDecomposition::NOOP_METHOD);
    }
  }

  { // Function Decomposition
    mgard::pb::FunctionDecomposition &function_decomposition =
        *header.mutable_function_decomposition();
    function_decomposition.set_transform(
        mgard::pb::FunctionDecomposition::MULTILEVEL_COEFFICIENTS);
    if (decomposition == decomposition_type::MultiDim) {
      function_decomposition.set_hierarchy(
          mgard::pb::FunctionDecomposition::MULTIDIMENSION_WITH_GHOST_NODES);
    } else if (decomposition == decomposition_type::SingleDim) {
      function_decomposition.set_hierarchy(
          mgard::pb::FunctionDecomposition::ONE_DIM_AT_A_TIME_WITH_GHOST_NODES);
    } else if (decomposition == decomposition_type::Hybrid) {
      function_decomposition.set_hierarchy(
          mgard::pb::FunctionDecomposition::HYBRID_HIERARCHY);
      // The hybrid hierarchy cannot be reconstructed from the hierarchy enum
      // alone: the local/global level counts fix the layout of the decomposed
      // buffer and the ROI map fixes the per-block quantization step. Without
      // them the decompressor would have to be handed the same parameters out
      // of band, and silently produce garbage when it was not.
      mgard::pb::HybridHierarchy &hybrid =
          *function_decomposition.mutable_hybrid_hierarchy();
      hybrid.set_num_local_levels(hybrid_num_local_levels);
      hybrid.set_num_global_levels(hybrid_num_global_levels);
      hybrid.set_local_block_size(hybrid_local_block_size);
      if (hybrid_projection_mode ==
          compression_projection_mode_type::Hierarchical) {
        hybrid.set_projection(mgard::pb::HybridHierarchy::HIERARCHICAL);
      } else if (hybrid_projection_mode ==
                 compression_projection_mode_type::Orthogonal) {
        hybrid.set_projection(mgard::pb::HybridHierarchy::ORTHOGONAL);
      } else {
        throw InvalidDataException(
            "cannot serialize an invalid hybrid projection mode.");
      }
      if (hybrid_enable_roi) {
        mgard::pb::RegionOfInterest &roi = *hybrid.mutable_region_of_interest();
        google::protobuf::RepeatedField<google::protobuf::uint64>
            &block_dimensions = *roi.mutable_block_dimensions();
        block_dimensions.Reserve(hybrid_roi_block_dimensions.size());
        for (uint64_t n : hybrid_roi_block_dimensions) {
          block_dimensions.Add(n);
        }
        SerializeROITolerances(hybrid_roi_tolerance_map, roi);
      }
    }
    // Not populated by MGARD-X: the reader rebuilds the hierarchy (and hence
    // the level count) from the shape, and under domain decomposition there is
    // no single value to record. Written for format compatibility only.
    function_decomposition.set_l_target(l_target);
    if (has_num_global_refactoring_level) {
      function_decomposition.set_num_global_refactoring_levels(
          num_global_refactoring_level);
    }
  }

  { // Quantization
    mgard::pb::Quantization &quantization = *header.mutable_quantization();
    if (otype == operation_type::Compression) {
      quantization.set_method(mgard::pb::Quantization::COEFFICIENTWISE_LINEAR);
      quantization.set_bin_widths(mgard::pb::Quantization::PER_COEFFICIENT);
      quantization.set_type(
          ::QuantizationTypeForWidth(sizeof(mgard_x::QUANTIZED_INT)));
      quantization.set_big_endian(big_endian<mgard_x::QUANTIZED_INT>());
      if (big_endian<mgard_x::QUANTIZED_INT>()) {
        etype = endiness_type::Big_Endian;
      } else {
        etype = endiness_type::Little_Endian;
      }
    } else {
      quantization.set_method(mgard::pb::Quantization::NOOP_QUANTIZATION);
    }
  }

  { // MDR
    mgard::pb::BitplaneEncoding &bitplane_encoding =
        *header.mutable_bitplane_encoding();
    if (otype == operation_type::MDR) {
      bitplane_encoding.set_method(
          mgard::pb::BitplaneEncoding::GROUPED_BITPLANE_ENCODING);
      bitplane_encoding.set_type(mgard::pb::BitplaneEncoding::INT32_T);
      bitplane_encoding.set_number_bitplanes(number_bitplanes);
      bitplane_encoding.set_big_endian(big_endian<std::int64_t>());
    } else {
      bitplane_encoding.set_method(
          mgard::pb::BitplaneEncoding::NOOP_BITPLANE_ENCODING);
    }
  }

  { // Encoding
    mgard::pb::Encoding &encoding = *header.mutable_encoding();
    encoding.set_preprocessor(mgard::pb::Encoding::NOOP_PREPROCESSOR);
    if (ltype == mgard_x::lossless_type::Huffman) {
      encoding.set_compressor(mgard::pb::Encoding::X_HUFFMAN);
      encoding.set_huffman_dictionary_size(huff_dict_size);
      encoding.set_huffman_block_size(huff_block_size);
    } else if (ltype == mgard_x::lossless_type::Huffman_LZ4) {
      encoding.set_compressor(mgard::pb::Encoding::X_HUFFMAN_LZ4);
      encoding.set_huffman_dictionary_size(huff_dict_size);
      encoding.set_huffman_block_size(huff_block_size);
    } else if (ltype == mgard_x::lossless_type::LZ4) {
      encoding.set_compressor(mgard::pb::Encoding::X_LZ4);
    } else if (ltype == mgard_x::lossless_type::Huffman_Zstd) {
      encoding.set_compressor(mgard::pb::Encoding::X_HUFFMAN_ZSTD);
      encoding.set_huffman_dictionary_size(huff_dict_size);
      encoding.set_huffman_block_size(huff_block_size);
    } else if (ltype == mgard_x::lossless_type::CPU_Lossless) {
      encoding.set_compressor(mgard::pb::Encoding::CPU_HUFFMAN_ZSTD);
    } else if (ltype == mgard_x::lossless_type::BlockDelta) {
      encoding.set_compressor(mgard::pb::Encoding::X_BLOCK_DELTA);
      encoding.set_block_delta_block_size(block_delta_block_size);
    } else if (ltype == mgard_x::lossless_type::SymbolRans) {
      encoding.set_compressor(mgard::pb::Encoding::X_SYMBOL_RANS);
      encoding.set_huffman_dictionary_size(huff_dict_size);
    } else if (ltype == mgard_x::lossless_type::ZeroRLE_Rans) {
      encoding.set_compressor(mgard::pb::Encoding::X_ZERORLE_RANS);
    }
  }

  { // Device
    mgard::pb::Device &device = *header.mutable_device();
    if (ptype == processor_type::X_SERIAL) {
      device.set_backend(mgard::pb::Device::X_SERIAL);
    } else if (ptype == processor_type::X_OPENMP) {
      device.set_backend(mgard::pb::Device::X_OPENMP);
    } else if (ptype == processor_type::X_CUDA) {
      device.set_backend(mgard::pb::Device::X_CUDA);
    } else if (ptype == processor_type::X_HIP) {
      device.set_backend(mgard::pb::Device::X_HIP);
    } else if (ptype == processor_type::X_SYCL) {
      device.set_backend(mgard::pb::Device::X_SYCL);
    }
  }

  // Serialize protobuf
  std::vector<SERIALIZED_TYPE> header_bytes = SerializeProtoBuf(header);
  uint64_t header_size = header_bytes.size();
  uint32_t header_crc32 = ComputeCRC32(header_bytes);

  metadata_size = SerializePreambleSize() + header_size;

  // start serializing
  std::vector<SERIALIZED_TYPE> serialized_data;
  serialized_data.reserve(metadata_size);

  ::SerializeSignature(serialized_data);
  ::Serialize(header_size, serialized_data);
  ::Serialize(header_crc32, serialized_data);
  ::SerializeBytes(header_bytes, serialized_data);

  return serialized_data;
}

uint64_t MetadataBase::DeserializeSize(
    std::vector<SERIALIZED_TYPE>::const_iterator &iter) {
  uint64_t header_size = 0;
  ::DeserializeSignature(iter);
  ::Deserialize(header_size, iter);

  header_size += SerializePreambleSize();

  return header_size;
}

void MetadataBase::DeserializeVersion(
    const std::vector<SERIALIZED_TYPE> &serialized_data) {
  SerializedIter iter = serialized_data.cbegin();
  DeserializeSize(iter);

  uint32_t header_crc32 = 0;
  ::Deserialize(header_crc32, iter);
  uint64_t offset = std::distance(serialized_data.begin(), iter);
  if (header_crc32 != ComputeCRC32(serialized_data, offset)) {
    throw InvalidDataException("header CRC32 mismatch.");
  }

  mgard::pb::Header header = DeserializeProtoBuf(serialized_data, offset);
  software_version[0] = header.mgard_version().major_();
  software_version[1] = header.mgard_version().minor_();
  software_version[2] = header.mgard_version().patch_();
  file_version[0] = header.file_format_version().major_();
  file_version[1] = header.file_format_version().minor_();
  file_version[2] = header.file_format_version().patch_();
}

bool MetadataBase::WrittenByV160OrEarlier() const {
  return software_version[0] < 1 ||
         (software_version[0] == 1 && software_version[1] <= 6);
}

void MetadataBase::Deserialize(
    const std::vector<SERIALIZED_TYPE> &serialized_data) {
  metadata_size = serialized_data.size();

  SerializedIter iter = serialized_data.cbegin();
  DeserializeSize(iter);

  uint32_t header_crc32 = 0;
  ::Deserialize(header_crc32, iter);

  // Actual serialized data is after header starting at index offset.
  uint64_t offset = std::distance(serialized_data.begin(), iter);

  if (header_crc32 != ComputeCRC32(serialized_data, offset)) {
    throw InvalidDataException("header CRC32 mismatch.");
  }

  mgard::pb::Header header = DeserializeProtoBuf(serialized_data, offset);

  { // Version Number
    const mgard::pb::VersionNumber mgard_version_number =
        header.mgard_version();
    software_version[0] = mgard_version_number.major_();
    software_version[1] = mgard_version_number.minor_();
    software_version[2] = mgard_version_number.patch_();
    if (software_version[0] > MGARD_VERSION_MAJOR) {
      throw InvalidDataException("MGARD version mismatch.");
    }

    const mgard::pb::VersionNumber format_version_number =
        header.file_format_version();
    file_version[0] = format_version_number.major_();
    file_version[1] = format_version_number.minor_();
    file_version[2] = format_version_number.patch_();
    if (file_version[0] > MGARD_FILE_VERSION_MAJOR) {
      throw InvalidDataException("MGARD file format version mismatch.");
    }
  }

  { // Domain
    const mgard::pb::Domain &domain = header.domain();
    const mgard::pb::CartesianGridTopology cartesian_grid_topology =
        domain.cartesian_grid_topology();
    total_dims = cartesian_grid_topology.dimension();
    const google::protobuf::RepeatedField<google::protobuf::uint64> shape_ =
        cartesian_grid_topology.shape();
    if (total_dims != shape_.size()) {
      throw InvalidDataException("grid shape does not match given dimension.");
    }
    shape = std::vector<uint64_t>(total_dims);
    std::copy(shape_.begin(), shape_.end(), shape.begin());

    const mgard::pb::Domain::Geometry geometry = domain.geometry();
    if (geometry == mgard::pb::Domain::UNIT_CUBE) {
      dstype = data_structure_type::Cartesian_Grid_Uniform;
    } else if (geometry == mgard::pb::Domain::EXPLICIT_CUBE) {
      dstype = data_structure_type::Cartesian_Grid_Non_Uniform;
      const mgard::pb::ExplicitCubeGeometry explicit_cube_geometry =
          domain.explicit_cube_geometry();
      const google::protobuf::RepeatedField<double> coordinates =
          explicit_cube_geometry.coordinates();
      uint64_t totel_len = 0;
      for (DIM d = 0; d < total_dims; d++)
        totel_len += shape[d];
      if (totel_len != coordinates.size()) {
        throw InvalidDataException(
            "mismatch between number of node coordinates and grid shape.");
      }
      using It = google::protobuf::RepeatedField<double>::const_iterator;
      It p = coordinates.begin();
      coords = std::vector<std::vector<double>>(total_dims);
      for (size_t d = 0; d < total_dims; d++) {
        const It q = p + shape[d];
        coords[d] = std::vector<double>(shape[d]);
        std::copy(p, q, coords[d].begin());
        p = q;
      }
      assert(p == coordinates.end());
    }
  }

  { // Dataset
    const mgard::pb::Dataset dataset = header.dataset();
    if (dataset.type() == mgard::pb::Dataset::FLOAT) {
      dtype = data_type::Float;
    } else if (dataset.type() == mgard::pb::Dataset::DOUBLE) {
      dtype = data_type::Double;
    }
    assert(dataset.dimension() == 1);
  }

  { // Error control
    const mgard::pb::ErrorControl error = header.error_control();
    if (error.mode() == mgard::pb::ErrorControl::ABSOLUTE) {
      ebtype = error_bound_type::ABS;
    } else if (error.mode() == mgard::pb::ErrorControl::RELATIVE) {
      ebtype = error_bound_type::REL;
      norm = error.norm_of_original_data();
    }

    if (error.norm() == mgard::pb::ErrorControl::L_INFINITY) {
      ntype = norm_type::L_Inf;
      s = std::numeric_limits<double>::infinity();
    } else if (error.norm() == mgard::pb::ErrorControl::S_NORM) {
      ntype = norm_type::L_2;
      s = error.s();
    }
    tol = error.tolerance();
  }

  { // Domain Decomposition
    const mgard::pb::DomainDecomposition domainDecomposition =
        header.domain_decomposition();
    // Every decomposition is read back as a block shape; the two layouts of
    // MGARD-X 1.6 and earlier are blocks of a particular shape.
    domain_decomposition_block_shape = shape;
    domain_decomposed = domainDecomposition.method() !=
                        mgard::pb::DomainDecomposition::NOOP_METHOD;
    if (domainDecomposition.method() == mgard::pb::DomainDecomposition::BLOCK) {
      if (domainDecomposition.block_shape_size() > 0) {
        if ((size_t)domainDecomposition.block_shape_size() != shape.size()) {
          throw InvalidDataException(
              "domain decomposition block shape does not match the data "
              "dimensions.");
        }
        for (size_t d = 0; d < shape.size(); d++) {
          domain_decomposition_block_shape[d] =
              domainDecomposition.block_shape(d);
        }
      } else {
        for (size_t d = 0; d < shape.size(); d++) {
          domain_decomposition_block_shape[d] =
              domainDecomposition.decomposition_size();
        }
      }
    } else if (domainDecomposition.method() ==
               mgard::pb::DomainDecomposition::MAX_DIMENSION) {
      uint64_t dim = domainDecomposition.decomposition_dimension();
      if (dim >= shape.size()) {
        throw InvalidDataException(
            "domain decomposition dimension out of range.");
      }
      domain_decomposition_block_shape[dim] =
          domainDecomposition.decomposition_size();
    } else if (domainDecomposition.method() !=
               mgard::pb::DomainDecomposition::NOOP_METHOD) {
      throw InvalidDataException(
          "unsupported domain decomposition method (variable-size "
          "decomposition is no longer supported).");
    }
  }

  { // Function Decomposition
    const mgard::pb::FunctionDecomposition function_decomposition =
        header.function_decomposition();
    assert(function_decomposition.transform() ==
           mgard::pb::FunctionDecomposition::MULTILEVEL_COEFFICIENTS);
    if (function_decomposition.hierarchy() ==
        mgard::pb::FunctionDecomposition::MULTIDIMENSION_WITH_GHOST_NODES) {
      decomposition = decomposition_type::MultiDim;
    } else if (function_decomposition.hierarchy() ==
               mgard::pb::FunctionDecomposition::
                   ONE_DIM_AT_A_TIME_WITH_GHOST_NODES) {
      decomposition = decomposition_type::SingleDim;
    } else if (function_decomposition.hierarchy() ==
               mgard::pb::FunctionDecomposition::HYBRID_HIERARCHY) {
      decomposition = decomposition_type::Hybrid;
      if (!function_decomposition.has_hybrid_hierarchy()) {
        // Written by a build from before the hybrid parameters were added to
        // the header. Its level counts and ROI map only ever existed in the
        // caller's Config, so we cannot reconstruct it here. Failing loudly is
        // the point: silently falling back to the Config defaults is what
        // produced wrong output without any error.
        throw InvalidDataException(
            "this file uses the hybrid hierarchy but predates the hybrid "
            "parameters being recorded in the header; it can only be "
            "decompressed by supplying the original local/global level counts "
            "and ROI tolerance map through Config.");
      }
      const mgard::pb::HybridHierarchy &hybrid =
          function_decomposition.hybrid_hierarchy();
      hybrid_num_local_levels = hybrid.num_local_levels();
      hybrid_num_global_levels = hybrid.num_global_levels();
      hybrid_local_block_size = hybrid.local_block_size();
      switch (hybrid.projection()) {
      case mgard::pb::HybridHierarchy::HIERARCHICAL:
        hybrid_projection_mode = compression_projection_mode_type::Hierarchical;
        break;
      case mgard::pb::HybridHierarchy::PROJECTION_UNSPECIFIED:
      case mgard::pb::HybridHierarchy::ORTHOGONAL:
        // A missing field denotes a legacy BlockMGARD stream, all of which
        // used the orthogonal basis.
        hybrid_projection_mode = compression_projection_mode_type::Orthogonal;
        break;
      default:
        throw InvalidDataException(
            "hybrid hierarchy records an unknown projection mode.");
      }
      if (hybrid_num_local_levels == 0 && hybrid_num_global_levels == 0) {
        throw InvalidDataException(
            "hybrid hierarchy records zero local and zero global levels.");
      }
      if (hybrid_local_block_size != MGARDX_HYBRID_LOCAL_BLOCK_SIZE) {
        throw InvalidDataException(
            "hybrid hierarchy was written with local block size " +
            std::to_string(hybrid_local_block_size) +
            ", but this build of MGARD-X only implements block size " +
            std::to_string(MGARDX_HYBRID_LOCAL_BLOCK_SIZE) + ".");
      }
      hybrid_enable_roi = hybrid.has_region_of_interest();
      if (hybrid_enable_roi) {
        const mgard::pb::RegionOfInterest &roi = hybrid.region_of_interest();
        const google::protobuf::RepeatedField<google::protobuf::uint64>
            &block_dimensions = roi.block_dimensions();
        hybrid_roi_block_dimensions.assign(block_dimensions.begin(),
                                           block_dimensions.end());
        hybrid_roi_tolerance_map = ::DeserializeROITolerances(roi);
        if (hybrid_roi_block_dimensions.size() != total_dims) {
          throw InvalidDataException(
              "ROI block grid has a different dimension than the data.");
        }
        std::size_t expected = 1;
        for (uint64_t n : hybrid_roi_block_dimensions) {
          expected *= n;
        }
        if (hybrid_roi_tolerance_map.size() != expected) {
          throw InvalidDataException(
              "ROI tolerance map holds " +
              std::to_string(hybrid_roi_tolerance_map.size()) +
              " entries but its block grid describes " +
              std::to_string(expected) + ".");
        }
      }
    } else {
      throw InvalidDataException(
          "this decomposition hierarchy mismatch the hierarchy used "
          "in MGARD-X.");
    }
    l_target = function_decomposition.l_target();
    has_num_global_refactoring_level =
        function_decomposition.has_num_global_refactoring_levels();
    num_global_refactoring_level =
        has_num_global_refactoring_level
            ? function_decomposition.num_global_refactoring_levels()
            : 0;
  }

  { // Quantization
    const mgard::pb::Quantization quantization = header.quantization();
    if (quantization.method() != mgard::pb::Quantization::NOOP_QUANTIZATION) {
      assert(quantization.bin_widths() ==
             mgard::pb::Quantization::PER_COEFFICIENT);
      // Checked rather than asserted: this decides how the payload is read, so
      // a mismatch must fail in release builds too. It fires when a file
      // written by a build with a different QUANTIZED_INT width is read.
      const std::size_t file_width =
          ::WidthForQuantizationType(quantization.type());
      if (file_width != sizeof(mgard_x::QUANTIZED_INT)) {
        throw InvalidDataException(
            "this file quantizes to " + std::to_string(file_width * 8) +
            "-bit integers, but this build of MGARD-X uses " +
            std::to_string(sizeof(mgard_x::QUANTIZED_INT) * 8) + "-bit.");
      }
      if (quantization.big_endian() != big_endian<mgard_x::QUANTIZED_INT>()) {
        throw InvalidDataException(
            "this file was written with the opposite endianness.");
      }
      if (big_endian<mgard_x::QUANTIZED_INT>()) {
        etype = endiness_type::Big_Endian;
      } else {
        etype = endiness_type::Little_Endian;
      }
    }
  }

  { // MDR
    const mgard::pb::BitplaneEncoding bitplane_encoding =
        header.bitplane_encoding();
    if (bitplane_encoding.method() !=
        mgard::pb::BitplaneEncoding::NOOP_BITPLANE_ENCODING) {
      number_bitplanes = bitplane_encoding.number_bitplanes();
      assert(bitplane_encoding.big_endian() == big_endian<std::int64_t>());
    }
  }

  {
    const mgard::pb::Quantization quantization = header.quantization();
    const mgard::pb::BitplaneEncoding bitplane_encoding =
        header.bitplane_encoding();
    if (quantization.method() != mgard::pb::Quantization::NOOP_QUANTIZATION &&
            bitplane_encoding.method() !=
                mgard::pb::BitplaneEncoding::NOOP_BITPLANE_ENCODING ||
        quantization.method() == mgard::pb::Quantization::NOOP_QUANTIZATION &&
            bitplane_encoding.method() ==
                mgard::pb::BitplaneEncoding::NOOP_BITPLANE_ENCODING) {
      throw InvalidDataException("cannot determine whether this is compressed "
                                 "or refactored data.");
    }
  }

  if (otype == operation_type::MDR) { // MDR
    mgard::pb::BitplaneEncoding &bitplane_encoding =
        *header.mutable_bitplane_encoding();
    bitplane_encoding.set_method(
        mgard::pb::BitplaneEncoding::GROUPED_BITPLANE_ENCODING);
    bitplane_encoding.set_type(mgard::pb::BitplaneEncoding::INT32_T);
    bitplane_encoding.set_number_bitplanes(number_bitplanes);
    bitplane_encoding.set_big_endian(big_endian<std::int64_t>());
  }

  { // Encoding
    const mgard::pb::Encoding encoding = header.encoding();
    if (encoding.compressor() == mgard::pb::Encoding::X_HUFFMAN) {
      ltype = mgard_x::lossless_type::Huffman;
      huff_dict_size = encoding.huffman_dictionary_size();
      huff_block_size = encoding.huffman_block_size();
    } else if (encoding.compressor() == mgard::pb::Encoding::X_HUFFMAN_LZ4) {
      ltype = mgard_x::lossless_type::Huffman_LZ4;
      huff_dict_size = encoding.huffman_dictionary_size();
      huff_block_size = encoding.huffman_block_size();
    } else if (encoding.compressor() == mgard::pb::Encoding::X_LZ4) {
      ltype = mgard_x::lossless_type::LZ4;
    } else if (encoding.compressor() == mgard::pb::Encoding::X_HUFFMAN_ZSTD) {
      ltype = mgard_x::lossless_type::Huffman_Zstd;
      huff_dict_size = encoding.huffman_dictionary_size();
      huff_block_size = encoding.huffman_block_size();
    } else if (encoding.compressor() == mgard::pb::Encoding::CPU_HUFFMAN_ZSTD) {
      ltype = mgard_x::lossless_type::CPU_Lossless;
    } else if (encoding.compressor() == mgard::pb::Encoding::X_BLOCK_DELTA) {
      ltype = mgard_x::lossless_type::BlockDelta;
      block_delta_block_size = encoding.block_delta_block_size();
    } else if (encoding.compressor() == mgard::pb::Encoding::X_SYMBOL_RANS) {
      ltype = mgard_x::lossless_type::SymbolRans;
      huff_dict_size = encoding.huffman_dictionary_size();
    } else if (encoding.compressor() == mgard::pb::Encoding::X_ZERORLE_RANS) {
      ltype = mgard_x::lossless_type::ZeroRLE_Rans;
    } else {
      throw InvalidDataException("unknown lossless compressor type.");
    }
  }

  { // Device
    const mgard::pb::Device device = header.device();
    if (device.backend() == mgard::pb::Device::X_SERIAL) {
      ptype = processor_type::X_SERIAL;
    } else if (device.backend() == mgard::pb::Device::X_OPENMP) {
      ptype = processor_type::X_OPENMP;
    } else if (device.backend() == mgard::pb::Device::X_CUDA) {
      ptype = processor_type::X_CUDA;
    } else if (device.backend() == mgard::pb::Device::X_HIP) {
      ptype = processor_type::X_HIP;
    } else if (device.backend() == mgard::pb::Device::X_SYCL) {
      ptype = processor_type::X_SYCL;
    } else if (device.backend() == mgard::pb::Device::CPU) {
      throw InvalidDataException("this data was not compressed with MGARD-X.");
    }
  }
}

} // namespace mgard_x
