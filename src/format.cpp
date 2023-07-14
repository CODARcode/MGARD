#include "format.hpp"

#include <climits>
#include <cstdint>

#include <fstream>
#include <new>
#include <numeric>
#include <stdexcept>

#include <zlib.h>

#include <google/protobuf/io/coded_stream.h>

#include "MGARDConfig.hpp"

#ifdef __NVCC__
// `utilities.hpp` wasn't included in the header.
#include "utilities.hpp"
#endif

namespace mgard {

std::uint_least64_t deserialize_header_size(
    const std::array<unsigned char, HEADER_SIZE_SIZE> &bytes) {
  static_assert(CHAR_BIT * HEADER_SIZE_SIZE >= 64,
                "deserialization array too small");
  return deserialize<std::uint_least64_t, HEADER_SIZE_SIZE>(bytes);
}

std::uint_least32_t deserialize_header_crc32(
    const std::array<unsigned char, HEADER_CRC32_SIZE> &bytes) {
  static_assert(CHAR_BIT * HEADER_SIZE_SIZE >= 32,
                "deserialization array too small");
  return deserialize<std::uint_least32_t, HEADER_CRC32_SIZE>(bytes);
}

std::array<unsigned char, HEADER_SIZE_SIZE>
serialize_header_size(std::uint_least64_t size) {
  return serialize<std::uint_least64_t, HEADER_SIZE_SIZE>(size);
}

std::array<unsigned char, HEADER_CRC32_SIZE>
serialize_header_crc32(std::uint_least64_t crc32) {
  return serialize<std::uint_least32_t, HEADER_CRC32_SIZE>(crc32);
}

template <> pb::Dataset::Type type_to_dataset_type<float>() {
  return pb::Dataset::FLOAT;
}

template <> pb::Dataset::Type type_to_dataset_type<double>() {
  return pb::Dataset::DOUBLE;
}

MemoryBuffer<unsigned char> quantization_buffer(const pb::Header &header,
                                                const std::size_t ndof) {
  static_assert(CHAR_BIT == 8, "unexpected number of bits in a byte");
  // Quantization type size.
  std::size_t qts;
  // Quantization type alignment.
  std::size_t qta;
  switch (header.quantization().type()) {
  case pb::Quantization::INT8_T:
    qts = sizeof(std::int8_t);
    qta = alignof(std::int8_t);
    break;
  case pb::Quantization::INT16_T:
    qts = sizeof(std::int16_t);
    qta = alignof(std::int16_t);
    break;
  case pb::Quantization::INT32_T:
    qts = sizeof(std::int32_t);
    qta = alignof(std::int32_t);
    break;
  case pb::Quantization::INT64_T:
    qts = sizeof(std::int64_t);
    qta = alignof(std::int64_t);
    break;
  default:
    throw std::runtime_error("unrecognized quantization type");
  }
  const std::size_t size = ndof * qts;

  return MemoryBuffer<unsigned char>(
      new (static_cast<std::align_val_t>(qta)) unsigned char[size], size);
}

namespace {

void set_version_number(pb::VersionNumber &version_number,
                        const google::protobuf::uint64 major_,
                        const google::protobuf::uint64 minor_,
                        const google::protobuf::uint64 patch_) {
  version_number.set_major_(major_);
  version_number.set_minor_(minor_);
  version_number.set_patch_(patch_);
}

} // namespace

void populate_version_numbers(pb::Header &header) {
  set_version_number(*header.mutable_mgard_version(), MGARD_VERSION_MAJOR,
                     MGARD_VERSION_MINOR, MGARD_VERSION_PATCH);
  set_version_number(*header.mutable_file_format_version(),
                     MGARD_FILE_VERSION_MAJOR, MGARD_FILE_VERSION_MINOR,
                     MGARD_FILE_VERSION_PATCH);
}

void populate_defaults(pb::Header &header) {
  populate_version_numbers(header);
  // `TensorMeshHierarchy::populate` sets all of the domain and dataset fields
  // and all of the decomposition fields except for `transform`.
  {
    pb::FunctionDecomposition &d = *header.mutable_function_decomposition();
    d.set_transform(pb::FunctionDecomposition::MULTILEVEL_COEFFICIENTS);
  }
  {
    pb::Quantization &q = *header.mutable_quantization();
    q.set_method(pb::Quantization::COEFFICIENTWISE_LINEAR);
    q.set_bin_widths(pb::Quantization::PER_COEFFICIENT);
    q.set_type(pb::Quantization::INT64_T);
    q.set_big_endian(big_endian<std::int64_t>());
  }
  {
    pb::Encoding &e = *header.mutable_encoding();
    e.set_preprocessor(pb::Encoding::SHUFFLE);
    e.set_compressor(
#ifdef MGARD_ZSTD
        pb::Encoding::CPU_HUFFMAN_ZSTD
#else
        pb::Encoding::CPU_HUFFMAN_ZLIB
#endif
    );
  }
  {
    pb::Device &device = *header.mutable_device();
    device.set_backend(pb::Device::CPU);
  }
}

BufferWindow::BufferWindow(void const *const data, const std::size_t size)
    : current(static_cast<unsigned char const *>(data)), end(current + size) {}

unsigned char const *BufferWindow::next(const std::size_t size) const {
  unsigned char const *const q = current + size;
  if (q > end) {
    throw std::runtime_error("next read will go past buffer endpoint");
  }
  return q;
}

void check_magic_number(BufferWindow &window) {
  unsigned char const *const next = window.next(SIGNATURE.size());
  if (not std::equal(window.current, next, SIGNATURE.begin())) {
    throw std::runtime_error("buffer does not start with MGARD magic bytes");
  }
  window.current = next;
}

std::uint_least64_t read_header_size(BufferWindow &window) {
  std::array<unsigned char, HEADER_SIZE_SIZE> bytes;
  unsigned char const *const next = window.next(bytes.size());
  std::copy(window.current, next, bytes.begin());
  window.current = next;
  return deserialize_header_size(bytes);
}

std::uint_least32_t read_header_crc32(BufferWindow &window) {
  std::array<unsigned char, HEADER_CRC32_SIZE> bytes;
  unsigned char const *const next = window.next(bytes.size());
  std::copy(window.current, next, bytes.begin());
  window.current = next;
  return deserialize_header_crc32(bytes);
}

namespace {

std::uint_least32_t compute_crc32(void const *const data,
                                  const std::size_t size) {
  // `crc32_z` takes a `size_t`.
  if (size > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("buffer is too large (size would overflow)");
  }
  uLong crc32_ = crc32_z(0, Z_NULL, 0);
  crc32_ = crc32_z(crc32_, static_cast<const Bytef *>(data), size);
  return crc32_;
}

} // namespace

void check_header_crc32(const BufferWindow &window,
                        const std::uint_least64_t header_size,
                        const std::uint_least32_t header_crc32) {
  // Check that the read will stay in the buffer.
  window.next(header_size);
  if (header_crc32 != compute_crc32(window.current, header_size)) {
    throw std::runtime_error("header CRC32 mismatch");
  }
}

pb::Header read_metadata(BufferWindow &window) {
  check_magic_number(window);
  const uint_least64_t header_size = read_header_size(window);
  const uint_least32_t header_crc32 = read_header_crc32(window);
  check_header_crc32(window, header_size, header_crc32);
  return read_header(window, header_size);
}

namespace {

template <typename T, std::size_t N>
void write(std::ostream &ostream, const std::array<T, N> &v) {
  ostream.write(reinterpret_cast<char const *>(v.data()), N * sizeof(T));
}

} // namespace

void write_metadata(std::ostream &ostream, const pb::Header &header) {
  write(ostream, SIGNATURE);

  const std::uint_least64_t header_size = header.ByteSize();
  write(ostream, serialize_header_size(header_size));

  unsigned char *const header_bytes = new unsigned char[header_size];
  header.SerializeToArray(header_bytes, header_size);

  write(ostream,
        serialize_header_crc32(compute_crc32(header_bytes, header_size)));

  ostream.write(reinterpret_cast<char const *>(header_bytes), header_size);
  delete[] header_bytes;
}

pb::Header read_header(BufferWindow &window,
                       const std::uint_least64_t header_size) {
  // The `CodedInputStream` constructor takes an `int`.
  if (header_size > std::numeric_limits<int>::max()) {
    throw std::runtime_error("header is too large (size would overflow)");
  }
  // Check that the read will stay in the buffer.
  unsigned char const *const next = window.next(header_size);
  mgard::pb::Header header;
  google::protobuf::io::CodedInputStream stream(
      static_cast<google::protobuf::uint8 const *>(window.current),
      header_size);
  if (not header.ParseFromCodedStream(&stream)) {
    throw std::runtime_error("header parsing encountered read or format error");
  }
  if (not stream.ConsumedEntireMessage()) {
    throw std::runtime_error("part of header left unparsed");
  }
  window.current = next;
  return header;
}

void check_mgard_version(const pb::Header &header) {
  const pb::VersionNumber &mgard_version = header.mgard_version();
  if (mgard_version.major_() > MGARD_VERSION_MAJOR) {
    throw std::runtime_error("MGARD version mismatch");
  }
}

void check_file_format_version(const pb::Header &header) {
  const pb::VersionNumber file_format_version = header.file_format_version();
  if (file_format_version.major_() > MGARD_FILE_VERSION_MAJOR) {
    throw std::runtime_error("MGARD file format version mismatch");
  }
}

CartesianGridTopology read_topology(const pb::Domain &domain) {
  CartesianGridTopology out;

  const pb::Domain::Topology topology = domain.topology();
  const pb::Domain::TopologyDefinitionCase top_def_case =
      domain.topology_definition_case();
  switch (topology) {
  case pb::Domain::CARTESIAN_GRID:
    switch (top_def_case) {
    case pb::Domain::TOPOLOGY_DEFINITION_NOT_SET:
      throw std::runtime_error("topology definition not set");
    case pb::Domain::kCartesianGridTopology: {
      const pb::CartesianGridTopology &cartesian_topology =
          domain.cartesian_grid_topology();
      const google::protobuf::uint64 dimension = cartesian_topology.dimension();
      const google::protobuf::RepeatedField<google::protobuf::uint64> &shape =
          cartesian_topology.shape();
      if (dimension != static_cast<google::protobuf::uint64>(shape.size())) {
        throw std::runtime_error("grid shape does not match given dimension");
      }
      out.dimension = dimension;
      out.shape.resize(out.dimension);
      std::copy(shape.begin(), shape.end(), out.shape.begin());
      break;
    }
    default:
      throw std::runtime_error("unrecognized topology definition");
    }
    break;
  default:
    throw std::runtime_error("unrecognized domain topology");
  }

  return out;
}

namespace {

std::vector<std::vector<double>>
read_explicit_coordinates(const CartesianGridTopology &topology,
                          const pb::ExplicitCubeGeometry &geometry) {
  const std::vector<std::size_t> &shape = topology.shape;
  const google::protobuf::RepeatedField<double> &coordinates =
      geometry.coordinates();
  if (coordinates.size() != std::accumulate(shape.begin(), shape.end(), 0)) {
    throw std::runtime_error(
        "mismatch between number of node coordinates and grid shape");
  }

  const std::size_t N = shape.size();
  std::vector<std::vector<double>> out(N);
  using It = google::protobuf::RepeatedField<double>::const_iterator;
  It p = coordinates.begin();
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t n = shape.at(i);
    std::vector<double> &xs = out.at(i);
    xs.resize(n);
    const It q = p + n;
    std::copy(p, q, xs.begin());
    p = q;
  }
  assert(p == coordinates.end());
  return out;
}

} // namespace

CartesianGridGeometry read_geometry(const pb::Domain &domain,
                                    const CartesianGridTopology &topology) {
  CartesianGridGeometry out;

  const pb::Domain::Geometry geometry = domain.geometry();
  const pb::Domain::GeometryDefinitionCase geo_def_case =
      domain.geometry_definition_case();
  switch (geometry) {
  case pb::Domain::UNIT_CUBE:
    out.uniform = true;
    switch (geo_def_case) {
    case pb::Domain::GEOMETRY_DEFINITION_NOT_SET:
      break;
    default:
      throw std::runtime_error("geometry definition unexpectedly set");
    }
    break;
  case pb::Domain::EXPLICIT_CUBE:
    out.uniform = false;
    switch (geo_def_case) {
    case pb::Domain::GEOMETRY_DEFINITION_NOT_SET:
      throw std::runtime_error("topology definition not set");
    case pb::Domain::kExplicitCubeGeometry: {
      const pb::ExplicitCubeGeometry &ecg = domain.explicit_cube_geometry();
      out.coordinates = read_explicit_coordinates(topology, ecg);
      break;
    }
    case pb::Domain::kExplicitCubeFilename: {
      std::ifstream stream(domain.explicit_cube_filename(),
                           std::ios_base::binary);
      pb::ExplicitCubeGeometry ecg;
      if (not ecg.ParseFromIstream(&stream)) {
        throw std::runtime_error(
            "coordinates parsing encountered read or format error");
      }
      out.coordinates = read_explicit_coordinates(topology, ecg);
      break;
    }
    default:
      throw std::runtime_error("unrecognized geometry definition");
    }
    break;
  default:
    throw std::runtime_error("unrecognized domain geometry");
  }
  return out;
}

pb::Dataset::Type read_dataset_type(const pb::Header &header) {
  const pb::Dataset &dataset = header.dataset();
  if (dataset.dimension() != 1) {
    throw std::runtime_error("unsupported dataset dimension");
  }
  const pb::Dataset::Type type = dataset.type();
  switch (type) {
  case pb::Dataset::FLOAT:
  case pb::Dataset::DOUBLE:
    break;
  default:
    throw std::runtime_error("unrecognized dataset type");
  }
  return type;
}

ErrorControlParameters read_error_control(const pb::Header &header) {
  ErrorControlParameters out;

  const pb::ErrorControl &error_control = header.error_control();

  const double tolerance_raw = error_control.tolerance();
  switch (error_control.mode()) {
  case pb::ErrorControl::ABSOLUTE:
    out.tolerance = tolerance_raw;
    break;
  case pb::ErrorControl::RELATIVE:
    out.tolerance = tolerance_raw * error_control.norm_of_original_data();
    break;
  default:
    throw std::runtime_error("unrecognized error control mode");
  }

  switch (error_control.norm()) {
  case pb::ErrorControl::L_INFINITY:
    out.s = std::numeric_limits<double>::infinity();
    break;
  case pb::ErrorControl::S_NORM:
    out.s = error_control.s();
    break;
  default:
    throw std::runtime_error("unrecognized error control norm");
  }

  return out;
}

void check_decomposition_parameters(const pb::Header &header) {
  const pb::FunctionDecomposition &function_decomposition =
      header.function_decomposition();

  switch (function_decomposition.transform()) {
  case pb::FunctionDecomposition::MULTILEVEL_COEFFICIENTS:
    break;
  default:
    throw std::runtime_error("unrecognized decomposition transform");
  }

  switch (function_decomposition.hierarchy()) {
  case pb::FunctionDecomposition::POWER_OF_TWO_PLUS_ONE:
    break;
  case pb::FunctionDecomposition::MULTIDIMENSION_WITH_GHOST_NODES:
  case pb::FunctionDecomposition::ONE_DIM_AT_A_TIME_WITH_GHOST_NODES:
    throw std::runtime_error("ghost nodes not yet supported in CPU version");
  default:
    throw std::runtime_error("unrecognized decomposition hierarchy");
  }
}

QuantizationParameters read_quantization(const pb::Header &header) {
  const pb::Quantization &quantization = header.quantization();
  QuantizationParameters out;

  switch (quantization.method()) {
  case pb::Quantization::COEFFICIENTWISE_LINEAR:
    break;
  default:
    throw std::runtime_error("unrecognized quantization method");
  }

  switch (quantization.bin_widths()) {
  case pb::Quantization::PER_COEFFICIENT:
    break;
  case pb::Quantization::PER_LEVEL:
    throw std::runtime_error(
        "per level quantization not yet supported in CPU version");
  default:
    throw std::runtime_error("unrecognized quantization bin width mode");
  }

  const pb::Quantization::Type type = quantization.type();
  switch (type) {
  case pb::Quantization::INT8_T:
  case pb::Quantization::INT16_T:
  case pb::Quantization::INT32_T:
  case pb::Quantization::INT64_T:
    out.type = type;
    break;
  default:
    throw std::runtime_error("unrecognized quantization output type");
  }

  out.big_endian = quantization.big_endian();

  return out;
}

pb::Encoding::Compressor read_encoding_compressor(const pb::Header &header) {
  const pb::Encoding &encoding = header.encoding();

  switch (encoding.preprocessor()) {
  case pb::Encoding::SHUFFLE:
    break;
  default:
    throw std::runtime_error("unrecognized encoding preprocessor");
  }

  const pb::Encoding::Compressor compressor = encoding.compressor();
  switch (compressor) {
  case pb::Encoding::X_HUFFMAN:
  case pb::Encoding::X_HUFFMAN_LZ4:
  case pb::Encoding::X_HUFFMAN_ZSTD:
    throw std::runtime_error(
        "X_HUFFMAN, X_HUFFMAN_LZ4, and X_HUFFMAN_ZSTD compressors not "
        "yet supported in CPU version");
  case pb::Encoding::NOOP_COMPRESSOR:
  case pb::Encoding::CPU_HUFFMAN_ZLIB:
    break;
  case pb::Encoding::CPU_HUFFMAN_ZSTD:
#ifdef MGARD_ZSTD
    break;
#else
    throw std::runtime_error("MGARD compiled without ZSTD support");
#endif
  default:
    throw std::runtime_error("unrecognized encoding compressor");
  }

  return compressor;
}

} // namespace mgard
