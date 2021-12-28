#include "compress.hpp"

#include <cstdlib>

#include "compressors.hpp"
#include "decompose.hpp"
#include "quantize.hpp"
#include "shuffle.hpp"

namespace mgard {

template <std::size_t N>
std::unique_ptr<unsigned char const []> decompress(void const *const data,
                                                   const std::size_t size,
                                                   const pb::Header &header) {
  const pb::Dataset::Type dataset_type = read_dataset_type(header);
  switch (dataset_type) {
  case pb::Dataset::FLOAT:
    return decompress<N, float>(data, size, header);
  case pb::Dataset::DOUBLE:
    return decompress<N, double>(data, size, header);
  default:
    throw std::runtime_error("unrecognized dataset type");
  }
}

template <std::size_t N, typename Real>
std::unique_ptr<unsigned char const []> decompress(void const *const data,
                                                   const std::size_t size,
                                                   const pb::Header &header) {
  const pb::Domain &domain = header.domain();
  const CartesianGridTopology topology = read_topology(domain);
  const CartesianGridGeometry geometry = read_geometry(domain, topology);
  std::array<std::size_t, N> shape;
  std::copy(topology.shape.begin(), topology.shape.end(), shape.begin());
  if (geometry.uniform) {
    const TensorMeshHierarchy<N, Real> hierarchy(shape);
    return decompress(data, size, header, hierarchy);
  } else {
    std::array<std::vector<Real>, N> coordinates_;
    // TODO: Could skip this if `std::is_same<Real, double>`.
    for (std::size_t i = 0; i < N; ++i) {
      const std::vector<double> &xs = geometry.coordinates.at(i);
      std::vector<Real> &xs_ = coordinates_.at(i);
      xs_.resize(xs.size());
      std::copy(xs.begin(), xs.end(), xs_.begin());
    }
    const TensorMeshHierarchy<N, Real> hierarchy(shape, coordinates_);
    return decompress(data, size, header, hierarchy);
  }
}

template <std::size_t N, typename Real>
std::unique_ptr<unsigned char const []> decompress(
    void const *const data, const std::size_t size, const pb::Header &header,
    const TensorMeshHierarchy<N, Real> &hierarchy) {
  const ErrorControlParameters error_control = read_error_control(header);
  // TODO: Figure out how best to do this later.
  void *const data_compressed_ = new unsigned char[size];
  {
    unsigned char const *const p = static_cast<unsigned char const *>(data);
    unsigned char *const q = static_cast<unsigned char *>(data_compressed_);
    std::copy(p, p + size, q);
  }
  const CompressedDataset<N, Real> compressed(hierarchy, error_control.s,
                                              error_control.tolerance,
                                              data_compressed_, size);
  const DecompressedDataset<N, Real> decompressed =
      decompress(compressed, header);
  // TODO: Figure out how best to do this later.
  const std::size_t nbytes = hierarchy.ndof() * sizeof(Real);
  unsigned char *const data_decompressed_ = new unsigned char[nbytes];
  {
    unsigned char const *const p =
        reinterpret_cast<unsigned char const *>(decompressed.data());
    std::copy(p, p + nbytes, data_decompressed_);
  }
  return std::unique_ptr<unsigned char const[]>(data_decompressed_);
}

template <std::size_t N, typename Real>
DecompressedDataset<N, Real> decompress(
    const CompressedDataset<N, Real> &compressed, const pb::Header &header) {
  const std::size_t ndof = compressed.hierarchy.ndof();

  const QuantizationParameters quantization = read_quantization(header);
  const std::size_t quantizedLen =
      ndof * quantization_type_size(quantization.type);

  // `quantized` will have the correct alignment for any quantization type. See
  // <https://en.cppreference.com/w/cpp/memory/c/malloc>.
  void *const quantized = std::malloc(quantizedLen);
  // TODO: Remove the `const_cast` if `decompress_memory_huffman` can be made to
  // take a `void const *`.
  decompress(const_cast<void *>(compressed.data()), compressed.size(),
             quantized, quantizedLen, header);

  Real *const dequantized =
      static_cast<Real *>(std::malloc(ndof * sizeof(*dequantized)));
  dequantize(compressed, quantized, dequantized, header);
  std::free(quantized);

  recompose(compressed.hierarchy, dequantized);
  Real *const v = new Real[ndof];
  unshuffle(compressed.hierarchy, dequantized, v);
  std::free(dequantized);

  return DecompressedDataset<N, Real>(compressed, v);
}

} // namespace mgard
