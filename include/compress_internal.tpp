#include <cstdlib>

#include "compress.hpp"
#include "compressors.hpp"
#include "decompose.hpp"
#include "quantize.hpp"
#include "shuffle.hpp"

namespace mgard {

template <std::size_t N>
MemoryBuffer<const unsigned char> decompress_N(const pb::Header &header,
                                               void const *const data,
                                               const std::size_t size) {
  const pb::Dataset::Type dataset_type = read_dataset_type(header);
  switch (dataset_type) {
  case pb::Dataset::FLOAT:
    return decompress_N_Real<N, float>(header, data, size);
  case pb::Dataset::DOUBLE:
    return decompress_N_Real<N, double>(header, data, size);
  default:
    throw std::runtime_error("unrecognized dataset type");
  }
}

template <std::size_t N, typename Real>
MemoryBuffer<const unsigned char> decompress_N_Real(const pb::Header &header,
                                                    void const *const data,
                                                    const std::size_t size) {
  const pb::Domain &domain = header.domain();
  const CartesianGridTopology topology = read_topology(domain);
  const CartesianGridGeometry geometry = read_geometry(domain, topology);
  std::array<std::size_t, N> shape;
  std::copy(topology.shape.begin(), topology.shape.end(), shape.begin());
  if (geometry.uniform) {
    const TensorMeshHierarchy<N, Real> hierarchy(shape);
    return decompress(hierarchy, header, data, size);
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
    return decompress(hierarchy, header, data, size);
  }
}

template <std::size_t N, typename Real>
MemoryBuffer<const unsigned char>
decompress(const TensorMeshHierarchy<N, Real> &hierarchy,
           const pb::Header &header, void const *const data,
           const std::size_t size) {
  const ErrorControlParameters error_control = read_error_control(header);
  // TODO: Figure out how best to do this later.
  void *const data_compressed_ = new unsigned char[size];
  {
    unsigned char const *const p = static_cast<unsigned char const *>(data);
    unsigned char *const q = static_cast<unsigned char *>(data_compressed_);
    std::copy(p, p + size, q);
  }
  const CompressedDataset<N, Real> compressed(
      hierarchy, header, error_control.s, error_control.tolerance,
      data_compressed_, size);
  const DecompressedDataset<N, Real> decompressed = decompress(compressed);
  // TODO: Figure out how best to do this later.
  const std::size_t nbytes = hierarchy.ndof() * sizeof(Real);
  unsigned char *const data_decompressed_ = new unsigned char[nbytes];
  {
    unsigned char const *const p =
        reinterpret_cast<unsigned char const *>(decompressed.data());
    std::copy(p, p + nbytes, data_decompressed_);
  }
  return MemoryBuffer<const unsigned char>(data_decompressed_, nbytes);
}

} // namespace mgard
