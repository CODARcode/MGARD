#include <stdexcept>

#include "format.hpp"

namespace mgard {

template <std::size_t N, typename Real>
CompressedDataset<N, Real>::CompressedDataset(
    const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
    const Real tolerance, void const *const data, const std::size_t size,
    const pb::Header &header)
    : hierarchy(hierarchy), s(s), tolerance(tolerance), header(header),
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
void CompressedDataset<N, Real>::write(std::ostream &ostream) const {
  write_metadata(ostream, header);
  ostream.write(static_cast<char const *>(data()), size());
}

template <std::size_t N, typename Real>
DecompressedDataset<N, Real>::DecompressedDataset(
    const CompressedDataset<N, Real> &compressed, Real const *const data)
    : hierarchy(compressed.hierarchy), s(compressed.s),
      tolerance(compressed.tolerance), header(compressed.header), data_(data) {}

template <std::size_t N, typename Real>
Real const *DecompressedDataset<N, Real>::data() const {
  return data_.get();
}

} // namespace mgard
