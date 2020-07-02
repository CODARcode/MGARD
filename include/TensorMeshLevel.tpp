#include <functional>
#include <numeric>

namespace mgard {

template <std::size_t N, typename Real>
TensorMeshLevel<N, Real>::TensorMeshLevel(
    const std::array<std::size_t, N> shape)
    : shape(shape) {}

template <std::size_t N, typename Real>
std::size_t TensorMeshLevel<N, Real>::ndof() const {
  return std::accumulate(shape.begin(), shape.end(), 1,
                         std::multiplies<Real>());
}

template <std::size_t N, typename Real>
bool operator==(const TensorMeshLevel<N, Real> &a,
                const TensorMeshLevel<N, Real> &b) {
  return a.shape == b.shape;
}

template <std::size_t N, typename Real>
bool operator!=(const TensorMeshLevel<N, Real> &a,
                const TensorMeshLevel<N, Real> &b) {
  return !operator==(a, b);
}

} // namespace mgard
