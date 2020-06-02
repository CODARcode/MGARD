namespace mgard {

template <std::size_t N, typename Real>
TensorMeshLevel<N, Real>::TensorMeshLevel(
    const std::array<std::size_t, N> shape)
    : shape(shape) {}

template <std::size_t N, typename Real>
std::size_t TensorMeshLevel<N, Real>::ndof() const {
  std::size_t M = 1;
  for (const std::size_t m : shape) {
    M *= m;
  }
  return M;
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
