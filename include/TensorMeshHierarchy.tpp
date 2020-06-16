#include <stdexcept>
#include <type_traits>

#include "mgard_mesh.hpp"

namespace mgard {

template <std::size_t N, typename Real>
TensorMeshHierarchy<N, Real>::TensorMeshHierarchy(
    const TensorMeshLevel<N, Real> &mesh) {
  const Dimensions2kPlus1<N> dims(mesh.shape);
  L = dims.nlevel;
  if (!dims.is_2kplus1()) {
    ++L;
  }
  meshes.reserve(L + 1);
  // TODO: This will likely have to be revisited in the case that one of the
  // dimensions is `1`.
  // Find the dimensions of the coarsest mesh.
  std::array<std::size_t, N> shape = dims.rnded;
  for (std::size_t &n : shape) {
    n -= 1;
    n >>= dims.nlevel;
    n += 1;
  }
  for (std::size_t i = 0; i <= static_cast<std::size_t>(dims.nlevel); ++i) {
    meshes.push_back(TensorMeshLevel<N, Real>(shape));
    for (std::size_t &n : shape) {
      n -= 1;
      n <<= 1;
      n += 1;
    }
  }
  if (!dims.is_2kplus1()) {
    meshes.push_back(mesh);
  }
}

template <std::size_t N, typename Real>
TensorMeshHierarchy<N, Real>::TensorMeshHierarchy(
    const std::array<std::size_t, N> &shape)
    : TensorMeshHierarchy(TensorMeshLevel<N, Real>(shape)) {}

template <std::size_t N, typename Real>
bool operator==(const TensorMeshHierarchy<N, Real> &a,
                const TensorMeshHierarchy<N, Real> &b) {
  return a.meshes == b.meshes;
}

template <std::size_t N, typename Real>
bool operator!=(const TensorMeshHierarchy<N, Real> &a,
                const TensorMeshHierarchy<N, Real> &b) {
  return !operator==(a, b);
}

template <std::size_t N, typename Real>
std::size_t TensorMeshHierarchy<N, Real>::ndof() const {
  return ndof(L);
}

template <std::size_t N, typename Real>
std::size_t
TensorMeshHierarchy<N, Real>::stride(const std::size_t l,
                                     const std::size_t dimension) const {
  check_mesh_index_bounds(l);
  check_dimension_index_bounds(dimension);
  const std::array<std::size_t, N> &shape = meshes.back().shape;
  std::size_t n = 1;
  for (std::size_t i = dimension + 1; i < N; ++i) {
    n *= shape.at(i);
  }
  return n * stride_from_index_difference(L - l);
}

template <std::size_t N, typename Real>
std::size_t
TensorMeshHierarchy<N, Real>::l(const std::size_t index_difference) const {
  //! It's not a mesh index, but it'll satisfy the same bounds.
  check_mesh_index_bounds(index_difference);
  return L - index_difference;
}

template <std::size_t N, typename Real>
std::vector<std::size_t>
TensorMeshHierarchy<N, Real>::indices(const std::size_t l,
                                      const std::size_t dimension) const {
  check_mesh_index_bounds(l);
  check_dimension_index_bounds(dimension);
  std::vector<std::size_t> indices(meshes.at(l).shape.at(dimension));
  if (!Dimensions2kPlus1<N>(meshes.at(L).shape).is_2kplus1()) {
    throw std::invalid_argument("currently only dyadic grids are supported");
  }
  const std::size_t stride_ = stride_from_index_difference(L - l);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    indices.at(i) = stride_ * i;
  }
  return indices;
}

// TODO: The body of this function is copied from
// `LevelValues<N, Real>::iterator::operator*`. That function can call this one
// once `Dimensions2kPlus1<N>` and `TensorMeshHierarchy<N, Real>` are combined.
template <std::size_t N, typename Real>
Real &TensorMeshHierarchy<N, Real>::at(
    Real *const v, const std::array<std::size_t, N> multiindex) const {
  static_assert(N, "`N` must be nonzero to access entries");
  // TODO: Should also check that `meshes` is nonempty. Maybe do this in the
  // constructor. Could possibly have a member
  // `const TensorMeshLevel<N> &finest`.
  const std::array<std::size_t, N> &shape = meshes.back().shape;
  std::size_t index = multiindex.at(0);
  for (std::size_t i = 1; i < N; ++i) {
    index *= shape.at(i);
    index += multiindex.at(i);
  }
  return v[index];
}

template <std::size_t N, typename Real>
std::size_t TensorMeshHierarchy<N, Real>::ndof(const std::size_t l) const {
  check_mesh_index_bounds(l);
  return meshes.at(l).ndof();
}

template <std::size_t N, typename Real>
void TensorMeshHierarchy<N, Real>::check_mesh_index_bounds(
    const std::size_t l) const {
  //`l` is guaranteed to be nonnegative because `std::size_t` is unsigned.
  if (!(std::is_unsigned<std::size_t>::value && l <= L)) {
    throw std::out_of_range("mesh index out of range encountered");
  }
}

template <std::size_t N, typename Real>
void TensorMeshHierarchy<N, Real>::check_mesh_indices_nondecreasing(
    const std::size_t l, const std::size_t m) const {
  if (!(l <= m)) {
    throw std::invalid_argument("mesh indices should be nondecreasing");
  }
}

template <std::size_t N, typename Real>
void TensorMeshHierarchy<N, Real>::check_mesh_index_nonzero(
    const std::size_t l) const {
  if (!l) {
    throw std::out_of_range("mesh index should be nonzero");
  }
}

template <std::size_t N, typename Real>
void TensorMeshHierarchy<N, Real>::check_dimension_index_bounds(
    const std::size_t dimension) const {
  if (dimension >= N) {
    throw std::out_of_range("dimension index out of range encountered");
  }
}

} // namespace mgard
