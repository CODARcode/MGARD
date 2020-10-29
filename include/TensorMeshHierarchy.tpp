#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "mgard_mesh.hpp"

namespace mgard {

template <std::size_t N, typename Real>
TensorMeshHierarchy<N, Real>::TensorMeshHierarchy(
    const TensorMeshLevel<N, Real> &mesh,
    const std::array<std::vector<Real>, N> &coordinates)
    : coordinates(coordinates) {
  for (std::size_t i = 0; i < N; ++i) {
    if (coordinates.at(i).size() != mesh.shape.at(i)) {
      throw std::invalid_argument("incorrect number of node coordinates given");
    }
  }

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

  for (std::size_t i = 0; i < N; ++i) {
    std::vector<std::size_t> &dobs = dates_of_birth.at(i);
    dobs.resize(mesh.shape.at(i));
    // Could be better to get all the levels' indices and iterate over
    // `dobs` once. More complicated and not necessary for now.
    for (std::size_t j = 0; j <= L; ++j) {
      const std::size_t l = L - j;
      for (const std::size_t index : indices(l, i)) {
        dobs.at(index) = l;
      }
    }
  }
}

namespace {

// TODO: This changes the previous default behavior, where the node spacing was
// set to `1` in every dimension with `std::iota`. Document somewhere.
template <std::size_t N, typename Real>
std::array<std::vector<Real>, N>
default_node_coordinates(const TensorMeshLevel<N, Real> &mesh) {
  std::array<std::vector<Real>, N> coordinates;
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t n = mesh.shape.at(i);
    std::vector<Real> &xs = coordinates.at(i);
    xs.resize(n);
    const Real h = n > 1 ? static_cast<Real>(1) / (n - 1) : 0;
    for (std::size_t j = 0; j < n; ++j) {
      xs.at(j) = j * h;
    }
  }
  return coordinates;
}

} // namespace

template <std::size_t N, typename Real>
TensorMeshHierarchy<N, Real>::TensorMeshHierarchy(
    const TensorMeshLevel<N, Real> &mesh)
    : TensorMeshHierarchy(mesh, default_node_coordinates(mesh)) {}

template <std::size_t N, typename Real>
TensorMeshHierarchy<N, Real>::TensorMeshHierarchy(
    const std::array<std::size_t, N> &shape,
    const std::array<std::vector<Real>, N> &coordinates)
    : TensorMeshHierarchy(TensorMeshLevel<N, Real>(shape), coordinates) {}

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
  check_dimension_index_bounds<N>(dimension);
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
TensorIndexRange
TensorMeshHierarchy<N, Real>::indices(const std::size_t l,
                                      const std::size_t dimension) const {
  check_mesh_index_bounds(l);
  check_dimension_index_bounds<N>(dimension);
  return TensorIndexRange(*this, l, dimension);
}

template <std::size_t N, typename Real>
std::size_t TensorMeshHierarchy<N, Real>::index(
    const std::array<std::size_t, N> multiindex) const {
  const std::size_t l = date_of_birth(multiindex);
  if (!l) {
    return number_nodes_before(l, multiindex);
  }
  return ndof(l - 1) + number_nodes_before(l, multiindex) -
         number_nodes_before(l - 1, multiindex);
}

template <std::size_t N, typename Real>
std::size_t TensorMeshHierarchy<N, Real>::number_nodes_before(
    const std::size_t l, const std::array<std::size_t, N> multiindex) const {
  check_mesh_index_bounds(l);
  const std::array<std::size_t, N> &SHAPE = meshes.back().shape;
  const std::array<std::size_t, N> &shape = meshes.at(l).shape;
  // Let `α` be the given node (its multiindex). A node (multiindex) `β` comes
  // before `α` if
  //     * `β_{1} < α_{1}` xor
  //     * `β_{1} = α_{1}` and `β_{2} < α_{2}` xor
  //     * …
  //     * `β_{1} = α_{1}`, …, `β_{N - 1} = α_{N - 1}` and `β_{N} < α_{N}`.
  // Consider one of these options: `β_{1} = α_{1}`, …, `β_{i - 1} = α_{i - 1}`
  // and `β_{i} < α_{i}`. Let `M_{k}` and `m_{k}` be the sizes of the finest and
  // `l`th meshes, respectively, in dimension `k`. `β` is unconstrained in
  // dimensions `i + 1` through `N`, so we start with a factor of `m_{i + 1} × ⋯
  // × m_{N}`. The values of `β_{1}`, …, `β_{i - 1}` are prescribed. `β_{i}` is
  // of the form `floor((j * (M_{i} - 1)) / (m_{i} - 1))`. We want `β_{i} <
  // α_{i}`, so `j` will go from zero up to some maximal value, after which
  // `β_{i} ≥ α_{i}`. The count of possible `j` values is the least `j` such
  // that `β_{i} ≥ α_{i}`. A bit of algebra shows that this is
  // `ceil((α_{i} * (m_{i} - 1)) / (M_{i} - 1))`. So, the count of possible
  // `β`s for this option is (assuming the constraints on `β_{1}`, …,
  // `β_{i - 1}` can be met – see below)
  // ```
  //   m_{i + 1} × ⋯ × m_{N} × ceil((α_{i} * (m_{i} - 1)) / (M_{i} - 1)).
  // ```
  // We compute the sum of these counts in the loop below, rearranging so that
  // we only have to multiply by each `m_{k}` once.
  //
  // One detail I missed: if `α` was introduced *after* the `l`th mesh, then it
  // may not be possible for `β_{k}` to equal `α_{k}`, since `β` must be present
  // in the `l`th mesh. Any option involving one of these 'impossible
  // constraints' will be knocked out and contribute nothing to the sum.
  std::size_t count = 0;
  bool impossible_constraint_encountered = false;
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t m = shape.at(i);
    const std::size_t M = SHAPE.at(i);
    // Notice that this has no effect in the first iteration.
    count *= m;
    if (impossible_constraint_encountered) {
      continue;
    }
    const std::size_t index = multiindex.at(i);
    const std::size_t numerator = index * (m - 1);
    const std::size_t denominator = M - 1;
    // We want to add `ceil(numerator / denominator)`. We can compute this term
    // using only integer divisions by adding one less than the denominator to
    // the numerator.
    count += (numerator + (denominator - 1)) / denominator;
    // The 'impossible constraint' will be encountered in the next iteration,
    // when we stipulate that `β_{i} = α_{i}` (current value of `i`).
    impossible_constraint_encountered =
        impossible_constraint_encountered || dates_of_birth.at(i).at(index) > l;
  }
  return count;
}

template <std::size_t N, typename Real>
std::size_t TensorMeshHierarchy<N, Real>::date_of_birth(
    const std::array<std::size_t, N> multiindex) const {
  // Initialized to zero, of course.
  std::size_t dob = std::numeric_limits<std::size_t>::min();
  for (std::size_t i = 0; i < N; ++i) {
    dob = std::max(dob, dates_of_birth.at(i).at(multiindex.at(i)));
  }
  return dob;
}

template <std::size_t N, typename Real>
Real &TensorMeshHierarchy<N, Real>::at(
    Real *const v, const std::array<std::size_t, N> multiindex) const {
  return v[index(multiindex)];
}

template <std::size_t N, typename Real>
const Real &TensorMeshHierarchy<N, Real>::at(
    Real const *const v, const std::array<std::size_t, N> multiindex) const {
  return v[index(multiindex)];
}

template <std::size_t N, typename Real>
UnshuffledTensorNodeRange<N, Real>
TensorMeshHierarchy<N, Real>::nodes(const std::size_t l) const {
  check_mesh_index_bounds(l);
  return UnshuffledTensorNodeRange<N, Real>(*this, l);
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

} // namespace mgard
