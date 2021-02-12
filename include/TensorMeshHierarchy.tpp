#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace mgard {

static std::size_t log2(std::size_t n) {
  if (!n) {
    throw std::domain_error("can only take logarithm of positive numbers");
  }
  int exp;
  for (exp = -1; n; ++exp, n >>= 1)
    ;
  return static_cast<std::size_t>(exp);
}

//! Compute `log2(n - 1)`.
//!
//!\param n Size of the mesh in a particular dimension.
static std::size_t nlevel_from_size(const std::size_t n) {
  if (n == 0) {
    throw std::domain_error("size must be nonzero");
  }
  return log2(n - 1);
}

//! Compute `2^n + 1`.
//!
//!\param n Level index in a particular dimension (assuming a dyadic grid).
static std::size_t size_from_nlevel(const std::size_t n) {
  return (1 << n) + 1;
}

template <std::size_t N, typename Real>
TensorMeshHierarchy<N, Real>::TensorMeshHierarchy(
    const std::array<std::size_t, N> &shape,
    const std::array<std::vector<Real>, N> &coordinates)
    : coordinates(coordinates) {
  for (std::size_t i = 0; i < N; ++i) {
    if (coordinates.at(i).size() != shape.at(i)) {
      throw std::invalid_argument("incorrect number of node coordinates given");
    }
  }

  bool any_nonflat = false;
  bool any_nondyadic = false;
  // Sizes rounded down to the nearest dyadic (`2^k + 1`) number. For this
  // purpose, one is a dyadic number. Later we will use `shape_` for the shape
  // of the mesh currently being added to the hierarchy.
  std::array<std::size_t, N> shape_;
  std::size_t L_dyadic = std::numeric_limits<std::size_t>::max();
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t size = shape.at(i);
    if (size == 0) {
      throw std::domain_error(
          "dataset must have size larger than 0 in every dimension");
    } else if (size == 1) {
      shape_.at(i) = size;
      continue;
    } else {
      any_nonflat = true;
      const std::size_t l = nlevel_from_size(size);
      L_dyadic = std::min(L_dyadic, l);
      // Note the assignment to `shape_.at(i)` always happens.
      any_nondyadic =
          (shape_.at(i) = size_from_nlevel(l)) != size || any_nondyadic;
    }
  }
  if (!any_nonflat) {
    throw std::domain_error(
        "dataset must have size larger than 1 in some dimension");
  };
  L = any_nondyadic ? L_dyadic + 1 : L_dyadic;
  shapes.resize(L + 1);
  shapes.at(L) = shape;

  for (std::size_t &n : shape_) {
    --n;
    n >>= L_dyadic;
    ++n;
  }
  // From now on `shape_` will be the shape of the mesh currently being added to
  // the hierarchy.

  for (std::size_t i = 0; i + 1 <= L; ++i) {
    shapes.at(i) = shape_;
    for (std::size_t &n : shape_) {
      --n;
      n <<= 1;
      ++n;
    }
  }

  for (std::size_t i = 0; i < N; ++i) {
    std::vector<std::size_t> &dobs = dates_of_birth.at(i);
    dobs.resize(shape.at(i));
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

template <std::size_t N, typename Real>
std::array<std::vector<Real>, N>
default_node_coordinates(const std::array<std::size_t, N> &shape) {
  std::array<std::vector<Real>, N> coordinates;
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t n = shape.at(i);
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
    const std::array<std::size_t, N> &shape)
    : TensorMeshHierarchy(shape, default_node_coordinates<N, Real>(shape)) {}

template <std::size_t N, typename Real>
bool operator==(const TensorMeshHierarchy<N, Real> &a,
                const TensorMeshHierarchy<N, Real> &b) {
  return a.shapes == b.shapes;
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
  const std::array<std::size_t, N> &SHAPE = shapes.back();
  const std::array<std::size_t, N> &shape = shapes.at(l);
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
  //
  // That above assumes that `M_{i} ≠ 1`. In that case, it is impossible for
  // `β_{i}` to be less than `α_{i}` (both must be zero), so instead of
  // `ceil((α_{i} * (m_{i} - 1)) / (M_{i} - 1))` we get a factor of zero.
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
    // If the mesh is flat in this dimension (if `M == 1`), then `β_{i}` cannot
    // be less than `α_{i}` and so this case contributes nothing to the count.
    count += denominator ? (numerator + (denominator - 1)) / denominator : 0;
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
template <typename T>
PseudoArray<T>
TensorMeshHierarchy<N, Real>::on_nodes(T *const v, const std::size_t l) const {
  check_mesh_index_bounds(l);
  return PseudoArray<T>(v, ndof(l));
}

template <std::size_t N, typename Real>
PseudoArray<Real>
TensorMeshHierarchy<N, Real>::on_nodes(Real *const v,
                                       const std::size_t l) const {
  return on_nodes<Real>(v, l);
}

template <std::size_t N, typename Real>
PseudoArray<const Real>
TensorMeshHierarchy<N, Real>::on_nodes(Real const *const v,
                                       const std::size_t l) const {
  return on_nodes<const Real>(v, l);
}

template <std::size_t N, typename Real>
template <typename T>
PseudoArray<T>
TensorMeshHierarchy<N, Real>::on_new_nodes(T *const v,
                                           const std::size_t l) const {
  check_mesh_index_bounds(l);
  const std::size_t ndof_old = l ? ndof(l - 1) : 0;
  const std::size_t ndof_new = ndof(l) - ndof_old;
  return PseudoArray<T>(v + ndof_old, ndof_new);
}

template <std::size_t N, typename Real>
PseudoArray<Real>
TensorMeshHierarchy<N, Real>::on_new_nodes(Real *const v,
                                           const std::size_t l) const {
  return on_new_nodes<Real>(v, l);
}

template <std::size_t N, typename Real>
PseudoArray<const Real>
TensorMeshHierarchy<N, Real>::on_new_nodes(Real const *const v,
                                           const std::size_t l) const {
  return on_new_nodes<const Real>(v, l);
}

template <std::size_t N, typename Real>
template <typename T>
T &TensorMeshHierarchy<N, Real>::at(
    T *const v, const std::array<std::size_t, N> multiindex) const {
  return v[index(multiindex)];
}

template <std::size_t N, typename Real>
Real &TensorMeshHierarchy<N, Real>::at(
    Real *const v, const std::array<std::size_t, N> multiindex) const {
  return at<Real>(v, multiindex);
}

template <std::size_t N, typename Real>
const Real &TensorMeshHierarchy<N, Real>::at(
    Real const *const v, const std::array<std::size_t, N> multiindex) const {
  return at<const Real>(v, multiindex);
}

template <std::size_t N, typename Real>
std::size_t TensorMeshHierarchy<N, Real>::ndof(const std::size_t l) const {
  check_mesh_index_bounds(l);
  const std::array<std::size_t, N> &shape = shapes.at(l);
  return std::accumulate(shape.begin(), shape.end(), 1,
                         std::multiplies<Real>());
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
