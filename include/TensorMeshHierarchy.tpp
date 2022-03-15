#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include "format.hpp"

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
    : coordinates(coordinates), uniform(false) {
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

  const std::array<std::size_t, N> &SHAPE = shapes.back();
  for (std::size_t i = 0; i < N; ++i) {
    std::vector<std::vector<std::size_t>> &_indices_i = _indices.at(i);
    _indices_i.resize(L + 1);
    const std::size_t numerator = SHAPE.at(i) - 1;
    if (numerator) {
      for (std::size_t l = 0; l <= L; ++l) {
        std::vector<std::size_t> &_indices_i_l = _indices_i.at(l);
        const std::size_t n = shapes.at(l).at(i);
        _indices_i_l.resize(n);
        const std::size_t denominator = n - 1;
        for (std::size_t j = 0; j < n; ++j) {
          _indices_i_l.at(j) = (j * numerator) / denominator;
        }
      }
    } else {
      for (std::vector<std::size_t> &_indices_i_l : _indices_i) {
        _indices_i_l = {0};
      }
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

  _shuffled_indices.resize(ndof());
  std::size_t shuffled_index = 0;
  for (const TensorNode<N> node : ShuffledTensorNodeRange(*this, L)) {
    _shuffled_indices.at(unshuffled_index(node.multiindex)) = shuffled_index++;
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
    : TensorMeshHierarchy(shape, default_node_coordinates<N, Real>(shape)) {
  uniform = true;
}

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
  const std::vector<std::size_t> &_indices_dimension_l =
      _indices.at(dimension).at(l);
  std::size_t const *const data = _indices_dimension_l.data();
  const std::size_t n = _indices_dimension_l.size();
  return {.begin_ = data, .end_ = data + n};
}

template <std::size_t N, typename Real>
std::size_t TensorMeshHierarchy<N, Real>::unshuffled_index(
    const std::array<std::size_t, N> multiindex) const {
  const std::array<std::size_t, N> &SHAPE = shapes.back();
  std::size_t unshuffled_index_ = 0;
  for (std::size_t i = 0; i < N; ++i) {
    unshuffled_index_ *= SHAPE.at(i);
    unshuffled_index_ += multiindex.at(i);
  }
  return unshuffled_index_;
}

template <std::size_t N, typename Real>
std::size_t TensorMeshHierarchy<N, Real>::index(
    const std::array<std::size_t, N> multiindex) const {
  return _shuffled_indices.at(unshuffled_index(multiindex));
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
void TensorMeshHierarchy<N, Real>::populate_domain(pb::Header &header) const {
  const std::array<std::size_t, N> &SHAPE = shapes.back();
  pb::Domain &domain = *header.mutable_domain();

  domain.set_topology(pb::Domain::CARTESIAN_GRID);

  pb::CartesianGridTopology &cartesian_grid_topology =
      *domain.mutable_cartesian_grid_topology();
  cartesian_grid_topology.set_dimension(N);
  google::protobuf::RepeatedField<google::protobuf::uint64> &shape =
      *cartesian_grid_topology.mutable_shape();
  shape.Resize(N, 0);
  std::copy(SHAPE.begin(), SHAPE.end(), shape.mutable_data());

  pb::Domain::Geometry geometry;
  if (uniform) {
    geometry = pb::Domain::UNIT_CUBE;
  } else {
    geometry = pb::Domain::EXPLICIT_CUBE;
    pb::ExplicitCubeGeometry &explicit_cube_geometry =
        *domain.mutable_explicit_cube_geometry();
    google::protobuf::RepeatedField<double> &coordinates_ =
        *explicit_cube_geometry.mutable_coordinates();
    coordinates_.Resize(std::accumulate(SHAPE.begin(), SHAPE.end(), 0), 0);
    double *p = coordinates_.mutable_data();
    for (const std::vector<Real> &xs : coordinates) {
      std::copy(xs.begin(), xs.end(), p);
      p += xs.size();
    }
  }
  domain.set_geometry(geometry);
}

template <std::size_t N, typename Real>
void TensorMeshHierarchy<N, Real>::populate_dataset(pb::Header &header) const {
  pb::Dataset &dataset = *header.mutable_dataset();
  dataset.set_type(type_to_dataset_type<Real>());
  dataset.set_dimension(1);
}

template <std::size_t N, typename Real>
void TensorMeshHierarchy<N, Real>::populate_decomposition(
    pb::Header &header) const {
  pb::FunctionDecomposition &function_decomposition =
      *header.mutable_function_decomposition();
  function_decomposition.set_hierarchy(
      pb::FunctionDecomposition::POWER_OF_TWO_PLUS_ONE);
}

template <std::size_t N, typename Real>
void TensorMeshHierarchy<N, Real>::populate(pb::Header &header) const {
  populate_domain(header);
  populate_dataset(header);
  populate_decomposition(header);
}

template <std::size_t N, typename Real>
std::size_t TensorMeshHierarchy<N, Real>::ndof(const std::size_t l) const {
  check_mesh_index_bounds(l);
  const std::array<std::size_t, N> &shape = shapes.at(l);
  return std::accumulate(shape.begin(), shape.end(),
                         static_cast<std::size_t>(1),
                         std::multiplies<std::size_t>());
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
