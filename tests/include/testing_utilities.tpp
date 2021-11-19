#include "catch2/catch.hpp"

#include <cassert>

#include <stdexcept>

template <typename T, typename U, typename SizeType>
void require_vector_equality(T p, U q, const SizeType N, const double margin) {
  bool all_close = true;
  for (SizeType i = 0; i < N; ++i) {
    all_close = all_close && *p++ == Catch::Approx(*q++).margin(margin);
  }
  REQUIRE(all_close);
}

template <typename T, typename U>
void require_vector_equality(const T &t, const U &u, const double margin) {
  const typename T::size_type N = t.size();
  [[maybe_unused]] const typename U::size_type M = u.size();
  assert(N == M);
  require_vector_equality(t.begin(), u.begin(), N, margin);
}

template <std::size_t N, typename Real>
std::array<Real, N>
coordinates(const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
            const mgard::TensorNode<N> &node) {
  std::array<Real, N> coordinates;
  for (std::size_t i = 0; i < N; ++i) {
    coordinates.at(i) = hierarchy.coordinates.at(i).at(node.multiindex.at(i));
  }
  return coordinates;
}

template <std::size_t N, std::size_t M, typename Real>
mgard::TensorMeshHierarchy<M, Real>
make_flat_hierarchy(const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
                    const std::array<std::size_t, M> shape) {
  const std::array<std::size_t, N> &SHAPE = hierarchy.shapes.back();
  const std::array<std::vector<Real>, N> &COORDINATES = hierarchy.coordinates;
  std::array<std::vector<Real>, M> coordinates;
  std::size_t j = 0;
  for (std::size_t i = 0; i < M; ++i) {
    if (j < N && shape.at(i) == SHAPE.at(j)) {
      coordinates.at(i) = COORDINATES.at(j);
      ++j;
    } else if (shape.at(i) == 1) {
      coordinates.at(i) = {0};
    } else {
      throw std::invalid_argument("desired mesh cannot be obtained by adding "
                                  "'flat' dimensions to input mesh");
    }
  }
  if (j != N) {
    throw std::invalid_argument(
        "desired mesh does not use all dimensions of input mesh");
  }
  return mgard::TensorMeshHierarchy<M, Real>(shape, coordinates);
}
