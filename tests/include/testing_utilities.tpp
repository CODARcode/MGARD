#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

#include <cassert>

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
  const typename U::size_type M = u.size();
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
