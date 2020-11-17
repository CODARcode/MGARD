#include "catch2/catch_test_macros.hpp"

#include <array>
#include <vector>

#include "TensorMeshHierarchy.hpp"
#include "shuffle.hpp"

#include "testing_utilities.hpp"

namespace {

template <std::size_t N, typename Real>
void test_shuffle(const std::array<std::size_t, N> shape,
                  const std::vector<Real> &expected) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(shape);
  const std::size_t ndof = hierarchy.ndof();
  std::vector<Real> u_(ndof);
  for (std::size_t i = 0; i < ndof; ++i) {
    u_.at(i) = static_cast<Real>(i);
  }
  Real const *const u = u_.data();

  std::vector<Real> v_(ndof);
  Real *const v = v_.data();

  mgard::shuffle(hierarchy, u, v);
  REQUIRE(v_ == expected);
}

} // namespace

TEST_CASE("shuffle behavior", "[shuffle]") {
  SECTION("1D") {
    const std::vector<float> expected = {0, 8, 4, 2, 6, 1, 3, 5, 7};
    test_shuffle<1, float>({9}, expected);
  }

  SECTION("2D") {
    const std::vector<float> expected = {0, 3,  8,  11, 20, 23, 1,  4,
                                         5, 7,  9,  12, 13, 15, 21, 2,
                                         6, 10, 14, 16, 17, 18, 19, 22};
    test_shuffle<2, float>({6, 4}, expected);
  }

  SECTION("3D") {
    const std::vector<double> expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    test_shuffle<3, double>({3, 2, 2}, expected);
  }
}

namespace {

template <std::size_t N, typename Real>
void test_unshuffle(const std::array<std::size_t, N> shape,
                    const std::vector<Real> &expected) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(shape);
  const std::size_t ndof = hierarchy.ndof();
  std::vector<Real> u_(ndof);
  for (std::size_t i = 0; i < ndof; ++i) {
    u_.at(i) = static_cast<Real>(i);
  }
  Real const *const u = u_.data();

  std::vector<Real> v_(ndof);
  Real *const v = v_.data();

  mgard::unshuffle(hierarchy, u, v);
  REQUIRE(v_ == expected);
}

} // namespace

TEST_CASE("unshuffle behavior", "[shuffle]") {
  SECTION("1D") {
    const std::vector<double> expected = {0, 3, 5, 2, 6, 4, 7, 1};
    test_unshuffle<1, double>({8}, expected);
  }

  SECTION("2D") {
    const std::vector<float> expected = {0, 4, 9, 1, 5, 6, 10, 7, 2, 8, 11, 3};
    test_unshuffle<2, float>({3, 4}, expected);
  }

  SECTION("3D") {
    const std::vector<double> expected = {0,  8,  1,  9,  10, 11, 2,  12, 3,
                                          13, 14, 15, 16, 17, 18, 19, 20, 21,
                                          4,  22, 5,  23, 24, 25, 6,  26, 7};
    test_unshuffle<3, double>({3, 3, 3}, expected);
  }
}

namespace {

template <typename Real>
void test_inversion_expected(const std::vector<Real> &u) {
  TrialTracker tracker;
  std::size_t expected = 0;
  for (const Real value : u) {
    tracker += value == static_cast<Real>(expected++);
  }
  REQUIRE(tracker);
}

template <std::size_t N, typename Real>
void test_inversion(const std::array<std::size_t, N> shape) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(shape);
  const std::size_t ndof = hierarchy.ndof();
  std::vector<Real> u_(ndof);
  for (std::size_t i = 0; i < ndof; ++i) {
    u_.at(i) = static_cast<Real>(i);
  }
  Real *const u = u_.data();

  std::vector<Real> v_(ndof);
  Real *const v = v_.data();

  mgard::shuffle(hierarchy, u, v);
  mgard::unshuffle(hierarchy, v, u);
  test_inversion_expected(u_);

  mgard::unshuffle(hierarchy, u, v);
  mgard::shuffle(hierarchy, v, u);
  test_inversion_expected(u_);
}

} // namespace

TEST_CASE("shuffle inversion", "[shuffle]") {
  SECTION("1D") {
    test_inversion<1, float>({27});
    // test_inversion<1, double>({33});
  }

  SECTION("2D") {
    test_inversion<2, float>({11, 21});
    test_inversion<2, double>({17, 15});
  }

  SECTION("3D") {
    test_inversion<3, float>({12, 12, 13});
    test_inversion<3, double>({8, 20, 13});
  }
}
