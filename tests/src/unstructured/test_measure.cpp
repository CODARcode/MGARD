#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

#include <cstddef>

#include "blas.hpp"

#include "unstructured/measure.hpp"

TEST_CASE("`orient_2d`", "[measure]") {
  // Just basic tests. Relying mostly on `test_tri_measure`.
  const std::size_t N = 2;
  const double a[N] = {5, 3};
  const double b[N] = {3, 4};
  const double c[N] = {3, 3};
  REQUIRE(mgard::orient_2d(a, b, c) == 2);
  REQUIRE(mgard::orient_2d(b, a, c) == -2);
  REQUIRE(mgard::orient_2d(a, a, c) == 0);
}

TEST_CASE("`orient_3d`", "[measure]") {
  // Just basic tests. Relying mostly on `test_tet_measure`.
  const std::size_t N = 3;
  const double a[N] = {0, 23, 1};
  const double b[N] = {-8, 3, -2};
  const double c[N] = {8, 8, 0};
  const double d[N] = {-4, 22, -1};
  const double determinant = 428;
  REQUIRE(mgard::orient_3d(a, b, c, d) == determinant);
  REQUIRE(mgard::orient_3d(a, b, d, c) == -determinant);
}

TEST_CASE("edge measure", "[measure]") {
  const std::size_t N = 6;
  const double a[N] = {0, 0, 0, 1, -2, 3};
  const double base_length = mgard::edge_measure(a);
  REQUIRE(base_length == Catch::Approx(std::sqrt(14)));

  SECTION("edge measure respects translation invariance") {
    double b[N];
    blas::copy(N, a, b);
    const double shift[3] = {2, -10, 5};
    for (std::size_t i = 0; i < N; i += 3) {
      blas::axpy(3, 1.0, shift, b + i);
    }
    REQUIRE(mgard::edge_measure(b) == Catch::Approx(base_length));
  }

  SECTION("edge measure behaves properly under dilation") {
    const double factors[3] = {0.5, 2, 5};
    for (double factor : factors) {
      double b[N];
      blas::copy(6, a, b);
      // Scale both endpoints at once.
      blas::scal(N, factor, b);
      // Relying on `factor` being nonnegative here.
      const double expected = factor * base_length;
      REQUIRE(mgard::edge_measure(b) == Catch::Approx(expected));
    }
  }

  SECTION("edge measure invariant under permutation") {
    double b[N];
    blas::copy(3, a + 0, b + 3);
    blas::copy(3, a + 3, b + 0);
    REQUIRE(mgard::edge_measure(b) == Catch::Approx(base_length));
  }
}

TEST_CASE("triangle measure", "[measure]") {
  const std::size_t N = 9;
  const double a[N] = {3, 1, 1, 0, 2, 0, 0, 4, -2};
  const double base_area = mgard::tri_measure(a);
  {
    const double expected = 4.242640687119284;
    REQUIRE(base_area == Catch::Approx(expected));
  }

  SECTION("triangle measure respects translation invariance") {
    double b[N];
    blas::copy(N, a, b);
    const double shift[3] = {21, 21, 3};
    for (std::size_t i = 0; i < N; i += 3) {
      blas::axpy(3, 1.0, shift, b + i);
    }
    REQUIRE(mgard::tri_measure(b) == Catch::Approx(base_area));
  }

  SECTION("triangle measure behaves properly under dilation") {
    const double factors[3] = {0.01, 121, 920};
    for (double factor : factors) {
      double b[N];
      blas::copy(N, a, b);
      // Scale all vertices at once.
      blas::scal(N, factor, b);
      const double expected = factor * factor * base_area;
      REQUIRE(mgard::tri_measure(b) == Catch::Approx(expected));
    }
  }

  SECTION("triangle measure invariant under permutation") {
    double b[N];
    blas::copy(3, a + 0, b + 3);
    blas::copy(3, a + 3, b + 6);
    blas::copy(3, a + 6, b + 0);
    REQUIRE(mgard::tri_measure(b) == Catch::Approx(base_area));
  }
}

TEST_CASE("tetrahedron measure", "[measure]") {
  const std::size_t N = 12;
  const double a[N] = {0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, -4};
  const double base_volume = mgard::tet_measure(a);
  {
    const double expected = 8. / 6.;
    REQUIRE(base_volume == Catch::Approx(expected));
  }

  SECTION("tetrahedron measure respects translation invariance") {
    double b[N];
    blas::copy(N, a, b);
    const double shift[3] = {-3, 17, 92};
    for (std::size_t i = 0; i < N; i += 3) {
      blas::axpy(3, 1.0, shift, b + i);
    }
    REQUIRE(mgard::tet_measure(b) == Catch::Approx(base_volume));
  }

  SECTION("tetrahedron measure behaves properly under dilation") {
    const double factors[3] = {0.375, 1.8, 12};
    for (double factor : factors) {
      double b[N];
      blas::copy(N, a, b);
      // Scale all vertices at once.
      blas::scal(N, factor, b);
      // Relying on `factor` being nonnegative here.
      const double expected = factor * factor * factor * base_volume;
      REQUIRE(mgard::tet_measure(b) == Catch::Approx(expected));
    }
  }

  SECTION("tetrahedron measure invariant under permutation") {
    double b[N];
    blas::copy(3, a + 0, b + 6);
    blas::copy(3, a + 3, b + 0);
    blas::copy(3, a + 6, b + 9);
    blas::copy(3, a + 9, b + 3);
    REQUIRE(mgard::tet_measure(b) == Catch::Approx(base_volume));
  }
}
