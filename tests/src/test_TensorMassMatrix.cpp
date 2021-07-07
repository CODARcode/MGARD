#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "testing_random.hpp"
#include "testing_utilities.hpp"

#include "TensorMassMatrix.hpp"
#include "TensorMeshHierarchy.hpp"
#include "blas.hpp"
#include "shuffle.hpp"
#include "utilities.hpp"

TEST_CASE("constituent mass matrices", "[TensorMassMatrix]") {
  SECTION("1D and default spacing") {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({9});
    const std::size_t ndof = 9;
    const std::array<float, ndof> u_ = {-2, 1, 1, 8, 3, -2, -7, -4, 0};
    std::array<float, ndof> v_;
    std::array<float, ndof> buffer_;
    float const *const u = u_.data();
    float *const v = v_.data();
    float *const buffer = buffer_.data();
    {
      const std::size_t l = 3;
      const std::size_t dimension = 0;
      const mgard::ConstituentMassMatrix<1, float> M(hierarchy, l, dimension);
      mgard::shuffle(hierarchy, u, v);
      M({0}, v);
      mgard::unshuffle(hierarchy, v, buffer);
      std::array<float, ndof> expected = {-3, 3, 13, 36, 18, -12, -34, -23, -4};
      blas::scal(ndof, static_cast<float>(1) / 48, expected.data());
      TrialTracker tracker;
      for (std::size_t i = 0; i < ndof; ++i) {
        tracker += buffer_.at(i) == Catch::Approx(expected.at(i));
      }
      REQUIRE(tracker);
    }
    {
      const std::size_t l = 1;
      const std::size_t dimension = 0;
      const mgard::ConstituentMassMatrix<1, float> M(hierarchy, l, dimension);
      mgard::shuffle(hierarchy, u, v);
      M({0}, v);
      mgard::unshuffle(hierarchy, v, buffer);
      std::array<float, ndof> expected = {-1, 1, 1, 8, 10, -2, -7, -4, 3};
      for (std::size_t i = 0; i < 3; ++i) {
        expected.at(4 * i) /= 12;
      }
      TrialTracker tracker;
      for (std::size_t i = 0; i < ndof; ++i) {
        tracker += buffer_.at(i) == Catch::Approx(expected.at(i));
      }
      REQUIRE(tracker);
    }
  }

  SECTION("1D and nondyadic") {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({7});
    const std::size_t ndof = 7;
    const std::array<float, ndof> u_ = {-1, 8, -9, -9, -1, -10, 6};
    std::array<float, ndof> v_;
    std::array<float, ndof> buffer_;
    float const *const u = u_.data();
    float *const v = v_.data();
    float *const buffer = buffer_.data();

    const std::size_t dimension = 0;
    const std::array<std::array<float, ndof>, 4> expecteds = {
        {{{2. / 3, 8, -9, -9, -1, -10, 11. / 6}},
         {{-11. / 12, 8, -9, -31. / 12, -1, -10, 0.25}},
         {{1. / 6, 29. / 36, -9, -39. / 36, -1. / 12, -10, 11. / 18}},
         {{1. / 6, 11. / 18, -37. / 36, -23. / 18, -23. / 36, -35. / 36,
           1. / 18}}}};
    for (std::size_t l = 0; l < 4; ++l) {
      const mgard::ConstituentMassMatrix<1, float> M(hierarchy, l, dimension);
      mgard::shuffle(hierarchy, u, v);
      M({0}, v);
      mgard::unshuffle(hierarchy, v, buffer);
      TrialTracker tracker;
      const std::array<float, ndof> &expected = expecteds.at(l);
      for (std::size_t i = 0; i < ndof; ++i) {
        tracker += buffer_.at(i) == Catch::Approx(expected.at(i));
      }
      REQUIRE(tracker);
    }
  }

  SECTION("2D and custom spacing") {
    const mgard::TensorMeshHierarchy<2, double> hierarchy(
        {5, 5}, {{{0, 0.1, 0.5, 0.75, 1}, {0, 0.65, 0.70, 0.75, 1}}});
    const std::size_t ndof = 5 * 5;
    const std::array<double, ndof> u_ = {8,  -3, 1,  5,  10, 9,  -10, -3, 8,
                                         10, -3, 6,  -7, -3, 3,  3,   -9, 0,
                                         -1, 8,  -6, 7,  1,  -2, 10};
    std::array<double, ndof> v_;
    std::array<double, ndof> buffer_;
    double const *const u = u_.data();
    double *const v = v_.data();
    double *const buffer = buffer_.data();
    {
      const std::size_t l = 2;
      const std::size_t dimension = 0;
      const mgard::ConstituentMassMatrix<2, double> M(hierarchy, l, dimension);
      mgard::shuffle(hierarchy, u, v);
      M({0, 0}, v);
      M({0, 3}, v);
      mgard::unshuffle(hierarchy, v, buffer);
      const std::array<double, ndof> expected = {2.5 / 6,
                                                 -3,
                                                 1,
                                                 1.8 / 6,
                                                 10,
                                                 2.6 / 6 + 6.0 / 6,
                                                 -10,
                                                 -3,
                                                 2.1 / 6 + 5.2 / 6,
                                                 10,
                                                 1.2 / 6 + -0.75 / 6,
                                                 6,
                                                 -7,
                                                 0.8 / 6 + -1.75 / 6,
                                                 3,
                                                 0.75 / 6 + 0.0 / 6,
                                                 -9,
                                                 0,
                                                 -1.25 / 6 + -1.0 / 6,
                                                 8,
                                                 -2.25 / 6,
                                                 7,
                                                 1,
                                                 -1.25 / 6,
                                                 10};
      TrialTracker tracker;
      for (std::size_t i = 0; i < ndof; ++i) {
        tracker += buffer_.at(i) == Catch::Approx(expected.at(i));
      }
      REQUIRE(tracker);
    }
    {
      const std::size_t l = 2;
      const std::size_t dimension = 1;
      const mgard::ConstituentMassMatrix<2, double> M(hierarchy, l, dimension);
      mgard::shuffle(hierarchy, u, v);
      M({1, 0}, v);
      M({2, 0}, v);
      mgard::unshuffle(hierarchy, v, buffer);
      const std::array<double, ndof> expected = {8,
                                                 -3,
                                                 1,
                                                 5,
                                                 10,
                                                 5.2 / 6,
                                                 -7.15 / 6 + -1.15 / 6,
                                                 -0.8 / 6 + 0.1 / 6,
                                                 0.65 / 6 + 6.5 / 6,
                                                 7.0 / 6,
                                                 0.0 / 6,
                                                 5.85 / 6 + 0.25 / 6,
                                                 -0.4 / 6 + -0.85 / 6,
                                                 -0.65 / 6 + -0.75 / 6,
                                                 0.75 / 6,
                                                 3,
                                                 -9,
                                                 0,
                                                 -1,
                                                 8,
                                                 -6,
                                                 7,
                                                 1,
                                                 -2,
                                                 10};
      TrialTracker tracker;
      for (std::size_t i = 0; i < ndof; ++i) {
        // Changed the margin because we were getting a tiny error at index 10
        // (where the exact value is zero).
        tracker += buffer_.at(i) == Catch::Approx(expected.at(i)).margin(1e-15);
      }
      REQUIRE(tracker);
    }
    {
      const std::size_t l = 1;
      const std::size_t dimension = 1;
      const mgard::ConstituentMassMatrix<2, double> M(hierarchy, l, dimension);
      mgard::shuffle(hierarchy, u, v);
      M({4, 0}, v);
      mgard::unshuffle(hierarchy, v, buffer);
      const std::array<double, 5> expected = {-7.7 / 6, 7, -2.8 / 6 + 3.6 / 6,
                                              -2, 6.3 / 6};
      TrialTracker tracker;
      for (std::size_t i = 0; i < ndof - 5; ++i) {
        tracker += buffer_.at(i) == u_.at(i);
      }
      for (std::size_t i = ndof - 5; i < ndof; ++i) {
        tracker += buffer_.at(i) == Catch::Approx(expected.at(i - 20));
      }
      REQUIRE(tracker);
    }
  }
}

TEST_CASE("tensor product mass matrices", "[TensorMassMatrix]") {
  const mgard::TensorMeshHierarchy<2, double> hierarchy(
      {3, 3}, {{{0, 0.5, 1}, {1, 1.25, 2}}});
  const std::size_t ndof = 3 * 3;
  const std::array<double, ndof> u_ = {2, 3, -9, -2, 6, 5, 2, -1, 5};
  std::array<double, ndof> v_;
  std::array<double, ndof> buffer_;
  double const *const u = u_.data();
  double *const v = v_.data();
  double *const buffer = buffer_.data();
  {
    const std::size_t l = 1;
    const mgard::TensorMassMatrix<2, double> M(hierarchy, l);
    mgard::shuffle(hierarchy, u, v);
    M(v);
    mgard::unshuffle(hierarchy, v, buffer);
    const std::array<double, ndof> expected = {0.05555555555555556,
                                               0.20486111111111108,
                                               -0.14583333333333334,
                                               0.0625,
                                               0.875,
                                               0.6041666666666666,
                                               0.027777777777777776,
                                               0.2743055555555555,
                                               0.3541666666666667};
    TrialTracker tracker;
    for (std::size_t i = 0; i < ndof; ++i) {
      tracker += buffer_.at(i) == Catch::Approx(expected.at(i));
    }
    REQUIRE(tracker);
  }
  {
    const std::size_t l = 0;
    const mgard::TensorMassMatrix<2, double> M(hierarchy, l);
    mgard::shuffle(hierarchy, u, v);
    M(v);
    mgard::unshuffle(hierarchy, v, buffer);
    const std::array<double, ndof> expected = {
        -0.02777777777777779, 3.0,  -0.5555555555555555, -2.0, 6.0, 5.0,
        0.3611111111111111,   -1.0, 0.22222222222222224};
    TrialTracker tracker;
    for (std::size_t i = 0; i < ndof; ++i) {
      tracker += buffer_.at(i) == Catch::Approx(expected.at(i));
    }
    REQUIRE(tracker);
  }
}

namespace {

template <std::size_t N, typename Real>
void exhaustive_constituent_inverse_test(
    const mgard::TensorMeshHierarchy<N, Real> &hierarchy, Real const *const u) {
  const std::size_t ndof = hierarchy.ndof();
  std::vector<Real> v_(ndof);
  std::vector<Real> buffer_(ndof);
  Real *const v = v_.data();
  Real *const buffer = buffer_.data();
  TrialTracker tracker;
  const std::size_t singleton = 0;
  for (std::size_t l = 0; l <= hierarchy.L; ++l) {
    std::array<mgard::TensorIndexRange, N> multiindex_components;
    for (std::size_t dimension = 0; dimension < N; ++dimension) {
      multiindex_components.at(dimension) = hierarchy.indices(l, dimension);
    }
    for (std::size_t dimension = 0; dimension < N; ++dimension) {
      const mgard::ConstituentMassMatrix<N, Real> M(hierarchy, l, dimension);
      const mgard::ConstituentMassMatrixInverse<N, Real> A(hierarchy, l,
                                                           dimension);

      std::array<mgard::TensorIndexRange, N> multiindex_components_ =
          multiindex_components;
      multiindex_components_.at(dimension) = {.begin_ = &singleton,
                                              .end_ = &singleton + 1};

      for (const std::array<std::size_t, N> multiindex :
           mgard::CartesianProduct<mgard::TensorIndexRange, N>(
               multiindex_components_)) {
        mgard::shuffle(hierarchy, u, v);
        M(multiindex, v);
        A(multiindex, v);
        mgard::unshuffle(hierarchy, v, buffer);
        for (std::size_t i = 0; i < ndof; ++i) {
          tracker += buffer[i] == Catch::Approx(u[i]);
        }
        multiindex_components_.at(dimension) =
            multiindex_components.at(dimension);
      }
    }
  }
  REQUIRE(tracker);
}

} // namespace

TEST_CASE("constituent mass matrix inverses", "[TensorMassMatrix]") {
  SECTION("1D and default spacing") {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({9});
    const std::size_t ndof = 9;
    const std::array<float, ndof> u_ = {-1, -1, 8, 2, 4, 3, -1, -4, 3};
    std::array<float, ndof> v_;
    std::array<float, ndof> buffer_;
    float const *const u = u_.data();
    float *const v = v_.data();
    float *const buffer = buffer_.data();
    {
      const std::size_t l = 3;
      const std::size_t dimension = 0;
      const mgard::ConstituentMassMatrix<1, float> M(hierarchy, l, dimension);
      const mgard::ConstituentMassMatrixInverse<1, float> A(hierarchy, l,
                                                            dimension);
      mgard::shuffle(hierarchy, u, v);
      M({0}, v);
      A({0}, v);
      mgard::unshuffle(hierarchy, v, buffer);
      TrialTracker tracker;
      for (std::size_t i = 0; i < ndof; ++i) {
        tracker += buffer_.at(i) == Catch::Approx(u_.at(i));
      }
      REQUIRE(tracker);
    }
    {
      const std::size_t l = 1;
      const std::size_t dimension = 0;
      const mgard::ConstituentMassMatrix<1, float> M(hierarchy, l, dimension);
      const mgard::ConstituentMassMatrixInverse<1, float> A(hierarchy, l,
                                                            dimension);
      mgard::shuffle(hierarchy, u, v);
      // Opposite order.
      A({0}, v);
      M({0}, v);
      mgard::unshuffle(hierarchy, v, buffer);
      TrialTracker tracker;
      for (std::size_t i = 0; i < ndof; ++i) {
        tracker += buffer_.at(i) == Catch::Approx(u_.at(i));
      }
      REQUIRE(tracker);
    }
  }

  SECTION("2D and custom spacing") {
    const std::vector<double> xs = {0.469, 1.207, 1.918, 2.265, 2.499,
                                    2.525, 2.879, 3.109, 3.713};
    const std::vector<double> ys = {0.137, 0.907, 1.363, 1.856, 2.188, 3.008,
                                    3.643, 4.580, 5.320, 5.464, 6.223, 6.856,
                                    7.083, 7.459, 7.748, 8.641, 8.740};
    const mgard::TensorMeshHierarchy<2, double> hierarchy({9, 17}, {{xs, ys}});
    const std::size_t ndof = 9 * 17;
    const std::array<double, ndof> u_ = {
        3,   -5,  -3, 3,  2,   -4, -7, -3, -8, -4,  1,  2,  5,   5,   4,  5,
        3,   1,   -7, -9, -10, 1,  -4, 8,  3,  -10, -4, -2, 1,   -9,  -4, 7,
        -9,  5,   -1, -5, -10, 7,  1,  7,  -5, 3,   2,  -3, -5,  -1,  -5, -9,
        2,   9,   6,  -9, 1,   4,  -3, -9, 8,  -6,  -7, 1,  -1,  -8,  9,  -6,
        -2,  5,   -5, 10, -2,  -5, -9, -2, -1, 6,   3,  2,  -5,  -4,  -8, -6,
        1,   6,   10, 1,  8,   -1, -6, 4,  5,  9,   10, 9,  -10, -8,  8,  4,
        -6,  -10, 6,  -8, -5,  5,  8,  -9, 2,  -1,  -9, -2, 1,   3,   -7, -8,
        -10, -5,  -2, 2,  -8,  1,  2,  1,  9,  1,   6,  10, -7,  -9,  -8, -6,
        8,   6,   10, 9,  1,   -6, -8, 5,  10, 6,   -9, -8, -7,  -10, 4,  -3,
        -3,  4,   -8, -4, 3,   -1, 4,  -4, -2};
    double const *const u = u_.data();
    exhaustive_constituent_inverse_test<2, double>(hierarchy, u);
  }

  SECTION("3D and nondyadic") {
    const mgard::TensorMeshHierarchy<3, float> hierarchy({9, 8, 7});
    const std::size_t ndof = 9 * 8 * 7;
    std::default_random_engine generator(731617);
    std::uniform_real_distribution<float> distribution(-5, -3);
    std::array<float, ndof> u_;
    std::generate(u_.begin(), u_.end(),
                  [&]() -> float { return distribution(generator); });
    float *const u = u_.data();
    exhaustive_constituent_inverse_test<3, float>(hierarchy, u);
  }
}

namespace {

template <std::size_t N>
void test_mass_matrix_inversion(
    std::default_random_engine &generator,
    std::uniform_real_distribution<float> &distribution,
    const std::array<float, 1089> &u_, std::array<std::size_t, N> shape) {
  const mgard::TensorMeshHierarchy<N, float> hierarchy =
      hierarchy_with_random_spacing(generator, distribution, shape);
  const std::size_t ndof = hierarchy.ndof();
  if (ndof > 1089) {
    throw std::invalid_argument("mesh too large");
  }
  std::vector<float> v_(ndof);
  std::vector<float> buffer_(ndof);
  float const *const u = u_.data();
  float *const v = v_.data();
  float *const buffer = buffer_.data();

  TrialTracker tracker;
  for (std::size_t l = 0; l <= hierarchy.L; ++l) {
    const mgard::TensorMassMatrix<N, float> M(hierarchy, l);
    const mgard::TensorMassMatrixInverse<N, float> A(hierarchy, l);
    mgard::shuffle(hierarchy, u, v);
    M(v);
    A(v);
    mgard::unshuffle(hierarchy, v, buffer);
    float u_square_norm = 0;
    float error_square_norm = 0;
    for (std::size_t i = 0; i < ndof; ++i) {
      const float expected = u_.at(i);
      const float obtained = buffer_.at(i);
      const float error = expected - obtained;
      u_square_norm += expected * expected;
      error_square_norm += error * error;
    }
    tracker += error_square_norm <= 0.00000001 * u_square_norm;
  }
  REQUIRE(tracker);
}

} // namespace

TEST_CASE("tensor product mass matrix inverses", "[TensorMassMatrix]") {
  std::default_random_engine generator(741495);
  std::array<float, 1089> u_;
  {
    std::uniform_real_distribution<float> distribution(-10, 10);
    std::generate(u_.begin(), u_.end(),
                  [&]() -> float { return distribution(generator); });
  }

  std::uniform_real_distribution<float> distribution(0.1, 0.3);

  SECTION("dyadic") {
    test_mass_matrix_inversion<1>(generator, distribution, u_, {1025});
    test_mass_matrix_inversion<2>(generator, distribution, u_, {33, 33});
    test_mass_matrix_inversion<3>(generator, distribution, u_, {17, 5, 9});
    test_mass_matrix_inversion<4>(generator, distribution, u_, {5, 5, 3, 9});
  }

  SECTION("nondyadic") {
    test_mass_matrix_inversion<1>(generator, distribution, u_, {999});
    test_mass_matrix_inversion<2>(generator, distribution, u_, {30, 29});
    test_mass_matrix_inversion<3>(generator, distribution, u_, {9, 6, 11});
    test_mass_matrix_inversion<4>(generator, distribution, u_, {5, 5, 4, 4});
  }
}

TEST_CASE("mass matrices and inverses on 'flat' meshes", "[TensorMassMatrix]") {
  const std::size_t ndof = 36;
  const std::size_t l = 2;
  std::vector<float> u_(ndof);
  std::vector<float> expected_(ndof);
  std::vector<float> obtained_(ndof);
  float *const u = u_.data();
  float *const expected = expected_.data();
  float *const obtained = obtained_.data();
  std::iota(u, u + ndof, 0);
  {
    const mgard::TensorMeshHierarchy<3, float> hierarchy({3, 3, 4});
    const mgard::TensorMassMatrix<3, float> M(hierarchy, l);
    std::copy(u, u + ndof, expected);
    M(expected);
  }

  {
    const mgard::TensorMeshHierarchy<4, float> hierarchy({3, 3, 1, 4});
    const mgard::TensorMassMatrix<4, float> M(hierarchy, l);
    std::copy(u, u + ndof, obtained);
    M(obtained);

    REQUIRE(obtained_ == expected_);
  }

  // Getting some small discrepancies here when using `-ffast-math`.
  {
    const mgard::TensorMeshHierarchy<7, float> hierarchy({1, 1, 3, 1, 3, 4, 1});
    const mgard::TensorMassMatrix<7, float> M(hierarchy, l);
    std::copy(u, u + ndof, obtained);
    M(obtained);

    TrialTracker tracker;
    for (std::size_t i = 0; i < ndof; ++i) {
      tracker += std::abs((obtained[i] - expected[i]) / expected[i]) < 1e-6;
    }
    REQUIRE(tracker);
  }
}
