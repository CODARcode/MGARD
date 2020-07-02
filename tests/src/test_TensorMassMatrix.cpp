#include "catch2/catch.hpp"

#include <array>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "testing_random.hpp"
#include "testing_utilities.hpp"

#include "TensorMassMatrix.hpp"
#include "TensorMeshHierarchy.hpp"
#include "utilities.hpp"

TEST_CASE("constituent mass matrices", "[TensorMassMatrix]") {
  SECTION("1D and default spacing") {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({9});
    const std::array<float, 9> u_ = {-2, 1, 1, 8, 3, -2, -7, -4, 0};
    {
      const std::size_t l = 3;
      const std::size_t dimension = 0;
      const mgard::ConstituentMassMatrix<1, float> M(hierarchy, l, dimension);
      std::array<float, 9> v_ = u_;
      float *const v = v_.data();
      M({0}, v);
      std::array<float, 9> expected = {-3, 3, 13, 36, 18, -12, -34, -23, -4};
      for (float &value : expected) {
        value /= 48;
      }
      TrialTracker tracker;
      for (std::size_t i = 0; i < 9; ++i) {
        tracker += v_.at(i) == Approx(expected.at(i));
      }
      REQUIRE(tracker);
    }
    {
      const std::size_t l = 1;
      const std::size_t dimension = 0;
      const mgard::ConstituentMassMatrix<1, float> M(hierarchy, l, dimension);
      std::array<float, 9> v_ = u_;
      float *const v = v_.data();
      M({0}, v);
      std::array<float, 9> expected = {-1, 1, 1, 8, 10, -2, -7, -4, 3};
      for (std::size_t i = 0; i < 3; ++i) {
        expected.at(4 * i) /= 12;
      }
      TrialTracker tracker;
      for (std::size_t i = 0; i < 9; ++i) {
        tracker += v_.at(i) == Approx(expected.at(i));
      }
      REQUIRE(tracker);
    }
  }

  SECTION("2D and custom spacing") {
    const mgard::TensorMeshHierarchy<2, double> hierarchy(
        {5, 5}, {{{0, 0.1, 0.5, 0.75, 1}, {0, 0.65, 0.70, 0.75, 1}}});
    const std::array<double, 25> u_ = {8,  -3, 1,  5,  10, 9,  -10, -3, 8,
                                       10, -3, 6,  -7, -3, 3,  3,   -9, 0,
                                       -1, 8,  -6, 7,  1,  -2, 10};
    {
      const std::size_t l = 2;
      const std::size_t dimension = 0;
      const mgard::ConstituentMassMatrix<2, double> M(hierarchy, l, dimension);
      std::array<double, 25> v_ = u_;
      double *const v = v_.data();
      M({0, 0}, v);
      M({0, 3}, v);
      const std::array<double, 25> expected = {2.5 / 6,
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
      for (std::size_t i = 0; i < 25; ++i) {
        tracker += v_.at(i) == Approx(expected.at(i));
      }
      REQUIRE(tracker);
    }
    {
      const std::size_t l = 2;
      const std::size_t dimension = 1;
      const mgard::ConstituentMassMatrix<2, double> M(hierarchy, l, dimension);
      std::array<double, 25> v_ = u_;
      double *const v = v_.data();
      M({1, 0}, v);
      M({2, 0}, v);
      const std::array<double, 25> expected = {8,
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
      for (std::size_t i = 0; i < 25; ++i) {
        // Changed the margin because we were getting a tiny error at index 10
        // (where the exact value is zero).
        tracker += v_.at(i) == Approx(expected.at(i)).margin(1e-15);
      }
      REQUIRE(tracker);
    }
    {
      const std::size_t l = 1;
      const std::size_t dimension = 1;
      const mgard::ConstituentMassMatrix<2, double> M(hierarchy, l, dimension);
      std::array<double, 25> v_ = u_;
      double *const v = v_.data();
      M({4, 0}, v);
      const std::array<double, 5> expected = {-7.7 / 6, 7, -2.8 / 6 + 3.6 / 6,
                                              -2, 6.3 / 6};
      TrialTracker tracker;
      for (std::size_t i = 0; i < 20; ++i) {
        tracker += v_.at(i) == u_.at(i);
      }
      for (std::size_t i = 20; i < 25; ++i) {
        tracker += v_.at(i) == Approx(expected.at(i - 20));
      }
      REQUIRE(tracker);
    }
  }
}

TEST_CASE("tensor product mass matrices", "[TensorMassMatrix]") {
  const mgard::TensorMeshHierarchy<2, double> hierarchy(
      {3, 3}, {{{0, 0.5, 1}, {1, 1.25, 2}}});
  const std::array<double, 9> u_ = {2, 3, -9, -2, 6, 5, 2, -1, 5};
  {
    const std::size_t l = 1;
    const mgard::TensorMassMatrix<2, double> M(hierarchy, l);
    std::array<double, 9> v_ = u_;
    double *const v = v_.data();
    M(v);
    const std::array<double, 9> expected = {0.05555555555555556,
                                            0.20486111111111108,
                                            -0.14583333333333334,
                                            0.0625,
                                            0.875,
                                            0.6041666666666666,
                                            0.027777777777777776,
                                            0.2743055555555555,
                                            0.3541666666666667};
    TrialTracker tracker;
    for (std::size_t i = 0; i < 9; ++i) {
      tracker += v_.at(i) == Approx(expected.at(i));
    }
    REQUIRE(tracker);
  }
  {
    const std::size_t l = 0;
    const mgard::TensorMassMatrix<2, double> M(hierarchy, l);
    std::array<double, 9> v_ = u_;
    double *const v = v_.data();
    M(v);
    const std::array<double, 9> expected = {
        -0.02777777777777779, 3.0,  -0.5555555555555555, -2.0, 6.0, 5.0,
        0.3611111111111111,   -1.0, 0.22222222222222224};
    TrialTracker tracker;
    for (std::size_t i = 0; i < 9; ++i) {
      tracker += v_.at(i) == Approx(expected.at(i));
    }
    REQUIRE(tracker);
  }
}

TEST_CASE("constituent mass matrice inverses", "[TensorMassMatrix]") {
  SECTION("1D and default spacing") {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({9});
    const std::array<float, 9> u_ = {-1, -1, 8, 2, 4, 3, -1, -4, 3};
    {
      const std::size_t l = 3;
      const std::size_t dimension = 0;
      const mgard::ConstituentMassMatrix<1, float> M(hierarchy, l, dimension);
      std::vector<float> buffer(M.dimension());
      const mgard::ConstituentMassMatrixInverse<1, float> A(
          hierarchy, l, dimension, buffer.data());
      std::array<float, 9> v_ = u_;
      float *const v = v_.data();
      M({0}, v);
      A({0}, v);
      TrialTracker tracker;
      for (std::size_t i = 0; i < 9; ++i) {
        tracker += v_.at(i) == Approx(u_.at(i));
      }
      REQUIRE(tracker);
    }
    {
      const std::size_t l = 1;
      const std::size_t dimension = 0;
      const mgard::ConstituentMassMatrix<1, float> M(hierarchy, l, dimension);
      std::vector<float> buffer(M.dimension());
      const mgard::ConstituentMassMatrixInverse<1, float> A(
          hierarchy, l, dimension, buffer.data());
      std::array<float, 9> v_ = u_;
      float *const v = v_.data();
      // Opposite order.
      A({0}, v);
      M({0}, v);
      TrialTracker tracker;
      for (std::size_t i = 0; i < 9; ++i) {
        tracker += v_.at(i) == Approx(u_.at(i));
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
    const std::array<double, 153> u_ = {
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
    // Maximum of the sizes.
    std::vector<double> buffer_(17);
    double *const buffer = buffer_.data();
    TrialTracker tracker;
    for (std::size_t l = 0; l <= hierarchy.L; ++l) {
      std::array<std::vector<std::size_t>, 2> multiindex_components;
      for (std::size_t dimension = 0; dimension < 2; ++dimension) {
        multiindex_components.at(dimension) = hierarchy.indices(l, dimension);
      }
      for (std::size_t dimension = 0; dimension < 2; ++dimension) {
        const mgard::ConstituentMassMatrix<2, double> M(hierarchy, l,
                                                        dimension);
        const mgard::ConstituentMassMatrixInverse<2, double> A(
            hierarchy, l, dimension, buffer);

        std::array<std::vector<std::size_t>, 2> multiindex_components_ =
            multiindex_components;
        multiindex_components_.at(dimension) = {0};

        for (const std::array<std::size_t, 2> multiindex :
             mgard::CartesianProduct<std::size_t, 2>(multiindex_components_)) {
          std::array<double, 153> v_ = u_;
          double *const v = v_.data();
          M(multiindex, v);
          A(multiindex, v);
          for (std::size_t i = 0; i < 153; ++i) {
            tracker += v_.at(i) == Approx(u_.at(i));
          }
          multiindex_components_.at(dimension) =
              multiindex_components.at(dimension);
        }
      }
    }
    REQUIRE(tracker);
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
  float *const v = v_.data();

  TrialTracker tracker;
  for (std::size_t l = 0; l <= hierarchy.L; ++l) {
    std::uninitialized_copy_n(u_.begin(), ndof, v_.begin());
    const mgard::TensorMassMatrix<N, float> M(hierarchy, l);
    const mgard::TensorMassMatrixInverse<N, float> A(hierarchy, l);
    M(v);
    A(v);
    float u_square_norm = 0;
    float error_square_norm = 0;
    for (std::size_t i = 0; i < ndof; ++i) {
      const float expected = u_.at(i);
      const float obtained = v_.at(i);
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

    for (float &value : u_) {
      value = distribution(generator);
    }
  }

  std::uniform_real_distribution<float> distribution(0.1, 0.3);
  test_mass_matrix_inversion<1>(generator, distribution, u_, {1025});
  test_mass_matrix_inversion<2>(generator, distribution, u_, {33, 33});
  test_mass_matrix_inversion<3>(generator, distribution, u_, {17, 5, 9});
  test_mass_matrix_inversion<4>(generator, distribution, u_, {5, 5, 3, 9});
}
