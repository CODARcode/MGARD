#include "catch2/catch.hpp"

#include <array>
#include <vector>

#include "testing_utilities.hpp"

#include "TensorMassMatrix.hpp"
#include "TensorMeshHierarchy.hpp"

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
        tracker += v_.at(i) == Approx(expected.at(i));
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
