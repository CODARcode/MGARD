#include "catch2/catch.hpp"

#include <cstddef>

#include <numeric>
#include <random>
#include <vector>

#include "testing_utilities.hpp"

#include "TensorMeshHierarchy.hpp"
#include "mgard.hpp"
#include "mgard_mesh.hpp"

TEMPLATE_TEST_CASE("uniform mass matrix", "[mgard]", float, double) {
  const mgard::TensorMeshHierarchy<1, TestType> hierarchy({17});
  const std::vector<TestType> v = {3, -5, -2, -5, -4, 0, -4, -2, 1,
                                   2, -5, 3,  -3, 4,  1, -2, -5};

  {
    std::vector<TestType> copy = v;
    mgard::mass_matrix_multiply(hierarchy, 0, 0, copy.data());
    const std::vector<TestType> expected = {
        1, -19, -18, -26, -21, -8, -18, -11, 4, 4, -15, 4, -5, 14, 6, -12, -12};
    REQUIRE(copy == expected);
  }

  {
    std::vector<TestType> copy = v;
    mgard::mass_matrix_multiply(hierarchy, 1, 0, copy.data());
    const std::vector<TestType> expected = {
        8, -5, -18, -5, -44, 0, -38, -2, -10, 2, -44, 3, -32, 4, -8, -2, -18};
    REQUIRE(copy == expected);
  }

  {
    std::vector<TestType> copy = v;
    mgard::mass_matrix_multiply(hierarchy, 2, 0, copy.data());
    const std::vector<TestType> expected = {8, -5, -2, -5,  -48, 0, -4, -2, -12,
                                            2, -5, 3,  -64, 4,   1, -2, -52};
    REQUIRE(copy == expected);
  }

  {
    std::vector<TestType> copy = v;
    mgard::mass_matrix_multiply(hierarchy, 3, 0, copy.data());
    const std::vector<TestType> expected = {56, -5, -2, -5, -4, 0, -4, -2, 16,
                                            2,  -5, 3,  -3, 4,  1, -2, -72};
    REQUIRE(copy == expected);
  }

  {
    std::vector<TestType> copy = v;
    mgard::mass_matrix_multiply(hierarchy, 4, 0, copy.data());
    const std::vector<TestType> expected = {16, -5, -2, -5, -4, 0, -4, -2,  1,
                                            2,  -5, 3,  -3, 4,  1, -2, -112};
    REQUIRE(copy == expected);
  }
}

TEMPLATE_TEST_CASE("inversion of uniform mass matrix", "[mgard]", float,
                   double) {
  std::vector<std::size_t> Ls = {0, 3, 7};

  std::default_random_engine generator(741495);
  std::uniform_real_distribution<TestType> distribution(-10, 10);

  for (const std::size_t L : Ls) {
    // Would be good to use a function for these sizes once that's been set up.
    const std::size_t N = (1 << L) + 1;
    std::vector<TestType> v(N);
    for (TestType &value : v) {
      value = distribution(generator);
    }

    for (std::size_t l = 0; l <= L; l += 1) {
      std::vector<TestType> copy = v;
      mgard::mass_matrix_multiply(l, copy);
      mgard::solve_tridiag_M(l, copy);
      TrialTracker tracker;
      for (std::size_t i = 0; i < N; ++i) {
        tracker += v.at(i) == Approx(copy.at(i));
      }
      REQUIRE(tracker);
    }
  }
}

TEMPLATE_TEST_CASE("uniform mass matrix restriction", "[mgard]", float,
                   double) {
  {
    const std::vector<TestType> v = {159, 181, 144, 113, 164};

    {
      std::vector<TestType> copy = v;
      mgard::restriction(1, copy);
      const std::vector<TestType> expected = {249.5, 181, 291, 113, 220.5};
      REQUIRE(copy == expected);
    }

    {
      std::vector<TestType> copy = v;
      mgard::restriction(2, copy);
      const std::vector<TestType> expected = {231, 181, 144, 113, 236};
      REQUIRE(copy == expected);
    }
  }

  std::default_random_engine generator(477899);
  std::uniform_real_distribution<TestType> distribution(-100, 100);

  const std::vector<std::size_t> Ls = {3, 4, 5};
  for (const std::size_t L : Ls) {
    const std::size_t N = (1 << L) + 1;
    std::vector<TestType> v(N);
    v.at(0) = distribution(generator);
    for (std::size_t i = 1; i < N; i += 2) {
      v.at(i) = 0.5 * (v.at(i - 1) + (v.at(i + 1) = distribution(generator)));
    }
    std::vector<TestType> copy = v;
    mgard::mass_matrix_multiply(0, copy);
    mgard::restriction(1, copy);
    mgard::solve_tridiag_M(1, copy);
    TrialTracker tracker;
    for (std::size_t i = 0; i < N; i += 2) {
      tracker += v.at(i) == Approx(copy.at(i));
    }
    REQUIRE(tracker);
  }
}

TEMPLATE_TEST_CASE("uniform interpolation", "[mgard]", float, double) {
  SECTION("1D interpolation") {
    std::vector<TestType> v = {8, -2, 27, 33, -22};
    {
      mgard::interpolate_from_level_nMl(1, v);
      const std::vector<TestType> expected = {8, 17.5, 27, 2.5, -22};
      REQUIRE(v == expected);
    }
    {
      mgard::interpolate_from_level_nMl(2, v);
      const std::vector<TestType> expected = {8, 17.5, -7, 2.5, -22};
      REQUIRE(v == expected);
    }
    REQUIRE_THROWS(mgard::interpolate_from_level_nMl(0, v));
  }

  SECTION("1D interpolation and subtraction") {
    std::vector<TestType> v = {-5, -2, 3, 13, 23, 13, 10, 14, 24};
    {
      mgard::interpolate_old_to_new_and_subtract(0, v);
      const std::vector<TestType> expected = {-5,   -1, 3,  0, 23,
                                              -3.5, 10, -3, 24};
      REQUIRE(v == expected);
    }
    {
      mgard::interpolate_old_to_new_and_subtract(1, v);
      const std::vector<TestType> expected = {-5,   -1,    -6, 0, 23,
                                              -3.5, -13.5, -3, 24};
      REQUIRE(v == expected);
    }
    {
      mgard::interpolate_old_to_new_and_subtract(2, v);
      const std::vector<TestType> expected = {-5,   -1,    -6, 0, 13.5,
                                              -3.5, -13.5, -3, 24};
      REQUIRE(v == expected);
    }
    REQUIRE_THROWS(mgard::interpolate_old_to_new_and_subtract(3, v));
  }

  SECTION("2D interpolation and subtraction") {
    {
      const std::size_t nrow = 3;
      const std::size_t ncol = 3;
      const mgard::TensorMeshHierarchy<2, TestType> hierarchy({nrow, ncol});
      std::vector<TestType> v = {11, 13, 15, 12, 9, 20, 16, 14, 23};
      std::vector<TestType> row_vec(ncol);
      std::vector<TestType> col_vec(nrow);
      {
        mgard::interpolate_old_to_new_and_subtract(hierarchy, 0, v.data(),
                                                   row_vec, col_vec);
        const std::vector<TestType> expected = {11, 0,  15,   -1.5, -7.25,
                                                1,  16, -5.5, 23};
        REQUIRE(v == expected);
      }
      REQUIRE_THROWS(mgard::interpolate_old_to_new_and_subtract(
          hierarchy, 1, v.data(), row_vec, col_vec));
    }
    {
      const std::size_t nrow = 5;
      const std::size_t ncol = 3;
      const mgard::TensorMeshHierarchy<2, TestType> hierarchy({nrow, ncol});
      std::vector<TestType> v = {-4, -4, -2, -4, -1, 2, -4, 1,
                                 5,  0,  3,  8,  2,  8, 9};
      std::vector<TestType> row_vec(ncol);
      std::vector<TestType> col_vec(nrow);
      {
        mgard::interpolate_old_to_new_and_subtract(hierarchy, 0, v.data(),
                                                   row_vec, col_vec);
        const std::vector<TestType> expected = {
            -4, -1, -2, 0, 0.25, 0.5, -4, 0.5, 5, 1, 0, 1, 2, 2.5, 9};
        REQUIRE(v == expected);
      }
      REQUIRE_THROWS(mgard::interpolate_old_to_new_and_subtract(
          hierarchy, 1, v.data(), row_vec, col_vec));
    }
  }
}

// Ideally these functions would test that `l` is within bounds.
TEMPLATE_TEST_CASE("BLAS-like level operations", "[mgard]", float, double) {
  SECTION("assignment") {
    const mgard::TensorMeshHierarchy<2, TestType> hierarchy({3, 5});
    std::vector<TestType> v(hierarchy.ndof());
    std::iota(v.begin(), v.end(), 1);
    {
      std::vector<TestType> copy = v;
      mgard::assign_num_level(hierarchy, 0, copy.data(),
                              static_cast<TestType>(-3));
      TrialTracker tracker;
      for (const TestType value : copy) {
        tracker += value == -3;
      }
      REQUIRE(tracker);
    }
    {
      std::vector<TestType> copy = v;
      mgard::assign_num_level(hierarchy, 1, copy.data(),
                              static_cast<TestType>(-1));
      const std::vector<TestType> expected = {-1, 2,  -1, 4,  -1, 6,  7, 8,
                                              9,  10, -1, 12, -1, 14, -1};
      REQUIRE(copy == expected);
    }
  }

  SECTION("copying") {
    {
      const mgard::TensorMeshHierarchy<2, TestType> hierarchy({3, 3});
      std::vector<TestType> v(hierarchy.ndof());
      std::iota(v.begin(), v.end(), 1);
      {
        std::vector<TestType> destination = v;
        const std::vector<TestType> source = {-2, 0, -3, 0, 0, 0, -5, 0, -7};
        mgard::copy_level(hierarchy, 1, source.data(), destination.data());
        const std::vector<TestType> expected = {-2, 2, -3, 4, 5, 6, -5, 8, -7};
        REQUIRE(destination == expected);
      }
      {
        std::vector<TestType> destination = v;
        const std::vector<TestType> source(hierarchy.ndof(), -1);
        mgard::copy_level(hierarchy, 0, source.data(), destination.data());
        TrialTracker tracker;
        for (const TestType value : destination) {
          tracker += value == -1;
        }
        REQUIRE(tracker);
      }
    }
  }

  SECTION("addition and subtraction") {
    const mgard::TensorMeshHierarchy<2, TestType> hierarchy({5, 5});
    std::vector<TestType> v(hierarchy.ndof());
    std::iota(v.begin(), v.end(), 1);
    {
      std::vector<TestType> addend(hierarchy.ndof());
      std::iota(addend.rbegin(), addend.rend(), 2);
      std::vector copy = v;
      mgard::add_level(hierarchy, 0, copy.data(), addend.data());
      TrialTracker tracker;
      for (const TestType value : copy) {
        tracker += value == 27;
      }
      REQUIRE(tracker);
    }
    {
      std::vector<TestType> expected = v;
      std::vector<TestType> subtrahend(hierarchy.ndof(), 0);
      for (const std::size_t i : {0, 4, 10, 14, 20, 24}) {
        subtrahend.at(i) = v.at(i);
        expected.at(i) = 0;
      }
      std::vector copy = v;
      mgard::subtract_level(hierarchy, 1, copy.data(), subtrahend.data());
      REQUIRE(copy == expected);
    }
    {
      std::vector<TestType> addend(hierarchy.ndof(), 0);
      std::vector<TestType> expected = v;
      for (const std::size_t index : {0, 4, 20, 24}) {
        addend.at(index) = 100;
        expected.at(index) = 100 + index + 1;
      }
      std::vector<TestType> copy = v;
      mgard::add_level(hierarchy, 2, copy.data(), addend.data());
      REQUIRE(copy == expected);
    }
  }
}

TEST_CASE("2D (de)quantization", "[mgard]") {
  std::default_random_engine generator(312769);

  SECTION("quantization followed by dequantization") {
    std::uniform_real_distribution<float> distribution(-30, -10);
    const mgard::TensorMeshHierarchy<2, float> hierarchy({23, 11});
    const std::size_t N = hierarchy.ndof();
    // The quantum will be `norm * tol`.
    const float norm = 1;
    const float tol = 0.01;

    std::vector<float> v(N);
    std::vector<int> quantized(sizeof(float) / sizeof(int) + N);
    std::vector<float> dequantized(N);

    for (float &value : v) {
      value = distribution(generator);
    }

    mgard::quantize_interleave(hierarchy, v.data(), quantized, norm, tol);
    mgard::dequantize_interleave(hierarchy, dequantized.data(), quantized);

    TrialTracker tracker;
    for (std::size_t i = 0; i < N; ++i) {
      tracker += std::abs(v.at(i) - dequantized.at(i)) <= norm * tol / 2;
    }
    REQUIRE(tracker);
  }

  SECTION("dequantization followed by quantization") {
    std::uniform_int_distribution<int> distribution(-50, 150);
    const mgard::TensorMeshHierarchy<3, double> hierarchy({5, 89, 2});
    const std::size_t N = hierarchy.ndof();
    const double norm = 5;
    const double tol = 0.01;
    const double quantum = norm * tol;
    std::vector<int> v(sizeof(double) / sizeof(int) + N);
    {
      double *const p = reinterpret_cast<double *>(v.data());
      *p = quantum;
    }
    {
      int *q = v.data() + sizeof(double) / sizeof(int);
      for (std::size_t i = 0; i < N; ++i) {
        *q++ = distribution(generator);
      }
    }
    std::vector<double> dequantized(N);
    std::vector<int> requantized(sizeof(double) / sizeof(int) + N);

    mgard::dequantize_interleave(hierarchy, dequantized.data(), v);
    mgard::quantize_interleave(hierarchy, dequantized.data(), requantized, norm,
                               tol);

    TrialTracker tracker;
    {
      double const *const p = reinterpret_cast<double const *>(v.data());
      double const *const q =
          reinterpret_cast<double const *>(requantized.data());
      tracker += *p == *q;
    }
    {
      int const *p = v.data() + sizeof(double) / sizeof(int);
      int const *q = requantized.data() + sizeof(double) / sizeof(int);
      for (std::size_t i = 0; i < N; ++i) {
        tracker += *p++ == *q++;
      }
    }
    REQUIRE(tracker);
  }
}
