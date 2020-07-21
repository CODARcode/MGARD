#include "catch2/catch.hpp"

#include <cstddef>

#include <numeric>
#include <random>
#include <vector>

#include "testing_utilities.hpp"

#include "TensorMeshHierarchy.hpp"
#include "mgard.hpp"
#include "mgard_mesh.hpp"

namespace {

template <std::size_t N>
std::array<std::size_t, N> get_dyadic_shape(const std::size_t L) {
  std::array<std::size_t, N> shape;
  shape.fill((1 << L) + 1);
  return shape;
}

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
    const mgard::TensorMeshHierarchy<1, TestType> hierarchy(
        get_dyadic_shape<1>(L));
    const std::size_t N = hierarchy.ndof();
    std::vector<TestType> v(N);
    for (TestType &value : v) {
      value = distribution(generator);
    }

    for (std::size_t l = 0; l <= L; l += 1) {
      std::vector<TestType> copy = v;
      mgard::mass_matrix_multiply(hierarchy, l, 0, copy.data());
      mgard::solve_tridiag_M(hierarchy, l, 0, copy.data());
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
    const mgard::TensorMeshHierarchy<1, TestType> hierarchy({5});
    const std::vector<TestType> v = {159, 181, 144, 113, 164};

    {
      std::vector<TestType> copy = v;
      mgard::restriction(hierarchy, 1, 0, copy.data());
      const std::vector<TestType> expected = {249.5, 181, 291, 113, 220.5};
      REQUIRE(copy == expected);
    }

    {
      std::vector<TestType> copy = v;
      mgard::restriction(hierarchy, 2, 0, copy.data());
      const std::vector<TestType> expected = {231, 181, 144, 113, 236};
      REQUIRE(copy == expected);
    }
  }

  std::default_random_engine generator(477899);
  std::uniform_real_distribution<TestType> distribution(-100, 100);

  const std::vector<std::size_t> Ls = {3, 4, 5};
  for (const std::size_t L : Ls) {
    const mgard::TensorMeshHierarchy<1, TestType> hierarchy(
        get_dyadic_shape<1>(L));
    const std::size_t N = hierarchy.ndof();
    std::vector<TestType> v(N);
    v.at(0) = distribution(generator);
    for (std::size_t i = 1; i < N; i += 2) {
      v.at(i) = 0.5 * (v.at(i - 1) + (v.at(i + 1) = distribution(generator)));
    }
    std::vector<TestType> copy = v;
    mgard::mass_matrix_multiply(hierarchy, 0, 0, copy.data());
    mgard::restriction(hierarchy, 1, 0, copy.data());
    mgard::solve_tridiag_M(hierarchy, 1, 0, copy.data());
    TrialTracker tracker;
    for (std::size_t i = 0; i < N; i += 2) {
      tracker += v.at(i) == Approx(copy.at(i));
    }
    REQUIRE(tracker);
  }
}

TEMPLATE_TEST_CASE("uniform interpolation", "[mgard]", float, double) {
  SECTION("1D interpolation") {
    const mgard::TensorMeshHierarchy<1, TestType> hierarchy({5});
    std::vector<TestType> v = {8, -2, 27, 33, -22};
    {
      mgard::interpolate_old_to_new_and_overwrite(hierarchy, 1, 0, v.data());
      const std::vector<TestType> expected = {8, 17.5, 27, 2.5, -22};
      REQUIRE(v == expected);
    }
    {
      mgard::interpolate_old_to_new_and_overwrite(hierarchy, 2, 0, v.data());
      const std::vector<TestType> expected = {8, 17.5, -7, 2.5, -22};
      REQUIRE(v == expected);
    }
    REQUIRE_THROWS(
        mgard::interpolate_old_to_new_and_overwrite(hierarchy, 0, 0, v.data()));
  }

  SECTION("1D interpolation and subtraction") {
    const mgard::TensorMeshHierarchy<1, TestType> hierarchy({9});
    std::vector<TestType> v = {-5, -2, 3, 13, 23, 13, 10, 14, 24};
    {
      mgard::interpolate_old_to_new_and_subtract(hierarchy, 0, 0, v.data());
      const std::vector<TestType> expected = {-5,   -1, 3,  0, 23,
                                              -3.5, 10, -3, 24};
      REQUIRE(v == expected);
    }
    {
      mgard::interpolate_old_to_new_and_subtract(hierarchy, 1, 0, v.data());
      const std::vector<TestType> expected = {-5,   -1,    -6, 0, 23,
                                              -3.5, -13.5, -3, 24};
      REQUIRE(v == expected);
    }
    {
      mgard::interpolate_old_to_new_and_subtract(hierarchy, 2, 0, v.data());
      const std::vector<TestType> expected = {-5,   -1,    -6, 0, 13.5,
                                              -3.5, -13.5, -3, 24};
      REQUIRE(v == expected);
    }
    REQUIRE_THROWS(
        mgard::interpolate_old_to_new_and_subtract(hierarchy, 3, 0, v.data()));
  }

  SECTION("multidimensional interpolation and subtraction") {
    {
      const std::vector<TestType> u = {11, 13, 15, 12, 9, 20, 16, 14, 23};
      {
        const mgard::TensorMeshHierarchy<2, TestType> hierarchy({3, 3});
        {
          std::vector<TestType> v = u;
          mgard::interpolate_old_to_new_and_subtract(hierarchy, 0, v.data());
          const std::vector<TestType> expected = {11, 0,  15,   -1.5, -7.25,
                                                  1,  16, -5.5, 23};
          REQUIRE(v == expected);
        }
        {
          std::vector<TestType> v = u;
          REQUIRE_THROWS(mgard::interpolate_old_to_new_and_subtract(
              hierarchy, 1, v.data()));
        }
      }
      {
        const mgard::TensorMeshHierarchy<1, TestType> hierarchy({9});
        {
          std::vector<TestType> v = u;
          mgard::interpolate_old_to_new_and_subtract(hierarchy, 0, v.data());
          const std::vector<TestType> expected = {11,  0,  15,   0, 9,
                                                  7.5, 16, -5.5, 23};
          REQUIRE(v == expected);
        }
        {
          std::vector<TestType> v = u;
          mgard::interpolate_old_to_new_and_subtract(hierarchy, 1, v.data());
          const std::vector<TestType> expected = {11, 13, 5,  12, 9,
                                                  20, 0,  14, 23};
          REQUIRE(v == expected);
        }
        {
          std::vector<TestType> v = u;
          mgard::interpolate_old_to_new_and_subtract(hierarchy, 2, v.data());
          const std::vector<TestType> expected = {11, 13, 15, 12, -8,
                                                  20, 16, 14, 23};
          REQUIRE(v == expected);
        }
        {
          std::vector<TestType> v = u;
          REQUIRE_THROWS(mgard::interpolate_old_to_new_and_subtract(
              hierarchy, 3, v.data()));
        }
      }
    }
    {
      const mgard::TensorMeshHierarchy<2, TestType> hierarchy({5, 3});
      std::vector<TestType> v = {-4, -4, -2, -4, -1, 2, -4, 1,
                                 5,  0,  3,  8,  2,  8, 9};
      {
        mgard::interpolate_old_to_new_and_subtract(hierarchy, 0, v.data());
        const std::vector<TestType> expected = {
            -4, -1, -2, 0, 0.25, 0.5, -4, 0.5, 5, 1, 0, 1, 2, 2.5, 9};
        REQUIRE(v == expected);
      }
      REQUIRE_THROWS(
          mgard::interpolate_old_to_new_and_subtract(hierarchy, 1, v.data()));
    }
    {
      const mgard::TensorMeshHierarchy<2, TestType> hierarchy({5, 9});
      const std::vector<TestType> u = {
          4.0,  4.0,  -20.0, -4.0,  12.0,  8.0,   4.0,   -4.0,  8.0,
          0.0,  16.0, -8.0,  12.0,  -12.0, -4.0,  -12.0, -16.0, 16.0,
          -4.0, 12.0, 16.0,  -12.0, -4.0,  -16.0, -16.0, 20.0,  0.0,
          8.0,  12.0, -16.0, 0.0,   4.0,   0.0,   16.0,  20.0,  -8.0,
          12.0, 8.0,  8.0,   12.0,  -4.0,  -20.0, 12.0,  -20.0, -16.0};
      {
        std::vector<TestType> v = u;
        mgard::interpolate_old_to_new_and_subtract(hierarchy, 0, v.data());
        const std::vector<TestType> expected = {
            4.0,  12.0, -20.0, 0.0,   12.0,  0.0,   4.0,   -10.0, 8.0,
            0.0,  17.0, -6.0,  11.0,  -16.0, -3.0,  -6.0,  -15.0, 12.0,
            -4.0, 6.0,  16.0,  -18.0, -4.0,  -6.0,  -16.0, 28.0,  0.0,
            4.0,  4.0,  -28.0, -4.0,  8.0,   3.0,   18.0,  25.0,  0.0,
            12.0, -2.0, 8.0,   10.0,  -4.0,  -24.0, 12.0,  -18.0, -16.0};
        REQUIRE(v == expected);
      }
      {
        std::vector<TestType> v = u;
        mgard::interpolate_old_to_new_and_subtract(hierarchy, 1, v.data());
        const std::vector<TestType> expected = {
            4.0,   4.0,  -28.0, -4.0,  12.0,  8.0,   -6.0,  -4.0,  8.0,
            0.0,   16.0, -8.0,  12.0,  -12.0, -4.0,  -12.0, -16.0, 16.0,
            -12.0, 12.0, 10.0,  -12.0, -8.0,  -16.0, -16.0, 20.0,  4.0,
            8.0,   12.0, -16.0, 0.0,   4.0,   0.0,   16.0,  20.0,  -8.0,
            12.0,  8.0,  4.0,   12.0,  -4.0,  -20.0, 22.0,  -20.0, -16.0};
        REQUIRE(v == expected);
      }
    }
    {
      const mgard::TensorMeshHierarchy<3, TestType> hierarchy({5, 3, 3});
      std::vector<TestType> v = {
          1.0,  0.0, 3.0,  0.0, 0.0, 0.0, 7.0,  0.0, 9.0,  0.0, 0.0, 0.0,
          0.0,  0.0, 0.0,  0.0, 0.0, 0.0, 19.0, 0.0, 21.0, 0.0, 0.0, 0.0,
          25.0, 0.0, 27.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0,
          37.0, 0.0, 39.0, 0.0, 0.0, 0.0, 43.0, 0.0, 45.0};
      mgard::interpolate_old_to_new_and_subtract(hierarchy, 0, v.data());
      const std::vector<TestType> expected = {
          1.0,   -2.0,  3.0,   -4.0,  -5.0,  -6.0,  7.0,   -8.0,  9.0,
          -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0, -17.0, -18.0,
          19.0,  -20.0, 21.0,  -22.0, -23.0, -24.0, 25.0,  -26.0, 27.0,
          -28.0, -29.0, -30.0, -31.0, -32.0, -33.0, -34.0, -35.0, -36.0,
          37.0,  -38.0, 39.0,  -40.0, -41.0, -42.0, 43.0,  -44.0, 45.0};
      REQUIRE(v == expected);
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

    mgard::quantize_interleave(hierarchy, v.data(), quantized.data(), norm,
                               tol);
    mgard::dequantize_interleave(hierarchy, dequantized.data(),
                                 quantized.data());

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

    mgard::dequantize_interleave(hierarchy, dequantized.data(), v.data());
    mgard::quantize_interleave(hierarchy, dequantized.data(),
                               requantized.data(), norm, tol);

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

template <std::size_t N, typename Real>
void test_dyadic_uniform_decomposition(
    const std::vector<Real> &u_,
    const std::vector<std::vector<Real>> &expecteds) {
  for (std::size_t L = 0; L < expecteds.size(); ++L) {
    const mgard::TensorMeshHierarchy<N, Real> hierarchy(get_dyadic_shape<N>(L));
    const std::size_t n = hierarchy.ndof();
    std::vector<Real> v_(u_.begin(), u_.begin() + n);
    Real *const v = v_.data();
    mgard::decompose(hierarchy, v);

    const std::vector<Real> &expected = expecteds.at(L);
    TrialTracker tracker;
    tracker += expected.size() == n;
    for (std::size_t i = 0; i < n; ++i) {
      tracker += v_.at(i) == Approx(expected.at(i));
    }
    REQUIRE(tracker);
  }
}

} // namespace

TEST_CASE("decomposition", "[mgard]") {
  SECTION("1D, dyadic, uniform") {
    const std::vector<float> u_ = {10, 3,   -8,  -6, 3,  0, -5, 0, 0,  -2, -8,
                                   -5, -10, -7,  8,  -2, 3, -1, 0, 9,  -4, -6,
                                   -8, -5,  -10, 1,  3,  7, -8, 1, 10, -2, 8};
    const std::vector<std::vector<float>> expecteds = {
        {10.0, 3.0},
        {11.0, 2.0, -7.0},
        {4.4375, 2.0, -14.5, -3.5, -6.687500000000001},
        {0.4374999999999991, 2.0, -15.678571428571429, -3.5, -4.625000000000002,
         1.0, -5.321428571428571, 2.5, -2.4375000000000004},
        {-0.95703125, 2.0, -15.652199926362297, -3.5, -4.122767857142856, 1.0,
         -4.978599042709867, 2.5, -4.765625000000001, 2.0, -1.173186671575852,
         4.0, -10.689732142857139, -6.0, 9.303985640648008, -7.5,
         -3.1054687499999987},
        {-2.640624999999999,  2.0,  -15.652212055333662, -3.5,
         -4.04917617820324,   1.0,  -4.978756719337627,  2.5,
         -6.024553571428571,  2.0,  -1.1753820153931231, 4.0,
         -9.73304031664212,   -6.0, 9.273408503833881,   -7.5,
         -3.0234374999999996, -2.5, 3.878101069067445,   11.0,
         -1.7446382547864507, 0.0,  -3.6049935368896406, 4.0,
         -8.00669642857143,   4.5,  13.02698941447754,   9.5,
         2.7768547496318097,  0.0,  8.232845339575196,   -11.0,
         0.14062500000000222}};
    test_dyadic_uniform_decomposition<1, float>(u_, expecteds);
  }

  SECTION("2D, dyadic, uniform") {
    const std::vector<double> u_ = {7,  4, 5,  -10, -6, 6,   -8, -5, 6,
                                    -2, 2, -2, 9,   2,  -10, 3,  8,  -8,
                                    -3, 7, -8, -9,  -6, -1,  -4};
    const std::vector<std::vector<double>> expecteds = {
        {7.0, 4.0, 5.0, -10.0},
        {0.9999999999999973, -2.0, 3.9999999999999982, -9.5, -8.5, 0.5,
         -15.000000000000004, -4.0, 3.9999999999999973},
        {3.8007812499999973,
         -2.0,
         -2.9062499999998854,
         -9.5,
         -2.910156250000001,
         1.5,
         -13.75,
         -12.0,
         6.5,
         6.0,
         -1.593749999999881,
         -7.5,
         2.8750000000004396,
         2.5,
         -1.0312499999998854,
         6.0,
         8.75,
         -9.5,
         -0.25,
         14.0,
         -2.5039062500000013,
         -2.0,
         -10.218749999999885,
         4.0,
         -0.6992187500000024}};
    test_dyadic_uniform_decomposition<2, double>(u_, expecteds);
  }

  SECTION("4D, dyadic, uniform") {
    const std::vector<float> u_ = {
        -5, -3, -3, -2, 3,  -9, -2,  1,  1,  2,   4,  5,  -7, -7, 2,  2,  -3,
        -8, -3, -1, -9, -1, -1, 4,   7,  4,  -4,  -8, 1,  7,  -6, -6, 10, 0,
        2,  10, -1, -4, 1,  -4, 5,   3,  2,  1,   3,  -5, 4,  9,  -8, -8, -1,
        -1, 2,  -6, -7, -8, 8,  -10, -4, 5,  -4,  -1, 7,  3,  -4, -5, -3, -2,
        -6, 6,  3,  -9, -9, 10, 2,   2,  -2, -10, -2, -3, -1};
    const std::vector<std::vector<float>> expecteds = {
        {-5.0, -3.0, -3.0, -2.0, 3.0, -9.0, -2.0, 1.0, 1.0, 2.0, 4.0, 5.0, -7.0,
         -7.0, 2.0, 2.0},
        {-2.6406250000000058,
         1.0,
         2.2968749999999902,
         1.5,
         5.25,
         -8.0,
         -2.046875,
         1.5,
         0.7656249999999982,
         6.0,
         9.0,
         11.0,
         -6.25,
         -4.75,
         5.75,
         -0.5,
         -3.5,
         -6.5,
         -6.234375000000018,
         5.0,
         4.078124999999988,
         -3.0,
         1.25,
         10.5,
         0.10937499999999013,
         2.5,
         -1.7031250000000093,
         -2.0,
         2.75,
         4.5,
         -1.5,
         -5.375,
         6.75,
         3.0,
         1.5,
         6.0,
         5.0,
         -0.75,
         1.5,
         -0.875,
         6.5,
         2.875,
         2.25,
         0.75,
         2.25,
         1.0,
         8.75,
         12.5,
         -6.25,
         -5.625,
         2.0,
         -3.5,
         2.0,
         -3.5,
         -10.765625000000007,
         -8.5,
         1.4218749999999971,
         -4.5,
         -5.0,
         -2.5,
         -2.2968750000000018,
         -2.5,
         8.265625,
         11.0,
         -2.5,
         -10.0,
         2.5,
         -1.25,
         -10.0,
         9.0,
         3.0,
         -12.0,
         0.2656249999999858,
         13.5,
         0.3281249999999911,
         7.5,
         0.5,
         -10.5,
         2.4843749999999933,
         -1.5,
         -9.078125000000007}};
    test_dyadic_uniform_decomposition<4, float>(u_, expecteds);
  }

  SECTION("1D, dyadic, nonuniform") {
    const std::array<float, 3> u_ = {{7, 2, 5}};
    const std::array<std::vector<float>, 1> coordinates = {{{10, 28, 34}}};
    const mgard::TensorMeshHierarchy<1, float> hierarchy({3}, coordinates);
    const std::array<float, 3> expected = {{6.125, -3.5, 2.375}};

    std::array<float, 3> v_ = u_;
    float *const v = v_.data();
    mgard::decompose(hierarchy, v);

    TrialTracker tracker;
    for (std::size_t i = 0; i < 3; ++i) {
      tracker += v_.at(i) == Approx(expected.at(i));
    }
    REQUIRE(tracker);
  }
}
