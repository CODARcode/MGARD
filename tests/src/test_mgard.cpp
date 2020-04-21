#include "catch2/catch.hpp"

#include <cstddef>

#include <random>
#include <vector>

#include "testing_utilities.hpp"

#include "mgard.h"
#include "mgard_mesh.hpp"

TEMPLATE_TEST_CASE("uniform mass matrix", "[mgard]", float, double) {
  const std::vector<TestType> v = {3, -5, -2, -5, -4, 0, -4, -2, 1,
                                   2, -5, 3,  -3, 4,  1, -2, -5};

  {
    std::vector<TestType> copy = v;
    mgard::mass_matrix_multiply(0, copy);
    const std::vector<TestType> expected = {
        1, -19, -18, -26, -21, -8, -18, -11, 4, 4, -15, 4, -5, 14, 6, -12, -12};
    REQUIRE(copy == expected);
  }

  {
    std::vector<TestType> copy = v;
    mgard::mass_matrix_multiply(1, copy);
    const std::vector<TestType> expected = {
        8, -5, -18, -5, -44, 0, -38, -2, -10, 2, -44, 3, -32, 4, -8, -2, -18};
    REQUIRE(copy == expected);
  }

  {
    std::vector<TestType> copy = v;
    mgard::mass_matrix_multiply(2, copy);
    const std::vector<TestType> expected = {8, -5, -2, -5,  -48, 0, -4, -2, -12,
                                            2, -5, 3,  -64, 4,   1, -2, -52};
    REQUIRE(copy == expected);
  }

  {
    std::vector<TestType> copy = v;
    mgard::mass_matrix_multiply(3, copy);
    const std::vector<TestType> expected = {56, -5, -2, -5, -4, 0, -4, -2, 16,
                                            2,  -5, 3,  -3, 4,  1, -2, -72};
    REQUIRE(copy == expected);
  }

  {
    std::vector<TestType> copy = v;
    mgard::mass_matrix_multiply(4, copy);
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
