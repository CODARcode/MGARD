#include "catch2/catch.hpp"

#include <array>

#include "testing_random.hpp"
#include "testing_utilities.hpp"

#include "TensorMeshHierarchy.hpp"
#include "TensorRestriction.hpp"
#include "utilities.hpp"

TEST_CASE("constituent restrictions", "[TensorRestriction]") {
  SECTION("1D and default spacing") {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({9});
    const std::array<float, 9> u_ = {9, 2, 4, -4, 7, 5, -2, 5, 6};
    const std::size_t dimension = 0;
    const std::array<std::array<float, 9>, 3> expecteds = {
        {{10, 2, 3, -4, 7.5, 5, 3, 5, 8.5},
         {11, 2, 4, -4, 8, 5, -2, 5, 5},
         {12.5, 2, 4, -4, 7, 5, -2, 5, 9.5}}};
    for (std::size_t l = 3; l > 0; --l) {
      const std::size_t i = 3 - l;
      const std::array<float, 9> &expected = expecteds.at(i);
      const mgard::ConstituentRestriction<1, float> R(hierarchy, l, dimension);
      std::array<float, 9> v_ = u_;
      float *const v = v_.data();
      R({0}, v);
      TrialTracker tracker;
      for (std::size_t i = 0; i < 9; ++i) {
        tracker += v_.at(i) == expected.at(i);
      }
      REQUIRE(tracker);
    }
  }

  SECTION("2D and custom spacing") {
    const mgard::TensorMeshHierarchy<2, double> hierarchy(
        {3, 3}, {{{0, 0.75, 1}, {0, 0.25, 1}}});
    const std::array<double, 9> u_ = {-9, -5, 1, 9, 3, 2, 9, 5, 6};
    {
      const std::size_t l = 1;
      const std::size_t dimension = 0;
      const mgard::ConstituentRestriction<2, double> R(hierarchy, l, dimension);
      const std::array<std::array<std::size_t, 2>, 3> multiindices = {
          {{0, 0}, {0, 1}}};
      const std::array<std::array<double, 9>, 2> expecteds = {{
          {-6.75, -5, 1, 9, 3, 2, 15.75, 5, 6},
          {-9, -4.25, 1, 9, 3, 2, 9, 7.25, 6},
      }};
      for (std::size_t i = 0; i < 2; ++i) {
        const std::array<std::size_t, 2> &multiindex = multiindices.at(i);
        const std::array<double, 9> &expected = expecteds.at(i);
        std::array<double, 9> v_ = u_;
        double *const v = v_.data();
        R(multiindex, v);
        TrialTracker tracker;
        for (std::size_t j = 0; j < 9; ++j) {
          tracker += v_.at(j) == Approx(expected.at(j));
        }
        REQUIRE(tracker);
      }
    }
    {
      const std::size_t l = 1;
      const std::size_t dimension = 1;
      const mgard::ConstituentRestriction<2, double> R(hierarchy, l, dimension);
      const std::array<std::array<std::size_t, 2>, 3> multiindices = {
          {{1, 0}, {2, 0}}};
      const std::array<std::array<double, 9>, 2> expecteds = {{
          {-9, -5, 1, 11.25, 3, 2.75, 9, 5, 6},
          {-9, -5, 1, 9, 3, 2, 12.75, 5, 7.25},
      }};
      for (std::size_t i = 0; i < 2; ++i) {
        const std::array<std::size_t, 2> &multiindex = multiindices.at(i);
        const std::array<double, 9> &expected = expecteds.at(i);
        std::array<double, 9> v_ = u_;
        double *const v = v_.data();
        R(multiindex, v);
        TrialTracker tracker;
        for (std::size_t j = 0; j < 9; ++j) {
          tracker += v_.at(j) == Approx(expected.at(j));
        }
        REQUIRE(tracker);
      }
    }
    {
      bool thrown = false;
      try {
        const mgard::ConstituentRestriction<2, double> R(hierarchy, 0, 0);
      } catch (...) {
        // The constructor will throw a `std::invalid_argument` exception, but
        // (as of this writing) the attempt to initialize `coarse_indices`
        // throws first.
        thrown = true;
      }
      REQUIRE(thrown);
    }
  }
}
