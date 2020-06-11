#include "catch2/catch.hpp"

#include <array>
#include <numeric>

#include "testing_utilities.hpp"

#include "MultidimensionalArray.hpp"

TEST_CASE("ndarray construction and size calculation",
          "[MultidimensionalArray]") {
  const mgard::MultidimensionalArray<int, 3> u(NULL, {3, 5, 7});
  const mgard::MultidimensionalArray<int, 3> v(NULL, {3, 5, 7}, 2);
  REQUIRE(u.size() == 105);
  REQUIRE(v.size() == 105);
}

TEST_CASE("ndarray indexing", "[MultidimensionalArray]") {
  std::array<int, 48> values;
  std::iota(values.begin(), values.end(), 5);

  {
    const mgard::MultidimensionalArray<int, 3> u(values.data(), {6, 2, 4});
    REQUIRE(u.at({0, 1, 0}) == 9);
    REQUIRE(u.at({0, 1, 1}) == 10);
    REQUIRE(u.at({5, 0, 2}) == 47);
    REQUIRE(u.at({5, 1, 2}) == 51);
    REQUIRE(u.at({2, 1, 1}) == 26);
    REQUIRE(u.at({3, 1, 1}) == 34);
    REQUIRE_THROWS(u.at({3, 2, 3}));
  }

  {
    const mgard::MultidimensionalArray<int, 1> u(values.data(), {15});
    struct TrialTracker tracker;
    for (std::size_t i = 0; i < 15; ++i) {
      tracker += values.at(i) == u.at({i});
    }
    REQUIRE(tracker);
  }

  {
    const mgard::MultidimensionalArray<int, 1> u(values.data() + 3, {3}, 4);
    struct TrialTracker tracker;
    for (std::size_t i = 0; i < 3; ++i) {
      tracker += u.at({i}) == 8 + 4 * static_cast<int>(i);
    }
    REQUIRE(tracker);
  }

  {
    const mgard::MultidimensionalArray<int, 2> u(values.data(), {4, 4}, 3);
    std::array<int, 16> obtained;
    std::array<int, 16>::iterator p = obtained.begin();
    std::array<std::size_t, 2> multiindex({0, 0});
    for (std::size_t i = 0; i < 4; ++i) {
      for (std::size_t j = 0; j < 4; ++j) {
        *p++ = u.at({i, j});
      }
    }
    const std::array<int, 16> expected = {5,  8,  11, 14, 17, 20, 23, 26,
                                          29, 32, 35, 38, 41, 44, 47, 50};
    REQUIRE(obtained == expected);
  }
}
