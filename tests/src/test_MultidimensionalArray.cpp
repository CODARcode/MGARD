#include "catch2/catch_test_macros.hpp"

#include <array>
#include <numeric>
#include <vector>

#include "testing_utilities.hpp"

#include "MultidimensionalArray.hpp"

static bool operator==(const mgard::MultidimensionalArray<int, 1> u,
                       const std::vector<int> expected) {
  const std::size_t N = u.size();
  if (N != expected.size()) {
    return false;
  }
  TrialTracker tracker;
  for (std::size_t i = 0; i < N; ++i) {
    tracker += u.at({i}) == expected.at(i);
  }
  return static_cast<bool>(tracker);
}

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

TEST_CASE("ndarray slicing", "[MultidimensionalArray]") {
  std::array<int, 18> values = {7, 0, 5, 4, 8, 6, 2, 6, 8,
                                4, 2, 4, 1, 6, 1, 1, 0, 6};

  {
    const mgard::MultidimensionalArray<int, 3> u(values.data(), {3, 2, 3});

    {
      const std::vector<int> expected = {4, 4, 1};
      REQUIRE(u.slice({0, 1, 0}, 0) == expected);
    }
    {
      const std::vector<int> expected = {5, 8, 1};
      REQUIRE(u.slice({0, 0, 2}, 0) == expected);
    }

    {
      const std::vector<int> expected = {5, 6};
      REQUIRE(u.slice({0, 0, 2}, 1) == expected);
    }
    {
      const std::vector<int> expected = {6, 0};
      REQUIRE(u.slice({2, 0, 1}, 1) == expected);
    }

    {
      const std::vector<int> expected = {1, 0, 6};
      REQUIRE(u.slice({2, 1, 0}, 2) == expected);
    }
    {
      const std::vector<int> expected = {7, 0, 5};
      REQUIRE(u.slice({0, 0, 0}, 2) == expected);
    }

    REQUIRE_THROWS(u.slice({1, 1, 1}, 0));
  }

  {
    const mgard::MultidimensionalArray<int, 2> u(values.data(), {2, 2}, 2);

    {
      const std::vector<int> expected = {7, 5};
      REQUIRE(u.slice({0, 0}, 1) == expected);
    }
    {
      const std::vector<int> expected = {8, 2};
      REQUIRE(u.slice({1, 0}, 1) == expected);
    }

    {
      const std::vector<int> expected = {7, 8};
      REQUIRE(u.slice({0, 0}, 0) == expected);
    }
    {
      const std::vector<int> expected = {5, 2};
      REQUIRE(u.slice({0, 1}, 0) == expected);
    }

    REQUIRE_THROWS(u.slice({2, 2}, 0));
  }
}
