#include "catch2/catch.hpp"

#include <cstddef>

#include <algorithm>
#include <array>
#include <vector>

#include "utilities.hpp"

TEST_CASE("PseudoArray iteration", "[utilities]") {
  int values[10] = {0, 1, 4, 9, 16, 25, 36, 0, -2, 1};

  SECTION("comparison with manual iteration") {
    const std::size_t ns[5] = {0, 1, 5, 9, 10};
    for (std::size_t n : ns) {
      std::vector<int> manual;
      manual.reserve(n);
      for (std::size_t i = 0; i < n; ++i) {
        manual.push_back(values[i]);
      }

      std::vector<int> pseudo;
      pseudo.reserve(n);
      for (int value : mgard::PseudoArray(values, n)) {
        pseudo.push_back(value);
      }

      REQUIRE(std::equal(manual.begin(), manual.end(), pseudo.begin()));
    }
  }

  SECTION("signed lengths") {
    const std::size_t ns[2] = {0, 8};
    for (std::size_t n : ns) {
      std::vector<int> normal;
      normal.reserve(n);
      for (int value : mgard::PseudoArray(values, n)) {
        normal.push_back(value);
      }

      std::vector<int> integral;
      integral.reserve(n);
      for (int value : mgard::PseudoArray(values, static_cast<int>(n))) {
        integral.push_back(value);
      };

      REQUIRE(std::equal(normal.begin(), normal.end(), integral.begin()));
    }
  }

  SECTION("construction exceptions") {
    // There's no check against `p` being `NULL`.
    REQUIRE_NOTHROW(mgard::PseudoArray<int>(NULL, 1));

    // On the other hand, negative lengths are not allowed.
    REQUIRE_THROWS(mgard::PseudoArray<double>(NULL, -1));
  }
}

TEST_CASE("Enumeration iteration", "[utilities]") {
  const std::vector<float> xs = {-1.375, 0, 732.5, -0.875};
  std::vector<std::size_t> indices;
  std::vector<float> values;
  for (auto pair : mgard::Enumeration<std::vector<float>::const_iterator>(xs)) {
    indices.push_back(pair.first);
    values.push_back(pair.second);
  }
  const std::vector<std::size_t> expected_indices = {0, 1, 2, 3};
  REQUIRE(indices == expected_indices);
  REQUIRE(values == xs);

  // This compiles and we never execute the body of the loop.
  const std::vector<int> ys;
  for (auto pair : mgard::Enumeration<std::vector<int>::const_iterator>(ys)) {
    // Using `pair` so the compiler doesn't complain.
    static_cast<void>(pair);
    REQUIRE(false);
  }
}

TEST_CASE("ZippedRange iteration", "[utilities]") {
  using T = std::vector<float>;
  using U = std::array<unsigned short int, 5>;
  const T xs = {-3.28, 17.37, 0, 0.2388, -99.1};
  const U ys = {12, 0, 0, 77, 3};
  std::size_t i = 0;
  bool all_equal = true;
  using It = T::const_iterator;
  using Jt = U::const_iterator;
  for (auto pair : mgard::ZippedRange<It, Jt>(xs, ys)) {
    all_equal = all_equal && pair.first == xs.at(i) && pair.second == ys.at(i);
    ++i;
  }
  REQUIRE(all_equal);
}

TEST_CASE("RangeSlice iteration", "[utilities]") {
  const std::array<int, 8> xs = {2, 3, 5, 7, 11, 13, 17, 19};
  std::vector<int> middle;
  using It = std::array<int, 8>::const_iterator;
  for (const int x : mgard::RangeSlice<It> {xs.begin() + 2, xs.end() - 2}) {
    middle.push_back(x);
  }
  const std::vector<int> expected_middle = {5, 7, 11, 13};
  REQUIRE(middle == expected_middle);
}
