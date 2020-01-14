#include "catch2/catch.hpp"

#include <cstddef>

#include <algorithm>
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
