#include "catch2/catch_test_macros.hpp"

#include <cstddef>

#include <algorithm>
#include <array>
#include <iterator>
#include <vector>

#include "testing_utilities.hpp"

#include "utilities.hpp"

TEST_CASE("PseudoArray iteration", "[utilities]") {
  int values[10] = {0, 1, 4, 9, 16, 25, 36, 0, -2, 1};

  SECTION("comparison with manual iteration") {
    const std::size_t ns[5] = {0, 1, 5, 9, 10};
    for (const std::size_t n : ns) {
      // Originally we used manual loops here. Now it's almost tautological.
      const mgard::PseudoArray v(values, n);
      REQUIRE(std::equal(values, values + n, v.begin()));
    }
  }

  SECTION("signed lengths") {
    const std::size_t ns[2] = {0, 8};
    for (const std::size_t n : ns) {
      const mgard::PseudoArray u(values, n);
      const mgard::PseudoArray v(values, static_cast<int>(n));
      REQUIRE(std::equal(u.begin(), u.end(), v.begin()));
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
  for (const auto pair :
       mgard::Enumeration<std::vector<float>::const_iterator>(xs)) {
    indices.push_back(pair.index);
    values.push_back(pair.value);
  }
  const std::vector<std::size_t> expected_indices = {0, 1, 2, 3};
  REQUIRE(indices == expected_indices);
  REQUIRE(values == xs);

  // This compiles and we never execute the body of the loop.
  const std::vector<int> ys;
  for (const auto pair :
       mgard::Enumeration<std::vector<int>::const_iterator>(ys)) {
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
  TrialTracker tracker;
  using It = T::const_iterator;
  using Jt = U::const_iterator;
  for (const auto pair : mgard::ZippedRange<It, Jt>(xs, ys)) {
    tracker += pair.first == xs.at(i) && pair.second == ys.at(i);
    ++i;
  }
  REQUIRE(tracker);
}

TEST_CASE("RangeSlice iteration", "[utilities]") {
  const std::array<int, 8> xs = {2, 3, 5, 7, 11, 13, 17, 19};
  using It = std::array<int, 8>::const_iterator;
  const mgard::RangeSlice<It> slice{xs.begin() + 2, xs.end() - 2};
  const std::vector<int> middle(slice.begin(), slice.end());
  const std::vector<int> expected_middle = {5, 7, 11, 13};
  REQUIRE(middle == expected_middle);
}

TEST_CASE("CartesianProduct iterator", "[utilities]") {
  {
    const std::vector<int> a = {1, 3, 5};
    const std::vector<int> b = {2, 4, 6};
    const std::vector<int> c = {0, 0};
    const mgard::CartesianProduct<std::vector<int>, 3> product({a, b, c});
    REQUIRE(std::distance(product.begin(), product.end()) == 18);
  }

  {
    const std::vector<char> a = {'a', 'b', 'c', 'd'};
    const std::vector<char> b = {'y', 'z'};
    mgard::CartesianProduct<std::vector<char>, 2> product({a, b});
    const std::vector<std::array<char, 2>> obtained(product.begin(),
                                                    product.end());
    const std::vector<std::array<char, 2>> expected = {
        {'a', 'y'}, {'a', 'z'}, {'b', 'y'}, {'b', 'z'},
        {'c', 'y'}, {'c', 'z'}, {'d', 'y'}, {'d', 'z'}};
    REQUIRE(obtained == expected);
  }
}

TEST_CASE("CartesianProduct predecessors and successors", "[utilities]") {
  const std::vector<char> a = {'a', 'b', 'c'};
  const std::vector<char> b = {'d', 'e'};
  const mgard::CartesianProduct<std::vector<char>, 2> product({a, b});

  TrialTracker tracker;

  using It = mgard::CartesianProduct<std::vector<char>, 2>::iterator;
  std::vector<It> iterators;
  for (It it = product.begin(); it != product.end(); ++it) {
    iterators.push_back(it);
  }

  It p = product.begin();
  tracker += p == iterators.at(0);
  tracker += p.predecessor(0) == iterators.at(0);
  tracker += p.predecessor(1) == iterators.at(0);
  tracker += p.successor(0) == iterators.at(2);
  tracker += p.successor(1) == iterators.at(1);

  ++p;
  tracker += p == iterators.at(1);
  tracker += p.predecessor(0) == iterators.at(1);
  tracker += p.predecessor(1) == iterators.at(0);
  tracker += p.successor(0) == iterators.at(3);
  tracker += p.successor(1) == iterators.at(1);

  ++p;
  tracker += p == iterators.at(2);
  tracker += p.predecessor(0) == iterators.at(0);
  tracker += p.predecessor(1) == iterators.at(2);
  tracker += p.successor(0) == iterators.at(4);
  tracker += p.successor(1) == iterators.at(3);

  ++p;
  tracker += p == iterators.at(3);
  tracker += p.predecessor(0) == iterators.at(1);
  tracker += p.predecessor(1) == iterators.at(2);
  tracker += p.successor(0) == iterators.at(5);
  tracker += p.successor(1) == iterators.at(3);

  ++p;
  tracker += p == iterators.at(4);
  tracker += p.predecessor(0) == iterators.at(2);
  tracker += p.predecessor(1) == iterators.at(4);
  tracker += p.successor(0) == iterators.at(4);
  tracker += p.successor(1) == iterators.at(5);

  ++p;
  tracker += p == iterators.at(5);
  tracker += p.predecessor(0) == iterators.at(3);
  tracker += p.predecessor(1) == iterators.at(4);
  tracker += p.successor(0) == iterators.at(5);
  tracker += p.successor(1) == iterators.at(5);

  ++p;
  tracker += p == product.end();

  REQUIRE(tracker);
}

namespace {

void test_bit_equality(const mgard::Bits &bits,
                       const std::vector<bool> &expected) {
  TrialTracker tracker;
  std::vector<bool>::const_iterator p = expected.begin();
  for (const bool b : bits) {
    tracker += b == *p++;
  }
  REQUIRE(tracker);
}

} // namespace

TEST_CASE("Bits iteration", "[utilities]") {
  SECTION("zero end offsets") {
    {
      unsigned char const a[1]{0x3d};
      const mgard::Bits bits(a, a + 1);
      const std::vector<bool> expected{// `3`.
                                       false, false, true, true,
                                       // `d`.
                                       true, true, false, true};
      test_bit_equality(bits, expected);
    }
    {
      unsigned char const a[2]{0xe6, 0x0a};
      const mgard::Bits bits(a, a + 2);
      const std::vector<bool> expected{// `e`.
                                       true, true, true, false,
                                       // `6`.
                                       false, true, true, false,
                                       // `0`.
                                       false, false, false, false,
                                       // `a`.
                                       true, false, true, false};
      test_bit_equality(bits, expected);
    }
    {
      unsigned char const a[3]{0x12, 0x0c, 0xff};
      const mgard::Bits bits(a, a + 3);
      const std::vector<bool> expected{// `1`.
                                       false, false, false, true,
                                       // `2`.
                                       false, false, true, false,
                                       // `0`.
                                       false, false, false, false,
                                       // `c`.
                                       true, true, false, false,
                                       // `f`.
                                       true, true, true, true,
                                       // `f`.
                                       true, true, true, true};
      test_bit_equality(bits, expected);
    }
  }
}
