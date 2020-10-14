#include "catch2/catch.hpp"

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
    indices.push_back(pair.index);
    values.push_back(pair.value);
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
  TrialTracker tracker;
  using It = T::const_iterator;
  using Jt = U::const_iterator;
  for (auto pair : mgard::ZippedRange<It, Jt>(xs, ys)) {
    tracker += pair.first == xs.at(i) && pair.second == ys.at(i);
    ++i;
  }
  REQUIRE(tracker);
}

TEST_CASE("RangeSlice iteration", "[utilities]") {
  const std::array<int, 8> xs = {2, 3, 5, 7, 11, 13, 17, 19};
  std::vector<int> middle;
  using It = std::array<int, 8>::const_iterator;
  for (const int x : mgard::RangeSlice<It>{xs.begin() + 2, xs.end() - 2}) {
    middle.push_back(x);
  }
  const std::vector<int> expected_middle = {5, 7, 11, 13};
  REQUIRE(middle == expected_middle);
}

TEST_CASE("MultiindexRectangle iteration", "[utilities]") {
  {
    const mgard::MultiindexRectangle<2> rectangle({1, 1}, {2, 3});
    mgard::RangeSlice<mgard::MultiindexRectangle<2>::iterator> iterable =
        rectangle.indices(1);
    const std::vector<std::array<std::size_t, 2>> visited(iterable.begin(),
                                                          iterable.end());
    const std::vector<std::array<std::size_t, 2>> expected = {
        {1, 1}, {1, 2}, {1, 3}, {2, 1}, {2, 2}, {2, 3}};
    REQUIRE(visited == expected);
  }

  {
    const mgard::MultiindexRectangle<1> rectangle({5});
    const mgard::RangeSlice<mgard::MultiindexRectangle<1>::iterator> iterable =
        rectangle.indices(3);
    const std::vector<std::array<std::size_t, 1>> visited(iterable.begin(),
                                                          iterable.end());
    const std::vector<std::array<std::size_t, 1>> expected = {{0}, {3}};
    REQUIRE(visited == expected);
  }

  {
    const mgard::MultiindexRectangle<3> rectangle({3, 3, 3});
    const mgard::RangeSlice<mgard::MultiindexRectangle<3>::iterator> iterable =
        rectangle.indices(2);
    const std::vector<std::array<std::size_t, 3>> visited(iterable.begin(),
                                                          iterable.end());
    const std::vector<std::array<std::size_t, 3>> expected = {
        {0, 0, 0}, {0, 0, 2}, {0, 2, 0}, {0, 2, 2},
        {2, 0, 0}, {2, 0, 2}, {2, 2, 0}, {2, 2, 2}};
    REQUIRE(visited == expected);
  }

  {
    const mgard::MultiindexRectangle<5> rectangle({60, 88, 60, 53, 26});
    const mgard::RangeSlice<mgard::MultiindexRectangle<5>::iterator> iterable =
        rectangle.indices(100);
    REQUIRE(++iterable.begin() == iterable.end());
  }

  {
    // Checking the edge case.
    const mgard::MultiindexRectangle<0> rectangle({});
    const mgard::RangeSlice<mgard::MultiindexRectangle<0>::iterator> iterable =
        rectangle.indices(1);
    REQUIRE(iterable.begin() == iterable.end());
  }

  {
    const mgard::MultiindexRectangle<2> rectangle({10, 10}, {15, 15});
    REQUIRE_THROWS(rectangle.indices(0));
  }
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
