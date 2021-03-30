#include "catch2/catch_test_macros.hpp"

#include <array>
#include <vector>

#include "testing_utilities.hpp"

#include "TensorMeshHierarchy.hpp"
#include "TensorMeshHierarchyIteration.hpp"
#include "utilities.hpp"

TEST_CASE("TensorIndexRange size and iteration",
          "[TensorMeshHierarchyIteration]") {
  // Dyadic.
  {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({17});

    {
      const mgard::TensorIndexRange range(hierarchy, 4, 0);
      REQUIRE(range.size() == 17);
      TrialTracker tracker;
      std::size_t expected = 0;
      for (const std::size_t i : range) {
        tracker += i == expected;
        expected += 1;
      }
      REQUIRE(tracker);
    }

    {
      const mgard::TensorIndexRange range(hierarchy, 2, 0);
      REQUIRE(range.size() == 5);
      TrialTracker tracker;
      std::size_t expected = 0;
      for (const std::size_t i : range) {
        tracker += i == expected;
        expected += 4;
      }
      REQUIRE(tracker);
    }
  }

  // Nondyadic.
  {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({10});

    {
      const mgard::TensorIndexRange range(hierarchy, 4, 0);
      REQUIRE(range.size() == 10);
      TrialTracker tracker;
      std::size_t expected = 0;
      for (const std::size_t i : range) {
        tracker += i == expected;
        expected += 1;
      }
      REQUIRE(tracker);
    }

    {
      const mgard::TensorIndexRange range(hierarchy, 2, 0);
      REQUIRE(range.size() == 5);
      TrialTracker tracker;
      const std::vector<std::size_t> expected = {0, 2, 4, 6, 9};
      const std::vector<std::size_t> observed(range.begin(), range.end());
      REQUIRE(expected == observed);
    }
  }
}

namespace {

// We expect `It` to be `mgard::UnshuffledTensorNodeRange<2>::iterator` or
// `mgard::ShuffledTensorNodeRange<2>::iterator`. Quick efforts to
// generalize this more (parametrizing on `N` and so on) have led to problems
// getting the compiler to find this when looking for matching functions.
template <typename It>
void increment_and_test_neighbors(
    TrialTracker &tracker, It &p,
    const std::array<std::size_t, 2> exp_multiindex,
    const std::array<std::size_t, 2> exp_pred_0_multiindex,
    const std::array<std::size_t, 2> exp_pred_1_multiindex,
    const std::array<std::size_t, 2> exp_succ_0_multiindex,
    const std::array<std::size_t, 2> exp_succ_1_multiindex) {
  const mgard::TensorNode<2> node = *p++;
  tracker += node.multiindex == exp_multiindex;
  tracker += node.predecessor(0).multiindex == exp_pred_0_multiindex;
  tracker += node.predecessor(1).multiindex == exp_pred_1_multiindex;
  tracker += node.successor(0).multiindex == exp_succ_0_multiindex;
  tracker += node.successor(1).multiindex == exp_succ_1_multiindex;
}

} // namespace

TEST_CASE("TensorNode predecessors and successors",
          "[TensorMeshHierarchyIteration]") {
  const mgard::TensorMeshHierarchy<2, float> hierarchy({3, 3});
  SECTION("'normal' nodes") {
    // Finest level.
    {
      const mgard::UnshuffledTensorNodeRange<2, float> nodes(hierarchy, 1);
      mgard::UnshuffledTensorNodeRange<2, float>::iterator p = nodes.begin();

      TrialTracker tracker;
      increment_and_test_neighbors(tracker, p, {0, 0}, {0, 0}, {0, 0}, {1, 0},
                                   {0, 1});
      increment_and_test_neighbors(tracker, p, {0, 1}, {0, 1}, {0, 0}, {1, 1},
                                   {0, 2});
      increment_and_test_neighbors(tracker, p, {0, 2}, {0, 2}, {0, 1}, {1, 2},
                                   {0, 2});
      increment_and_test_neighbors(tracker, p, {1, 0}, {0, 0}, {1, 0}, {2, 0},
                                   {1, 1});
      increment_and_test_neighbors(tracker, p, {1, 1}, {0, 1}, {1, 0}, {2, 1},
                                   {1, 2});
      increment_and_test_neighbors(tracker, p, {1, 2}, {0, 2}, {1, 1}, {2, 2},
                                   {1, 2});
      increment_and_test_neighbors(tracker, p, {2, 0}, {1, 0}, {2, 0}, {2, 0},
                                   {2, 1});
      increment_and_test_neighbors(tracker, p, {2, 1}, {1, 1}, {2, 0}, {2, 1},
                                   {2, 2});
      increment_and_test_neighbors(tracker, p, {2, 2}, {1, 2}, {2, 1}, {2, 2},
                                   {2, 2});
      REQUIRE(tracker);
    }

    // Coarse level.
    {
      const mgard::UnshuffledTensorNodeRange<2, float> nodes(hierarchy, 0);
      mgard::UnshuffledTensorNodeRange<2, float>::iterator p = nodes.begin();
      TrialTracker tracker;
      increment_and_test_neighbors(tracker, p, {0, 0}, {0, 0}, {0, 0}, {2, 0},
                                   {0, 2});
      increment_and_test_neighbors(tracker, p, {0, 2}, {0, 2}, {0, 0}, {2, 2},
                                   {0, 2});
      increment_and_test_neighbors(tracker, p, {2, 0}, {0, 0}, {2, 0}, {2, 0},
                                   {2, 2});
      increment_and_test_neighbors(tracker, p, {2, 2}, {0, 2}, {2, 0}, {2, 2},
                                   {2, 2});
      REQUIRE(tracker);
    }
  }

  SECTION("'reserved' nodes") {
    // Finest level.
    {
      const mgard::ShuffledTensorNodeRange<2, float> nodes(hierarchy, 1);
      mgard::ShuffledTensorNodeRange<2, float>::iterator p = nodes.begin();

      TrialTracker tracker;
      increment_and_test_neighbors(tracker, p, {0, 0}, {0, 0}, {0, 0}, {2, 0},
                                   {0, 2});
      increment_and_test_neighbors(tracker, p, {0, 2}, {0, 2}, {0, 0}, {2, 2},
                                   {0, 2});
      increment_and_test_neighbors(tracker, p, {2, 0}, {0, 0}, {2, 0}, {2, 0},
                                   {2, 2});
      increment_and_test_neighbors(tracker, p, {2, 2}, {0, 2}, {2, 0}, {2, 2},
                                   {2, 2});
      increment_and_test_neighbors(tracker, p, {0, 1}, {0, 1}, {0, 0}, {1, 1},
                                   {0, 2});
      increment_and_test_neighbors(tracker, p, {1, 0}, {0, 0}, {1, 0}, {2, 0},
                                   {1, 1});
      increment_and_test_neighbors(tracker, p, {1, 1}, {0, 1}, {1, 0}, {2, 1},
                                   {1, 2});
      increment_and_test_neighbors(tracker, p, {1, 2}, {0, 2}, {1, 1}, {2, 2},
                                   {1, 2});
      increment_and_test_neighbors(tracker, p, {2, 1}, {1, 1}, {2, 0}, {2, 1},
                                   {2, 2});
      REQUIRE(tracker);
    }

    // Coarse level.
    {
      const mgard::ShuffledTensorNodeRange<2, float> nodes(hierarchy, 0);
      mgard::ShuffledTensorNodeRange<2, float>::iterator p = nodes.begin();
      TrialTracker tracker;
      increment_and_test_neighbors(tracker, p, {0, 0}, {0, 0}, {0, 0}, {2, 0},
                                   {0, 2});
      increment_and_test_neighbors(tracker, p, {0, 2}, {0, 2}, {0, 0}, {2, 2},
                                   {0, 2});
      increment_and_test_neighbors(tracker, p, {2, 0}, {0, 0}, {2, 0}, {2, 0},
                                   {2, 2});
      increment_and_test_neighbors(tracker, p, {2, 2}, {0, 2}, {2, 0}, {2, 2},
                                   {2, 2});
      REQUIRE(tracker);
    }
  }
}

// See `test_TensorMeshHierarchy.cpp` for more `TensorNode` iteration tests.

namespace {

template <std::size_t N>
void test_shuffled_dereferencing(
    const std::array<std::size_t, N> shape,
    const std::vector<std::array<std::size_t, N>> &expected) {
  const mgard::TensorMeshHierarchy<N, float> hierarchy(shape);
  std::vector<std::array<std::size_t, N>> obtained(hierarchy.ndof());
  const mgard::ShuffledTensorNodeRange nodes(hierarchy, hierarchy.L);
  std::transform(
      nodes.begin(), nodes.end(), obtained.begin(),
      [](const mgard::TensorNode<N> &node) -> std::array<std::size_t, N> {
        return node.multiindex;
      });
  REQUIRE(obtained == expected);
}

} // namespace

TEST_CASE("ShuffledTensorNodeRange dereferencing",
          "[TensorMeshHierarchyIteration]") {
  SECTION("1D") {
    const std::vector<std::array<std::size_t, 1>> expected = {{0}, {5}, {2},
                                                              {1}, {3}, {4}};
    test_shuffled_dereferencing<1>({6}, expected);
  }

  SECTION("2D") {
    const std::vector<std::array<std::size_t, 2>> expected = {
        {0, 0}, {0, 2}, {0, 4}, {3, 0}, {3, 2}, {3, 4}, {0, 1},
        {0, 3}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {3, 1},
        {3, 3}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}};
    test_shuffled_dereferencing<2>({4, 5}, expected);
  }

  SECTION("3D") {
    const std::vector<std::array<std::size_t, 3>> expected = {
        {0, 0, 0}, {0, 0, 2}, {0, 2, 0}, {0, 2, 2}, {2, 0, 0}, {2, 0, 2},
        {2, 2, 0}, {2, 2, 2}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2},
        {0, 2, 1}, {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, {1, 1, 1},
        {1, 1, 2}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2}, {2, 0, 1}, {2, 1, 0},
        {2, 1, 1}, {2, 1, 2}, {2, 2, 1}};
    test_shuffled_dereferencing<3>({3, 3, 3}, expected);
  }
}
