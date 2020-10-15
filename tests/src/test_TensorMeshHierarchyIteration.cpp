#include "catch2/catch.hpp"

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

// We expect `It` to be `mgard::TensorNodeRange<2>::iterator` or
// `mgard::TensorReservedNodeRange<2>::iterator`. Quick efforts to
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
      const mgard::TensorNodeRange<2, float> nodes = hierarchy.nodes(1);
      mgard::TensorNodeRange<2, float>::iterator p = nodes.begin();

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
      const mgard::TensorNodeRange<2, float> nodes = hierarchy.nodes(0);
      mgard::TensorNodeRange<2, float>::iterator p = nodes.begin();
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
      const mgard::TensorReservedNodeRange<2, float> nodes(hierarchy, 1);
      mgard::TensorReservedNodeRange<2, float>::iterator p = nodes.begin();

      TrialTracker tracker;
      increment_and_test_neighbors(tracker, p, {0, 0}, {0, 0}, {0, 0}, {2, 0},
                                   {0, 2});
      increment_and_test_neighbors(tracker, p, {0, 1}, {0, 1}, {0, 0}, {1, 1},
                                   {0, 2});
      increment_and_test_neighbors(tracker, p, {0, 2}, {0, 2}, {0, 0}, {2, 2},
                                   {0, 2});
      increment_and_test_neighbors(tracker, p, {1, 0}, {0, 0}, {1, 0}, {2, 0},
                                   {1, 1});
      increment_and_test_neighbors(tracker, p, {1, 1}, {0, 1}, {1, 0}, {2, 1},
                                   {1, 2});
      increment_and_test_neighbors(tracker, p, {1, 2}, {0, 2}, {1, 1}, {2, 2},
                                   {1, 2});
      increment_and_test_neighbors(tracker, p, {2, 0}, {0, 0}, {2, 0}, {2, 0},
                                   {2, 2});
      increment_and_test_neighbors(tracker, p, {2, 1}, {1, 1}, {2, 0}, {2, 1},
                                   {2, 2});
      increment_and_test_neighbors(tracker, p, {2, 2}, {0, 2}, {2, 0}, {2, 2},
                                   {2, 2});
      REQUIRE(tracker);
    }

    // Coarse level.
    {
      const mgard::TensorReservedNodeRange<2, float> nodes(hierarchy, 0);
      mgard::TensorReservedNodeRange<2, float>::iterator p = nodes.begin();
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
