#include "catch2/catch.hpp"

#include <array>
#include <vector>

#include "testing_utilities.hpp"

#include "TensorMeshHierarchy.hpp"

TEST_CASE("TensorMeshHierarchy construction", "[TensorMeshHierarchy]") {
  {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({17});
    REQUIRE(hierarchy.L == 4);
    std::vector<std::size_t> ndofs(hierarchy.L + 1);
    for (std::size_t l = 0; l <= hierarchy.L; ++l) {
      ndofs.at(l) = hierarchy.ndof(l);
    }
    REQUIRE(ndofs.back() == hierarchy.ndof());
    const std::vector<std::size_t> expected = {2, 3, 5, 9, 17};
    REQUIRE(ndofs == expected);
  }
  {
    const mgard::TensorMeshHierarchy<2, double> hierarchy({12, 39});
    REQUIRE(hierarchy.L == 4);
    std::vector<std::array<std::size_t, 2>> shapes(hierarchy.L + 1);
    for (std::size_t l = 0; l <= hierarchy.L; ++l) {
      shapes.at(l) = hierarchy.meshes.at(l).shape;
    }
    const std::vector<std::array<std::size_t, 2>> expected = {
        {2, 5}, {3, 9}, {5, 17}, {9, 33}, {12, 39}};
    REQUIRE(shapes == expected);
  }
  {
    const mgard::TensorMeshHierarchy<3, double> hierarchy({15, 6, 129});
    REQUIRE(hierarchy.L == 3);
    std::vector<std::array<std::size_t, 3>> shapes(hierarchy.L + 1);
    for (std::size_t l = 0; l <= hierarchy.L; ++l) {
      shapes.at(l) = hierarchy.meshes.at(l).shape;
    }
    // Note that the final dimension doesn't begin decreasing until every index
    // is of the form `2^k + 1`.
    const std::vector<std::array<std::size_t, 3>> expected = {
        {3, 2, 33}, {5, 3, 65}, {9, 5, 129}, {15, 6, 129}};
    REQUIRE(shapes == expected);
  }
}

TEST_CASE("TensorMeshHierarchy dimension indexing", "[TensorMeshHierarchy]") {
  {
    const mgard::TensorMeshHierarchy<3, float> hierarchy({5, 3, 17});
    REQUIRE(hierarchy.L == 1);
    // Every index is included in the finest level.
    for (std::size_t i = 0; i < 3; ++i) {
      const std::vector<std::size_t> indices = hierarchy.indices(1, i);
      TrialTracker tracker;
      std::size_t expected = 0;
      for (const std::size_t index : indices) {
        tracker += index == expected++;
      }
      REQUIRE(tracker);
    }
    // On the coarse level, we get every other index.
    {
      // Had a problem with brace initialization when this was an `array`.
      const std::array<std::vector<std::size_t>, 3> expected = {
          {{0, 2, 4}, {0, 2}, {0, 2, 4, 6, 8, 10, 12, 14, 16}}};
      std::array<std::vector<std::size_t>, 3> obtained;
      for (std::size_t i = 0; i < 3; ++i) {
        obtained.at(i) = hierarchy.indices(0, i);
      }
      REQUIRE(obtained == expected);
    }
  }

  {
    const mgard::TensorMeshHierarchy<2, double> hierarchy({5, 6});
    REQUIRE_THROWS(hierarchy.indices(0, 0));
  }
}
