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

    const mgard::TensorMeshLevel<2, double> &MESH = hierarchy.meshes.back();
    TrialTracker tracker;
    for (std::size_t i = 0; i < 2; ++i) {
      const std::vector<double> &xs = hierarchy.coordinates.at(i);
      const std::size_t n = MESH.shape.at(i);
      for (std::size_t j = 0; j < n; ++j) {
        tracker += xs.at(j) == Approx(static_cast<double>(j) / (n - 1));
      }
    }
    REQUIRE(tracker);
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
  {
    const mgard::TensorMeshHierarchy<3, float> hierarchy(
        {5, 3, 2}, {{{0.0, 0.5, 0.75, 1.0, 1.25}, {-3, -2, -1}, {10.5, 9.5}}});
    REQUIRE_NOTHROW(hierarchy);
  }
  {
    bool thrown = false;
    try {
      const mgard::TensorMeshHierarchy<2, double> hierarchy(
          {10, 5}, {{{1, 2, 3}, {1, 2, 3, 4, 5}}});
    } catch (const std::invalid_argument &exception) {
      thrown = true;
    }
    REQUIRE(thrown);
  }
}

TEST_CASE("TensorMeshHierarchy dimension indexing", "[TensorMeshHierarchy]") {
  // Currently we don't test `TensorMeshHierarchy::at`.
  SECTION("offset helper member") {
    {
      const mgard::TensorMeshHierarchy<1, float> hierarchy({6});
      REQUIRE(hierarchy.offset({0}) == 0);
      REQUIRE(hierarchy.offset({2}) == 2);
      REQUIRE(hierarchy.offset({5}) == 5);
    }
    {
      const mgard::TensorMeshHierarchy<2, double> hierarchy({13, 4});
      REQUIRE(hierarchy.offset({0, 3}) == 3);
      REQUIRE(hierarchy.offset({10, 1}) == 41);
      REQUIRE(hierarchy.offset({12, 3}) == 51);
    }
    {
      const mgard::TensorMeshHierarchy<4, float> hierarchy({3, 3, 2, 100});
      REQUIRE(hierarchy.offset({1, 2, 1, 90}) == 1190);
      REQUIRE(hierarchy.offset({2, 0, 1, 15}) == 1315);
      REQUIRE(hierarchy.offset({2, 2, 1, 27}) == 1727);
    }
  }

  SECTION("indices helper member") {
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
}
