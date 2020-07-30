#include "catch2/catch.hpp"

#include <array>
#include <vector>

#include "testing_utilities.hpp"

#include "TensorMeshHierarchy.hpp"
#include "utilities.hpp"

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
      const std::array<std::vector<std::size_t>, 4> expected = {
          {{0, 5}, {0, 2, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4, 5}}};
      std::array<std::vector<std::size_t>, 4> obtained;
      for (std::size_t l = 0; l < 4; ++l) {
        obtained.at(l) = hierarchy.indices(l, 1);
      }
      REQUIRE(obtained == expected);
    }

    {
      const mgard::TensorMeshHierarchy<1, double> hierarchy({60});
      const std::array<std::vector<std::size_t>, 4> expected = {
          {{0, 59},
           {0, 29, 59},
           {0, 14, 29, 44, 59},
           {0, 7, 14, 22, 29, 36, 44, 51, 59}}};
      std::array<std::vector<std::size_t>, 4> obtained;
      for (std::size_t l = 0; l < 4; ++l) {
        obtained.at(l) = hierarchy.indices(l, 0);
      }
      REQUIRE(obtained == expected);
    }
  }
}

TEST_CASE("level values iteration", "[TensorMeshHierarchy]") {
  const std::size_t N = 27;
  std::vector<float> v_(N);
  std::iota(v_.begin(), v_.end(), 1);
  float *const v = v_.data();

  {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({17});
    std::vector<float> encountered_values;
    std::vector<std::size_t> encountered_ls;
    for (const mgard::SituatedCoefficient<1, float> coeff :
         hierarchy.on_nodes(v, 2)) {
      encountered_values.push_back(*coeff.value);
      encountered_ls.push_back(coeff.l);
    }
    const std::vector<float> expected_values = {1, 5, 9, 13, 17};
    const std::vector<std::size_t> expected_ls = {0, 2, 1, 2, 0};
    REQUIRE(encountered_values == expected_values);
    REQUIRE(encountered_ls == expected_ls);
  }

  {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({3, 5});
    std::vector<float> encountered;
    // For the indices.
    TrialTracker tracker;
    for (const mgard::SituatedCoefficient<2, float> coeff :
         hierarchy.on_nodes(v, 0)) {
      encountered.push_back(*coeff.value);
      tracker += coeff.l == 0;
    }
    const std::vector<float> expected = {1, 3, 5, 11, 13, 15};
    REQUIRE(encountered == expected);
    REQUIRE(tracker);
  }

  {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({9, 3});
    std::vector<float> encountered;
    // For the indices.
    TrialTracker tracker;
    for (const mgard::SituatedCoefficient<2, float> coeff :
         hierarchy.on_nodes(v, 0)) {
      encountered.push_back(*coeff.value);
      tracker += coeff.l == 0;
    }
    const std::vector<float> expected = {1, 3, 7, 9, 13, 15, 19, 21, 25, 27};
    REQUIRE(encountered == expected);
    REQUIRE(tracker);
  }

  {
    const mgard::TensorMeshHierarchy<3, float> hierarchy({3, 3, 3});
    float expected_value = 1;
    TrialTracker tracker;
    // For the indices.
    std::vector<std::size_t> encountered_ls;
    for (const mgard::SituatedCoefficient<3, float> coeff :
         hierarchy.on_nodes(v, hierarchy.L)) {
      tracker += *coeff.value == expected_value;
      expected_value += 1;
      encountered_ls.push_back(coeff.l);
    }
    std::vector<std::size_t> expected_ls(27, 1);
    for (const std::size_t index : {0, 2, 6, 8, 18, 20, 24, 26}) {
      expected_ls.at(index) = 0;
    }
    REQUIRE(tracker);
    REQUIRE(encountered_ls == expected_ls);
  }

  {
    const mgard::TensorMeshHierarchy<2, double> hierarchy({11, 14});
    std::vector<std::array<std::size_t, 2>> encountered_multiindices;
    std::vector<std::size_t> encountered_ls;
    for (const mgard::SituatedCoefficient<2, double> coeff :
         hierarchy.on_nodes(static_cast<double *const>(nullptr), 2)) {
      encountered_multiindices.push_back(coeff.multiindex);
      encountered_ls.push_back(coeff.l);
    }
    const std::vector<std::array<std::size_t, 2>> expected_multiindices = {
        {{0, 0}},  {{0, 3}},  {{0, 6}},  {{0, 9}},  {{0, 13}},
        {{2, 0}},  {{2, 3}},  {{2, 6}},  {{2, 9}},  {{2, 13}},
        {{5, 0}},  {{5, 3}},  {{5, 6}},  {{5, 9}},  {{5, 13}},
        {{7, 0}},  {{7, 3}},  {{7, 6}},  {{7, 9}},  {{7, 13}},
        {{10, 0}}, {{10, 3}}, {{10, 6}}, {{10, 9}}, {{10, 13}}};
    const std::vector<std::size_t> expected_ls = {0, 2, 1, 2, 0, 2, 2, 2, 2,
                                                  2, 1, 2, 1, 2, 1, 2, 2, 2,
                                                  2, 2, 0, 2, 1, 2, 0};
    REQUIRE(encountered_multiindices == expected_multiindices);
    REQUIRE(encountered_ls == expected_ls);
  }
}

TEST_CASE("dates of birth", "[TensorMeshHierarchy]") {
  {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({9});
    std::vector<std::size_t> encountered;
    for (std::size_t i = 0; i < 9; ++i) {
      encountered.push_back(hierarchy.date_of_birth({i}));
    }
    const std::vector<std::size_t> expected = {0, 3, 2, 3, 1, 3, 2, 3, 0};
    REQUIRE(encountered == expected);
  }

  {
    const mgard::TensorMeshHierarchy<2, double> hierarchy({6, 3});
    std::vector<std::size_t> encountered;
    const mgard::MultiindexRectangle<2> multiindices(
        hierarchy.meshes.at(hierarchy.L).shape);
    for (const std::array<std::size_t, 2> multiindex :
         multiindices.indices(1)) {
      encountered.push_back(hierarchy.date_of_birth(multiindex));
    }
    const std::vector<std::size_t> expected = {0, 1, 0, 1, 1, 1, 0, 1, 0,
                                               1, 1, 1, 2, 2, 2, 0, 1, 0};
    REQUIRE(encountered == expected);
  }

  {
    const mgard::TensorMeshHierarchy<3, float> hierarchy({5, 17, 12});
    TrialTracker tracker;
    {
      tracker += hierarchy.date_of_birth({0, 4, 5}) == 0;
      tracker += hierarchy.date_of_birth({4, 8, 11}) == 0;

      tracker += hierarchy.date_of_birth({2, 14, 2}) == 1;
      tracker += hierarchy.date_of_birth({2, 16, 8}) == 1;

      tracker += hierarchy.date_of_birth({3, 9, 1}) == 2;
      tracker += hierarchy.date_of_birth({4, 8, 9}) == 2;

      tracker += hierarchy.date_of_birth({3, 8, 10}) == 3;
      tracker += hierarchy.date_of_birth({1, 0, 7}) == 3;
    }
    REQUIRE(tracker);
  }
}
