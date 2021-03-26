#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "testing_utilities.hpp"

#include "TensorMeshHierarchy.hpp"
#include "TensorMeshHierarchyIteration.hpp"
#include "shuffle.hpp"
#include "utilities.hpp"

TEST_CASE("hierarchy mesh shapes", "[TensorMeshHierarchy]") {
  {
    const std::array<std::size_t, 1> shape = {5};
    const mgard::TensorMeshHierarchy<1, float> hierarchy(shape);
    REQUIRE(hierarchy.L == 2);

    REQUIRE(hierarchy.shapes.back() == shape);
  }
  {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({11, 32});
    REQUIRE(hierarchy.L == 4);

    const std::array<std::size_t, 2> expected = {9, 17};
    REQUIRE(hierarchy.shapes.at(3) == expected);
  }
  {
    const std::array<std::size_t, 5> shape = {1, 257, 129, 129, 1};
    const mgard::TensorMeshHierarchy<5, float> hierarchy(shape);
    REQUIRE(hierarchy.L == 7);

    REQUIRE(hierarchy.shapes.back() == shape);
  }
  {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({6, 5});
    REQUIRE(hierarchy.L == 3);

    const std::array<std::size_t, 2> expected = {5, 5};
    REQUIRE(hierarchy.shapes.at(2) == expected);
  }

  REQUIRE_THROWS(mgard::TensorMeshHierarchy<3, float>({1, 1, 1}));
  REQUIRE_THROWS(mgard::TensorMeshHierarchy<2, float>({17, 0}));
}

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
    std::vector<std::size_t> ndofs(hierarchy.L + 1);
    for (std::size_t l = 0; l <= hierarchy.L; ++l) {
      ndofs.at(l) = hierarchy.ndof(l);
    }
    {
      const std::vector<std::array<std::size_t, 2>> expected = {
          {2, 5}, {3, 9}, {5, 17}, {9, 33}, {12, 39}};
      REQUIRE(hierarchy.shapes == expected);
    }
    {
      const std::vector<std::size_t> expected = {10, 27, 85, 297, 468};
      REQUIRE(ndofs == expected);
    }

    const std::array<std::size_t, 2> &SHAPE = hierarchy.shapes.back();
    TrialTracker tracker;
    for (std::size_t i = 0; i < 2; ++i) {
      const std::vector<double> &xs = hierarchy.coordinates.at(i);
      const std::size_t n = SHAPE.at(i);
      for (std::size_t j = 0; j < n; ++j) {
        tracker += xs.at(j) == Catch::Approx(static_cast<double>(j) / (n - 1));
      }
    }
    REQUIRE(tracker);
  }
  {
    const mgard::TensorMeshHierarchy<3, double> hierarchy({15, 6, 129});
    REQUIRE(hierarchy.L == 3);
    // Note that the final dimension doesn't begin decreasing until every index
    // is of the form `2^k + 1`.
    const std::vector<std::array<std::size_t, 3>> expected = {
        {3, 2, 33}, {5, 3, 65}, {9, 5, 129}, {15, 6, 129}};
    REQUIRE(hierarchy.shapes == expected);
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

namespace {

template <std::size_t N, typename Real>
void test_entry_indexing_manual(
    const std::array<std::size_t, N> shape,
    const std::vector<std::array<std::size_t, N>> &multiindices,
    const std::vector<Real> &expected) {
  const std::size_t ntrials = multiindices.size();
  if (expected.size() != ntrials) {
    throw std::invalid_argument("number of trials inconsistently specified");
  }
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(shape);
  const std::size_t ndof = hierarchy.ndof();
  std::vector<Real> v_unshuffled_(ndof);
  std::vector<Real> v_shuffled_(ndof);
  std::iota(v_unshuffled_.begin(), v_unshuffled_.end(), 0);
  Real *const v = v_shuffled_.data();
  mgard::shuffle(hierarchy, v_unshuffled_.data(), v);
  TrialTracker tracker;
  for (std::size_t i = 0; i < ntrials; ++i) {
    tracker += hierarchy.at(v, multiindices.at(i)) == expected.at(i);
  }
  REQUIRE(tracker);
}

template <std::size_t N>
void test_entry_indexing_exhaustive(const std::array<std::size_t, N> shape) {
  const mgard::TensorMeshHierarchy<N, float> hierarchy(shape);
  const std::size_t ndof = hierarchy.ndof();
  float *const buffer = static_cast<float *>(std::malloc(ndof * sizeof(float)));
  std::iota(buffer, buffer + ndof, 0);
  float *const u = static_cast<float *>(std::malloc(ndof * sizeof(float)));
  mgard::shuffle(hierarchy, buffer, u);
  std::free(buffer);
  int expected = 0;
  TrialTracker tracker;
  for (const mgard::TensorNode<N> node :
       mgard::UnshuffledTensorNodeRange(hierarchy, hierarchy.L)) {
    tracker += hierarchy.at(u, node.multiindex) == static_cast<float>(expected);
    ++expected;
  }
  REQUIRE(tracker);
  std::free(u);
}

} // namespace

TEST_CASE("TensorMeshHierarchy indexing", "[TensorMeshHierarchy]") {
  SECTION("accessing elements") {
    {
      const std::vector<std::array<std::size_t, 1>> multiindices = {
          {0}, {2}, {5}};
      const std::vector<float> expected = {0, 2, 5};
      test_entry_indexing_manual({6}, multiindices, expected);
    }
    {
      const std::vector<std::array<std::size_t, 2>> multiindices = {
          {0, 3}, {10, 1}, {12, 3}};
      const std::vector<double> expected = {3, 41, 51};
      test_entry_indexing_manual({13, 4}, multiindices, expected);
    }
    {
      const std::vector<std::array<std::size_t, 4>> multiindices = {
          {1, 2, 1, 90}, {2, 0, 1, 15}, {2, 2, 1, 27}};
      const std::vector<float> expected = {1190, 1315, 1727};
      test_entry_indexing_manual({3, 3, 2, 100}, multiindices, expected);
    }

    {
      test_entry_indexing_exhaustive<1>({95});
      test_entry_indexing_exhaustive<2>({20, 6});
      test_entry_indexing_exhaustive<3>({7, 11, 12});
    }
  }

  SECTION("accessing indices directly") {
    {
      const mgard::TensorMeshHierarchy<3, float> hierarchy({5, 3, 17});
      REQUIRE(hierarchy.L == 1);
      // Every index is included in the finest level.
      for (std::size_t i = 0; i < 3; ++i) {
        const mgard::TensorIndexRange indices = hierarchy.indices(1, i);
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
          const mgard::TensorIndexRange indices = hierarchy.indices(0, i);
          obtained.at(i).assign(indices.begin(), indices.end());
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
        const mgard::TensorIndexRange indices = hierarchy.indices(l, 1);
        obtained.at(l).assign(indices.begin(), indices.end());
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
        const mgard::TensorIndexRange indices = hierarchy.indices(l, 0);
        obtained.at(l).assign(indices.begin(), indices.end());
      }
      REQUIRE(obtained == expected);
    }
  }

  SECTION("'flat' meshes") {
    test_entry_indexing_exhaustive<2>({1, 35});
    test_entry_indexing_exhaustive<2>({7, 1});
    test_entry_indexing_exhaustive<3>({12, 1, 13});
    test_entry_indexing_exhaustive<3>({9, 1, 1});
    test_entry_indexing_exhaustive<4>({1, 8, 22, 1});
  }
}

namespace {

template <std::size_t N>
void test_index_iteration(
    const std::array<std::size_t, N> shape,
    const std::vector<std::array<std::vector<std::size_t>, N>> &expected) {
  const mgard::TensorMeshHierarchy<N, float> hierarchy(shape);
  std::vector<std::array<std::vector<std::size_t>, N>> encountered(hierarchy.L +
                                                                   1);
  for (std::size_t l = 0; l <= hierarchy.L; ++l) {
    std::array<std::vector<std::size_t>, N> &enc = encountered.at(l);
    for (std::size_t i = 0; i < N; ++i) {
      const mgard::TensorIndexRange indices = hierarchy.indices(l, i);
      enc.at(i).assign(indices.begin(), indices.end());
    }
  }
  REQUIRE(encountered == expected);
}

} // namespace

TEST_CASE("index iteration", "[TensorMeshHierarchy]") {
  {
    const std::array<std::size_t, 2> shape = {9, 5};
    const std::vector<std::array<std::vector<std::size_t>, 2>> expected = {
        {{{{0, 4, 8}, {0, 4}}},
         {{{0, 2, 4, 6, 8}, {0, 2, 4}}},
         {{{0, 1, 2, 3, 4, 5, 6, 7, 8}, {0, 1, 2, 3, 4}}}}};
    test_index_iteration(shape, expected);
  }
  {
    const std::array<std::size_t, 1> shape = {8};
    const std::vector<std::array<std::vector<std::size_t>, 1>> expected = {
        {{{{0, 7}}},
         {{{0, 3, 7}}},
         {{{0, 1, 3, 5, 7}}},
         {{{0, 1, 2, 3, 4, 5, 6, 7}}}}};
    test_index_iteration(shape, expected);
  }
  {
    const mgard::TensorIndexRange indices =
        mgard::TensorIndexRange::singleton();
    const std::vector<std::size_t> obtained(indices.begin(), indices.end());
    const std::vector<std::size_t> expected = {0};
    REQUIRE(obtained == expected);
  }
}

TEST_CASE("node iteration", "[TensorMeshHierarchy]") {
  // The largest of the mesh sizes used below.
  const std::size_t N = 11 * 14;
  float *const buffer = static_cast<float *>(std::malloc(N * sizeof(float)));
  std::iota(buffer, buffer + N, 1);

  {
    const mgard::TensorMeshHierarchy<1, float> hierarchy({17});
    const std::size_t ndof = hierarchy.ndof();
    float *const v = static_cast<float *>(std::malloc(ndof * sizeof(float)));
    mgard::shuffle(hierarchy, buffer, v);
    std::vector<float> encountered_values;
    std::vector<std::size_t> encountered_ls;
    for (const mgard::TensorNode<1> node :
         mgard::UnshuffledTensorNodeRange(hierarchy, 2)) {
      encountered_values.push_back(hierarchy.at(v, node.multiindex));
      encountered_ls.push_back(hierarchy.date_of_birth(node.multiindex));
    }
    const std::vector<float> expected_values = {1, 5, 9, 13, 17};
    const std::vector<std::size_t> expected_ls = {0, 2, 1, 2, 0};
    REQUIRE(encountered_values == expected_values);
    REQUIRE(encountered_ls == expected_ls);
    std::free(v);
  }

  {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({3, 5});
    const std::size_t ndof = hierarchy.ndof();
    float *const v = static_cast<float *>(std::malloc(ndof * sizeof(float)));
    mgard::shuffle(hierarchy, buffer, v);
    std::vector<float> encountered;
    // For the indices.
    TrialTracker tracker;
    for (const mgard::TensorNode<2> node :
         mgard::UnshuffledTensorNodeRange(hierarchy, 0)) {
      encountered.push_back(hierarchy.at(v, node.multiindex));
      tracker += hierarchy.date_of_birth(node.multiindex) == 0;
    }
    const std::vector<float> expected = {1, 3, 5, 11, 13, 15};
    REQUIRE(encountered == expected);
    REQUIRE(tracker);
    std::free(v);
  }

  {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({9, 3});
    const std::size_t ndof = hierarchy.ndof();
    float *const v = static_cast<float *>(std::malloc(ndof * sizeof(float)));
    mgard::shuffle(hierarchy, buffer, v);
    std::vector<float> encountered;
    // For the indices.
    TrialTracker tracker;
    for (const mgard::TensorNode<2> node :
         mgard::UnshuffledTensorNodeRange(hierarchy, 0)) {
      encountered.push_back(hierarchy.at(v, node.multiindex));
      tracker += hierarchy.date_of_birth(node.multiindex) == 0;
    }
    const std::vector<float> expected = {1, 3, 7, 9, 13, 15, 19, 21, 25, 27};
    REQUIRE(encountered == expected);
    REQUIRE(tracker);
    std::free(v);
  }

  {
    const mgard::TensorMeshHierarchy<3, float> hierarchy({3, 3, 3});
    const std::size_t ndof = hierarchy.ndof();
    float *const v = static_cast<float *>(std::malloc(ndof * sizeof(float)));
    mgard::shuffle(hierarchy, buffer, v);
    float expected_value = 1;
    TrialTracker tracker;
    // For the indices.
    std::vector<std::size_t> encountered_ls;
    for (const mgard::TensorNode<3> node :
         mgard::UnshuffledTensorNodeRange(hierarchy, hierarchy.L)) {
      tracker += hierarchy.at(v, node.multiindex) == expected_value;
      expected_value += 1;
      encountered_ls.push_back(hierarchy.date_of_birth(node.multiindex));
    }
    std::vector<std::size_t> expected_ls(27, 1);
    for (const std::size_t index : {0, 2, 6, 8, 18, 20, 24, 26}) {
      expected_ls.at(index) = 0;
    }
    REQUIRE(tracker);
    REQUIRE(encountered_ls == expected_ls);
    std::free(v);
  }

  {
    const mgard::TensorMeshHierarchy<2, float> hierarchy({11, 14});
    const std::size_t ndof = hierarchy.ndof();
    float *const v = static_cast<float *>(std::malloc(ndof * sizeof(float)));
    mgard::shuffle(hierarchy, buffer, v);
    std::vector<std::array<std::size_t, 2>> encountered_multiindices;
    std::vector<std::size_t> encountered_ls;
    for (const mgard::TensorNode<2> node :
         mgard::UnshuffledTensorNodeRange(hierarchy, 2)) {
      encountered_multiindices.push_back(node.multiindex);
      encountered_ls.push_back(hierarchy.date_of_birth(node.multiindex));
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
    std::free(v);
  }

  std::free(buffer);
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
    const mgard::UnshuffledTensorNodeRange<2, double> nodes(hierarchy,
                                                            hierarchy.L);
    // Cheating a little here in predetermining the size.
    std::vector<std::size_t> encountered(hierarchy.ndof());
    std::transform(nodes.begin(), nodes.end(), encountered.begin(),
                   [&](const mgard::TensorNode<2> &node) -> std::size_t {
                     return hierarchy.date_of_birth(node.multiindex);
                   });
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
