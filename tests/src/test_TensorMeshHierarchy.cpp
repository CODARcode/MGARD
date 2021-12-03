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

#ifdef MGARD_PROTOBUF
#include "proto/mgard.pb.h"
#endif

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
    REQUIRE(hierarchy.uniform);
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
    REQUIRE(hierarchy.uniform);
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
    REQUIRE(hierarchy.uniform);
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
    REQUIRE(not hierarchy.uniform);
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

//! Count the nodes in a given mesh level preceding a given node.
//!
//! If the node is contained in the mesh level, the count is equal to the
//! position of the node in the 'physical' ordering of the nodes (`{0, 0, 0}`,
//! `{0, 0, 1}`, and so on).
//!
//!\param hierarchy Mesh hierarchy.
//!\param l Index of the mesh level whose nodes are to be counted.
//!\param multiindex Multiindex of the node.
template <std::size_t N, typename Real>
std::size_t
number_nodes_before(const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
                    const std::size_t l,
                    const std::array<std::size_t, N> multiindex) {
  const std::array<std::size_t, N> &SHAPE = hierarchy.shapes.back();
  const std::array<std::size_t, N> &shape = hierarchy.shapes.at(l);
  // Let `α` be the given node (its multiindex). A node (multiindex) `β` comes
  // before `α` if
  //     * `β_{1} < α_{1}` xor
  //     * `β_{1} = α_{1}` and `β_{2} < α_{2}` xor
  //     * …
  //     * `β_{1} = α_{1}`, …, `β_{N - 1} = α_{N - 1}` and `β_{N} < α_{N}`.
  // Consider one of these options: `β_{1} = α_{1}`, …, `β_{i - 1} = α_{i - 1}`
  // and `β_{i} < α_{i}`. Let `M_{k}` and `m_{k}` be the sizes of the finest and
  // `l`th meshes, respectively, in dimension `k`. `β` is unconstrained in
  // dimensions `i + 1` through `N`, so we start with a factor of `m_{i + 1} × ⋯
  // × m_{N}`. The values of `β_{1}`, …, `β_{i - 1}` are prescribed. `β_{i}` is
  // of the form `floor((j * (M_{i} - 1)) / (m_{i} - 1))`. We want `β_{i} <
  // α_{i}`, so `j` will go from zero up to some maximal value, after which
  // `β_{i} ≥ α_{i}`. The count of possible `j` values is the least `j` such
  // that `β_{i} ≥ α_{i}`. A bit of algebra shows that this is
  // `ceil((α_{i} * (m_{i} - 1)) / (M_{i} - 1))`. So, the count of possible
  // `β`s for this option is (assuming the constraints on `β_{1}`, …,
  // `β_{i - 1}` can be met – see below)
  // ```
  //   m_{i + 1} × ⋯ × m_{N} × ceil((α_{i} * (m_{i} - 1)) / (M_{i} - 1)).
  // ```
  // We compute the sum of these counts in the loop below, rearranging so that
  // we only have to multiply by each `m_{k}` once.
  //
  // One detail I missed: if `α` was introduced *after* the `l`th mesh, then it
  // may not be possible for `β_{k}` to equal `α_{k}`, since `β` must be present
  // in the `l`th mesh. Any option involving one of these 'impossible
  // constraints' will be knocked out and contribute nothing to the sum.
  //
  // That above assumes that `M_{i} ≠ 1`. In that case, it is impossible for
  // `β_{i}` to be less than `α_{i}` (both must be zero), so instead of
  // `ceil((α_{i} * (m_{i} - 1)) / (M_{i} - 1))` we get a factor of zero.
  std::size_t count = 0;
  bool impossible_constraint_encountered = false;
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t m = shape.at(i);
    const std::size_t M = SHAPE.at(i);
    // Notice that this has no effect in the first iteration.
    count *= m;
    if (impossible_constraint_encountered) {
      continue;
    }
    const std::size_t index = multiindex.at(i);
    const std::size_t numerator = index * (m - 1);
    const std::size_t denominator = M - 1;
    // We want to add `ceil(numerator / denominator)`. We can compute this term
    // using only integer divisions by adding one less than the denominator to
    // the numerator.
    // If the mesh is flat in this dimension (if `M == 1`), then `β_{i}` cannot
    // be less than `α_{i}` and so this case contributes nothing to the count.
    count += denominator ? (numerator + (denominator - 1)) / denominator : 0;
    // The 'impossible constraint' will be encountered in the next iteration,
    // when we stipulate that `β_{i} = α_{i}` (current value of `i`).
    impossible_constraint_encountered =
        impossible_constraint_encountered ||
        hierarchy.dates_of_birth.at(i).at(index) > l;
  }
  return count;
}

//! Compute the index of a node in the 'shuffled' ordering.
//!
//!\param hierarchy Mesh hierarchy.
//!\param multiindex Multiindex of the node.
template <std::size_t N, typename Real>
std::size_t index(const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
                  const std::array<std::size_t, N> multiindex) {
  const std::size_t l = hierarchy.date_of_birth(multiindex);
  if (!l) {
    return number_nodes_before(hierarchy, l, multiindex);
  }
  return hierarchy.ndof(l - 1) + number_nodes_before(hierarchy, l, multiindex) -
         number_nodes_before(hierarchy, l - 1, multiindex);
}

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
    const std::array<std::size_t, N> &multiindex = multiindices.at(i);
    const Real expected_ = expected.at(i);
    tracker += hierarchy.at(v, multiindex) == expected_;
    // Compare with original `TensorMeshHierarchy::index` implementation.
    tracker += v[index(hierarchy, multiindex)] == expected_;
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
    const std::array<std::size_t, N> &multiindex = node.multiindex;
    const float expected_ = static_cast<float>(expected);
    tracker += hierarchy.at(u, multiindex) == expected_;
    // Compare with original `TensorMeshHierarchy::index` implementation.
    tracker += u[index(hierarchy, multiindex)] == expected_;
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
    const std::size_t singleton = 0;
    const mgard::TensorIndexRange indices = {.begin_ = &singleton,
                                             .end_ = &singleton + 1};
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

#ifdef MGARD_PROTOBUF
namespace {

template <std::size_t N>
void check_cartesian_topology(const mgard::pb::Domain &domain,
                              const std::array<std::size_t, N> &shape) {
  REQUIRE(domain.topology() == mgard::pb::Domain::CARTESIAN_GRID);
  REQUIRE(domain.topology_definition_case() ==
          mgard::pb::Domain::kCartesianGridTopology);
  const mgard::pb::CartesianGridTopology &cgt =
      domain.cartesian_grid_topology();
  REQUIRE(cgt.dimension() == N);
  const google::protobuf::RepeatedField<google::protobuf::uint64> &shape_ =
      cgt.shape();
  REQUIRE(shape_.size() == N);
  REQUIRE(std::equal(shape_.begin(), shape_.end(), shape.begin()));
}

void check_decomposition_hierarchy(const mgard::pb::Header &header) {
  REQUIRE(header.decomposition().hierarchy() ==
          mgard::pb::Decomposition::POWER_OF_TWO_PLUS_ONE);
}

} // namespace

TEST_CASE("header field population", "[TensorMeshHierarchy]") {
  {
    mgard::pb::Header header;
    const std::array<std::size_t, 1> shape{123};
    const mgard::TensorMeshHierarchy<1, float> hierarchy(shape);
    hierarchy.populate(header);

    const mgard::pb::Domain &domain = header.domain();
    check_cartesian_topology(domain, shape);

    REQUIRE(domain.geometry() == mgard::pb::Domain::UNIT_CUBE);
    REQUIRE(domain.geometry_definition_case() ==
            mgard::pb::Domain::GEOMETRY_DEFINITION_NOT_SET);

    const mgard::pb::Dataset &dataset = header.dataset();
    REQUIRE(dataset.type() == mgard::pb::Dataset::FLOAT);
    REQUIRE(dataset.dimension() == 1);

    check_decomposition_hierarchy(header);
  }
  {
    mgard::pb::Header header;
    const std::array<std::size_t, 3> shape{5, 5, 4};
    std::array<std::vector<double>, 3> coordinates;
    coordinates.at(0) = {0, 0.2, 0.3, 0.5, 0.6};
    coordinates.at(1) = {20.5, 21.5, 22.5, 23, 23.5};
    coordinates.at(2) = {0, 1, 2, 3};
    // Expected flattened (as a single array) coordinates.
    std::vector<double> efc;
    efc.resize(std::accumulate(shape.begin(), shape.end(), 0));
    {
      std::vector<double>::iterator p = efc.begin();
      for (const std::vector<double> &xs : coordinates) {
        std::copy(xs.begin(), xs.end(), p);
        p += xs.size();
      }
    }
    const mgard::TensorMeshHierarchy<3, double> hierarchy(shape, coordinates);
    hierarchy.populate(header);

    const mgard::pb::Domain &domain = header.domain();
    check_cartesian_topology(domain, shape);

    REQUIRE(domain.geometry() == mgard::pb::Domain::EXPLICIT_CUBE);
    REQUIRE(domain.geometry_definition_case() ==
            mgard::pb::Domain::kExplicitCubeGeometry);
    const mgard::pb::ExplicitCubeGeometry &ecg =
        domain.explicit_cube_geometry();
    const google::protobuf::RepeatedField<double> &coordinates_ =
        ecg.coordinates();
    REQUIRE(std::equal(coordinates_.begin(), coordinates_.end(), efc.begin()));

    const mgard::pb::Dataset &dataset = header.dataset();
    REQUIRE(dataset.type() == mgard::pb::Dataset::DOUBLE);
    REQUIRE(dataset.dimension() == 1);

    check_decomposition_hierarchy(header);
  }
}
#endif
