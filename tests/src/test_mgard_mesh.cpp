#include "catch2/catch.hpp"

#include <cstddef>

#include <vector>

#include "testing_utilities.hpp"

#include "mgard_mesh.hpp"

TEST_CASE("helper functions", "[mgard_mesh]") {
  SECTION("indexing functions") {
    REQUIRE(mgard::get_index(12, 0, 0) == 0);
    REQUIRE(mgard::get_index(61, 0, 2) == 2);
    REQUIRE(mgard::get_index(5, 1, 1) == 6);

    REQUIRE(mgard::get_index3(4, 17, 3, 2, 10) == 248);
    TrialTracker tracker;
    {
      const int nrow = 6;
      const int ncol = 2;
      const int nfib = 3;
      int index = 0;
      for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
          for (int k = 0; k < nfib; ++k) {
            tracker += mgard::get_index3(ncol, nfib, i, j, k) == index++;
          }
        }
      }
    }
    REQUIRE(tracker);
    REQUIRE(tracker.ntrials == 36);

    // TODO: Add tests for `get_lindex` once its purpose is better understood.
  }

  SECTION("stride and mesh size functions") {
    REQUIRE(mgard::stride_from_index_difference(0) == 1);
    REQUIRE(mgard::stride_from_index_difference(1) == 2);
    REQUIRE(mgard::stride_from_index_difference(3) == 8);

    REQUIRE(mgard::nlevel_from_size(2) == 0);
    REQUIRE(mgard::nlevel_from_size(3) == 1);
    REQUIRE(mgard::nlevel_from_size(32) == 4);
    REQUIRE(mgard::nlevel_from_size(35) == 5);
    REQUIRE(mgard::nlevel_from_size(4100) == 12);

    REQUIRE(mgard::size_from_nlevel(0) == 2);
    REQUIRE(mgard::size_from_nlevel(4) == 17);
    REQUIRE(mgard::size_from_nlevel(10) == 1025);
  }
}

TEST_CASE("`Dimensions2kPlus1`", "[mgard_mesh]") {
  {
    const mgard::Dimensions2kPlus1<1> dims({5});
    REQUIRE(dims.is_2kplus1());
    REQUIRE(dims.nlevel == 2);
    {
      const std::array<int, 1> expected = {5};
      REQUIRE(dims.rnded == expected);
    }
  }
  {
    const mgard::Dimensions2kPlus1<2> dims({11, 32});
    REQUIRE(!dims.is_2kplus1());
    REQUIRE(dims.nlevel == 3);
    {
      const std::array<int, 2> expected = {9, 17};
      REQUIRE(dims.rnded == expected);
    }
  }
  {
    const mgard::Dimensions2kPlus1<5> dims({1, 257, 129, 129, 1});
    REQUIRE(dims.is_2kplus1());
    REQUIRE(dims.nlevel == 7);
    REQUIRE(dims.rnded == dims.input);
  }
  {
    const mgard::Dimensions2kPlus1<2> dims({6, 5});
    const std::array<std::size_t, 2> expected = {5, 5};
    REQUIRE(dims.rnded == expected);
  }

  REQUIRE_THROWS(mgard::Dimensions2kPlus1<3>({1, 1, 1}));
  REQUIRE_THROWS(mgard::Dimensions2kPlus1<2>({17, 0}));
}

TEST_CASE("level values iteration", "[mgard_mesh]") {
  const std::size_t N = 27;
  std::vector<float> v(N);
  std::iota(v.begin(), v.end(), 1);

  {
    const mgard::Dimensions2kPlus1<1> dims({17});
    std::vector<float> encountered;
    for (const float value : dims.on_nodes(v.data(), 2)) {
      encountered.push_back(value);
    }
    const std::vector<float> expected = {1, 5, 9, 13, 17};
    REQUIRE(encountered == expected);
  }

  {
    const mgard::Dimensions2kPlus1<2> dims({3, 5});
    std::vector<float> encountered;
    for (const float value : dims.on_nodes(v.data(), 1)) {
      encountered.push_back(value);
    }
    const std::vector<float> expected = {1, 3, 5, 11, 13, 15};
    REQUIRE(encountered == expected);
  }

  {
    const mgard::Dimensions2kPlus1<2> dims({9, 3});
    std::vector<float> encountered;
    for (const float value : dims.on_nodes(v.data(), 1)) {
      encountered.push_back(value);
    }
    const std::vector<float> expected = {1, 3, 7, 9, 13, 15, 19, 21, 25, 27};
    REQUIRE(encountered == expected);
  }

  {
    const mgard::Dimensions2kPlus1<3> dims({3, 3, 3});
    float expected = 1;
    TrialTracker tracker;
    for (const float value : dims.on_nodes(v.data(), 0)) {
      tracker += value == expected;
      expected += 1;
    }
    REQUIRE(tracker);
  }
}
