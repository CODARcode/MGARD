#include "catch2/catch.hpp"

#include <cstddef>

#include <vector>

#include "testing_utilities.hpp"

#include "mgard_mesh.hpp"

TEST_CASE("helper functions", "[mgard_mesh]") {
  SECTION("mesh size functions") {
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
      const std::array<std::size_t, 1> expected = {5};
      REQUIRE(dims.rnded == expected);
    }
  }
  {
    const mgard::Dimensions2kPlus1<2> dims({11, 32});
    REQUIRE(!dims.is_2kplus1());
    REQUIRE(dims.nlevel == 3);
    {
      const std::array<std::size_t, 2> expected = {9, 17};
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
