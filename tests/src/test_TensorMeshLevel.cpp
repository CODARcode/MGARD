#include "catch2/catch.hpp"

#include "TensorMeshLevel.hpp"

TEST_CASE("TensorMeshLevel size and equality", "[TensorMeshLevel]") {
  {
    const mgard::TensorMeshLevel<1, float> a({5});
    const mgard::TensorMeshLevel<1, float> b({7});
    const mgard::TensorMeshLevel<1, float> c({5});

    REQUIRE(a.ndof() == 5);
    REQUIRE(b.ndof() == 7);

    REQUIRE(a != b);
    REQUIRE(a == c);
    REQUIRE(b != c);
  }
  {
    const mgard::TensorMeshLevel<2, double> a({10, 10});
    const mgard::TensorMeshLevel<2, double> b({5, 20});

    REQUIRE(a.ndof() == 100);
    REQUIRE(b.ndof() == 100);
    REQUIRE(a != b);
  }
  {
    const mgard::TensorMeshLevel<3, float> a({5, 9, 5});
    const mgard::TensorMeshLevel<3, float> b({9, 5, 5});

    REQUIRE(a.ndof() == 225);
    REQUIRE(b.ndof() == 225);
    REQUIRE(a != b);
  }
}
