#include "catch2/catch.hpp"

#include "moab/Core.hpp"

#include "testing_utilities.hpp"

#include "EnumeratedMeshRange.hpp"
#include "MeshLevel.hpp"
#include "UniformMeshHierarchy.hpp"

TEST_CASE("EnumeratedMeshRange iteration", "[EnumeratedMeshRange]") {
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path("seated.msh").c_str());
  require_moab_success(ecode);
  const mgard::MeshLevel _mesh(mbcore);
  mgard::UniformMeshHierarchy hierarchy(_mesh, 2);

  const mgard::EnumeratedMeshRange emr(hierarchy);
  mgard::EnumeratedMeshRange::iterator p = emr.begin();
  REQUIRE(p == emr.begin());

  {
    const auto &[l, mesh] = *p;
    REQUIRE(l == 0);
    REQUIRE(mesh.ndof() == 5);
  }
  {
    const auto &[l, mesh] = *++p;
    REQUIRE(l == 1);
    REQUIRE(mesh.ndof() == 12);
  }
  {
    const auto &[l, mesh] = *++p;
    REQUIRE(l == 2);
    REQUIRE(mesh.ndof() == 35);
  }

  REQUIRE(++p == emr.end());
}
