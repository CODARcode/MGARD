#include "catch2/catch.hpp"

#include <string>

#include "moab/Core.hpp"

#include "testing_utilities.hpp"

#include "EnumeratedMeshRange.hpp"
#include "MeshLevel.hpp"
#include "UniformMeshHierarchy.hpp"

TEST_CASE("EnumeratedMeshRange iteration", "[EnumeratedMeshRange]") {
  const auto [filename, expected_ndofs] =
      GENERATE(table<std::string, std::vector<std::size_t>>(
          {{"seated.msh", {5, 12, 35}}, {"tetrahedron.msh", {4, 10, 35}}}));
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path(filename).c_str());
  require_moab_success(ecode);
  const mgard::MeshLevel _mesh(mbcore);
  mgard::UniformMeshHierarchy hierarchy(_mesh, 2);

  const mgard::EnumeratedMeshRange emr(hierarchy);
  mgard::EnumeratedMeshRange::iterator p = emr.begin();
  REQUIRE(p == emr.begin());
  std::vector<std::size_t>::const_iterator q = expected_ndofs.begin();

  {
    const auto &[l, mesh] = *p;
    REQUIRE(l == 0);
    REQUIRE(mesh.ndof() == *q++);
  }
  {
    const auto &[l, mesh] = *++p;
    REQUIRE(l == 1);
    REQUIRE(mesh.ndof() == *q++);
  }
  {
    const auto &[l, mesh] = *++p;
    REQUIRE(l == 2);
    REQUIRE(mesh.ndof() == *q++);
  }

  REQUIRE(++p == emr.end());
}
