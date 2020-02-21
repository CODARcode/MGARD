#include "catch2/catch.hpp"

#include <cmath>

#include "moab/Core.hpp"
#include "moab/EntityHandle.hpp"
#include "moab/EntityType.hpp"

#include "testing_utilities.hpp"

#include "MeshLevel.hpp"
#include "SituatedCoefficientRange.hpp"
#include "UniformMeshHierarchy.hpp"
#include "data.hpp"

// Function defined on the mesh.
static double f(const mgard::MeshLevel &mesh, const moab::EntityHandle node) {
  double xyz[3];
  const moab::ErrorCode ecode = mesh.impl.get_coords(&node, 1, xyz);
  require_moab_success(ecode);
  return (std::sin(2 + 0.5 * xyz[0] - 0.4 * xyz[1]) -
          2 * std::cos(xyz[1] - 4 * xyz[2]));
}

TEST_CASE("SituatedCoefficientRange iteration", "[SituatedCoefficientRange]") {
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path("pyramid.msh").c_str());
  require_moab_success(ecode);
  const mgard::MeshLevel _mesh(mbcore);
  const std::size_t L = 2;
  mgard::UniformMeshHierarchy hierarchy(_mesh, L);

  std::vector<double> u_(hierarchy.ndof());
  {
    std::vector<double>::iterator p = u_.begin();
    const mgard::MeshLevel &MESH = hierarchy.meshes.back();
    const moab::Range &NODES = MESH.entities[moab::MBVERTEX];
    for (const moab::EntityHandle node : NODES) {
      *p++ = f(MESH, node);
    }
  }
  const mgard::MultilevelCoefficients<double> u(u_.data());
  for (std::size_t l = 0; l <= L; ++l) {
    std::size_t count = 0;
    const mgard::MeshLevel &mesh = hierarchy.meshes.at(l);
    const mgard::SituatedCoefficientRange scr(hierarchy, u, l);
    bool all_as_expected = true;
    for (auto [node, coefficient] : scr) {
      all_as_expected = all_as_expected && coefficient == f(mesh, node);
      ++count;
    }
    REQUIRE(all_as_expected);
    REQUIRE(count == hierarchy.ndof_new(l));
  }
}
