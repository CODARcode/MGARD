#include "catch2/catch.hpp"

#include <cmath>

#include "moab/Core.hpp"
#include "moab/EntityHandle.hpp"
#include "moab/EntityType.hpp"

#include "testing_utilities.hpp"

#include "IndicatorInput.hpp"
#include "MeshLevel.hpp"
#include "UniformMeshHierarchy.hpp"
#include "data.hpp"

// Function defined on the mesh.
static double f(const mgard::MeshLevel &mesh, const moab::EntityHandle node) {
  double xyz[3];
  const moab::ErrorCode ecode = mesh.impl.get_coords(&node, 1, xyz);
  require_moab_success(ecode);
  return (2 * xyz[0] + 1) * (0.5 * xyz[1] - 1) / (std::abs(xyz[2]) + 1);
}

TEST_CASE("IndicatorInput iteration", "[IndicatorInput]") {
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path("triangle.msh").c_str());
  require_moab_success(ecode);
  const mgard::MeshLevel _mesh(mbcore);
  const std::size_t L = 4;
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
  std::vector<std::size_t> counts(L + 1);
  bool all_as_expected = true;
  double const *p = u.data;
  for (const mgard::IndicatorInput input :
       mgard::IndicatorInputRange(hierarchy)) {
    ++counts.at(input.l);
    all_as_expected = (all_as_expected && *p++ == f(input.mesh, input.node));
  }
  REQUIRE(all_as_expected);
  for (std::size_t l = 0; l <= L; ++l) {
    REQUIRE(counts.at(l) == hierarchy.ndof_new(l));
  }
}
