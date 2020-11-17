#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <cmath>

#include <string>

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
  const std::string filename = GENERATE("pyramid.msh", "hexahedron.msh");
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path(filename).c_str());
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
    const mgard::MeshLevel &mesh = hierarchy.meshes.at(l);
    const mgard::SituatedCoefficientRange scr(hierarchy, u, l);
    TrialTracker tracker;
    for (auto [node, coefficient] : scr) {
      tracker += coefficient == f(mesh, node);
    }
    REQUIRE(tracker);
    REQUIRE(tracker.ntrials == hierarchy.ndof_new(l));
  }
}
