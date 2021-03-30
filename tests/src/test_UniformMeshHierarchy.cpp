#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <cmath>
#include <cstddef>

#include <algorithm>
#include <fstream>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "blas.hpp"

#include "moab/Core.hpp"

#include "testing_utilities.hpp"

#include "MeshLevel.hpp"
#include "UniformMeshHierarchy.hpp"
#include "data.hpp"

// These tests call `decompose` and `recompose` in the old style and read the
// transformed coefficients from the same identifier.

TEST_CASE("basic properties", "[UniformMeshHierarchy]") {
  const std::string filename = GENERATE("pyramid.msh", "hexahedron.msh");
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path(filename).c_str());
  require_moab_success(ecode);
  const mgard::MeshLevel mesh(mbcore);
  mgard::UniformMeshHierarchy hierarchy(mesh, 2);
  REQUIRE(hierarchy.L == 2);

  const std::size_t N = hierarchy.ndof();
  std::vector<double> u(N);
  const mgard::MeshLevel &MESH = hierarchy.meshes.back();

  SECTION("zero coordinates with linear function") {
    std::vector<double>::iterator p = u.begin();
    for (const moab::EntityHandle node : MESH.entities[moab::MBVERTEX]) {
      double const *x;
      double const *y;
      double const *z;
      moab::ErrorCode ecode = mbcore.get_coords(node, x, y, z);
      require_moab_success(ecode);
      *p++ = 5 * *x - 3 * *y + 2 * *z;
    }
    hierarchy.decompose(mgard::NodalCoefficients<double>(u.data()));
    TrialTracker tracker;
    for (std::vector<double>::iterator p = u.begin() + hierarchy.ndof(0);
         p != u.end(); ++p) {
      tracker += std::abs(*p) < 1e-6;
    }
    REQUIRE(tracker);
  }

  std::random_device device;
  std::default_random_engine generator(device());
  std::uniform_real_distribution<double> distribution(-1, 1);
  const auto f = [&]() -> double { return distribution(generator); };

  SECTION("recompose inverts decompose") {
    std::generate(u.begin(), u.end(), f);
    std::vector<double> copy = u;
    hierarchy.decompose(mgard::NodalCoefficients<double>(u.data()));
    hierarchy.recompose(mgard::MultilevelCoefficients<double>(u.data()));
    require_vector_equality(u, copy);
  }

  SECTION("recompose inverts decompose") {
    std::generate(u.begin(), u.end(), f);
    std::vector<double> copy = u;
    hierarchy.recompose(mgard::MultilevelCoefficients<double>(u.data()));
    hierarchy.decompose(mgard::NodalCoefficients<double>(u.data()));
    require_vector_equality(u, copy);
  }

  SECTION("multilevel coefficients depend linearly on nodal coefficients") {
    const std::size_t N = u.size();
    std::vector<double> v(N);
    for (std::size_t i = 0; i < N; ++i) {
      u.at(i) = distribution(generator);
      v.at(i) = distribution(generator);
    }
    const double alpha = distribution(generator);
    std::vector<double> w(N);
    //`w = u + alpha * v`.
    blas::copy(N, u.data(), w.data());
    blas::axpy(N, alpha, v.data(), w.data());
    hierarchy.decompose(mgard::NodalCoefficients<double>(u.data()));
    hierarchy.decompose(mgard::NodalCoefficients<double>(v.data()));
    hierarchy.decompose(mgard::NodalCoefficients<double>(w.data()));
    // Copy just overwrite `u` instead.
    std::vector<double> expected(N);
    blas::copy(N, u.data(), expected.data());
    blas::axpy(N, alpha, v.data(), expected.data());
    require_vector_equality(w, expected);
  }
}

TEST_CASE("comparison with Python implementation: refinement and decomposition",
          "[UniformMeshHierarchy]") {
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path("circle.msh").c_str());
  require_moab_success(ecode);
  mgard::MeshLevel mesh(mbcore);
  mgard::UniformMeshHierarchy hierarchy(mesh, 2);

  const std::size_t N = hierarchy.ndof();
  const mgard::MeshLevel &MESH = hierarchy.meshes.back();
  const moab::Range &NODES = MESH.entities[moab::MBVERTEX];
  std::vector<double> coordinates(3 * N);
  ecode = MESH.impl.get_coords(NODES, coordinates.data());
  require_moab_success(ecode);

  std::vector<double> u_nc(N);

  std::set<std::size_t> accounted_for_indices;
  // This will have the ordering of our nodes, not the reference nodes.
  std::vector<double> ref_mc(N);

  double max_rel_xyz_err = 0;
  std::ifstream f;
  f.open(output_path("circle_L=2_coefficients.txt").c_str());
  double xyz[3];
  double nc;
  double mc;
  while (f >> xyz[0] >> xyz[1] >> xyz[2] >> nc >> mc) {
    // Fine because this mesh doesn't have a node at the origin.
    const double xyz_norm = blas::nrm2(3, xyz);
    double min_distance = std::numeric_limits<double>::max();
    std::size_t i;
    double const *_xyz = coordinates.data();
    double difference[3];
    for (std::size_t j = 0; j < N; ++j) {
      blas::copy(3, xyz, difference);
      blas::axpy(3, -1.0, _xyz, difference);
      const double distance = blas::nrm2(3, difference);
      if (distance < min_distance) {
        min_distance = distance;
        i = j;
      }
      _xyz += 3;
    }
    assert(min_distance != std::numeric_limits<double>::max());
    const double rel_xyz_err = min_distance / xyz_norm;
    max_rel_xyz_err = std::max(max_rel_xyz_err, rel_xyz_err);

    accounted_for_indices.insert(i);
    u_nc.at(i) = nc;
    ref_mc.at(i) = mc;
  }
  f.close();
  // Check that the references nodes were uniquely paired with ours.
  REQUIRE(accounted_for_indices.size() == N);
  REQUIRE(max_rel_xyz_err < 1e-6);

  std::vector<double> u_mc = u_nc;
  hierarchy.decompose(mgard::NodalCoefficients<double>(u_mc.data()));

  double max_rel_mc_err = 0;
  for (std::size_t j = 0; j < N; ++j) {
    const double rel_mc_err =
        std::abs((u_mc.at(j) - ref_mc.at(j)) / u_mc.at(j));
    max_rel_mc_err = std::max(max_rel_mc_err, rel_mc_err);
  }
  // Expect some error.
  assert(max_rel_mc_err != 0);
  REQUIRE(max_rel_mc_err < 1e-6);
}

static double square(const double x) { return x * x; }

// Function defined on the mesh.
static double f(const mgard::MeshLevel &mesh, const moab::EntityHandle node) {
  double xyz[3];
  const moab::ErrorCode ecode = mesh.impl.get_coords(&node, 1, xyz);
  require_moab_success(ecode);
  return 4.27 * square(xyz[0]) - 9.28 * square(xyz[1]) + 0.288 * square(xyz[2]);
}

TEST_CASE("iteration over nodes and values", "[UniformMeshHierarchy]") {
  const std::string filename = GENERATE("slope.msh", "hexahedron.msh");
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path(filename).c_str());
  require_moab_success(ecode);
  mgard::MeshLevel mesh_(mbcore);
  const std::size_t L = 3;
  mgard::UniformMeshHierarchy hierarchy(mesh_, L);

  std::vector<double> u_(hierarchy.ndof());
  {
    const mgard::MeshLevel &MESH = hierarchy.meshes.back();
    const moab::Range &NODES = MESH.entities[moab::MBVERTEX];
    std::transform(
        NODES.begin(), NODES.end(), u_.begin(),
        [&](const moab::EntityHandle node) -> double { return f(MESH, node); });
  }
  const mgard::NodalCoefficients<double> u(u_.data());

  for (std::size_t l = 0; l <= L; ++l) {
    const mgard::MeshLevel &mesh = hierarchy.meshes.at(l);

    TrialTracker new_tracker;
    double const *q = hierarchy.on_new_nodes(u, l).begin();
    for (const moab::EntityHandle node : hierarchy.new_nodes(l)) {
      new_tracker += *q++ == f(mesh, node);
    }
    REQUIRE(new_tracker);
    REQUIRE(new_tracker.ntrials == hierarchy.ndof_new(l));

    TrialTracker old_tracker;
    double const *r = hierarchy.on_old_nodes(u, l).begin();
    for (const moab::EntityHandle node : hierarchy.old_nodes(l)) {
      old_tracker += *r++ == f(mesh, node);
    }
    REQUIRE(old_tracker);
    REQUIRE(old_tracker.ntrials == hierarchy.ndof_old(l));
  }
}
