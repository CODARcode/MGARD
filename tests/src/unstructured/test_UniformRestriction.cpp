#include "catch2/catch_test_macros.hpp"

#include "moab/Core.hpp"

#include "unstructured/MeshLevel.hpp"
#include "unstructured/UniformMeshHierarchy.hpp"
#include "unstructured/UniformRestriction.hpp"

#include "testing_utilities.hpp"

TEST_CASE("uniform functional restriction", "[UniformRestriction]") {
  SECTION("triangles") {
    moab::Core mbcore;
    moab::ErrorCode ecode;
    ecode = mbcore.load_file(mesh_path("lopsided.msh").c_str());
    require_moab_success(ecode);
    mgard::MeshLevel mesh(mbcore);
    mgard::UniformMeshHierarchy hierarchy(mesh, 1);
    const mgard::MeshLevel &MESH = hierarchy.meshes.back();
    mgard::UniformRestriction R(MESH, mesh);
    std::vector<double> f(hierarchy.meshes.front().ndof());
    {
      std::vector<double> F = {4, 6, 10, 12, 1, 5, 8, 7, 9, 11, 2, 3};
      std::vector<double> expected = {11, 18.5, 23, 22, 3.5};
      R(F.data(), f.data());
      require_vector_equality(f, expected);
    }
    {
      std::vector<double> F = {2, 3, 5, 6, 0, -2, 4, -3, -4, -5, 1, -1};
      std::vector<double> expected = {0, 1.5, 3, 1.5, 0};
      R(F.data(), f.data());
      require_vector_equality(f, expected);
    }
  }
  SECTION("tetrahedra") {
    moab::Core mbcore;
    moab::ErrorCode ecode;
    ecode = mbcore.load_file(mesh_path("tetrahedron.msh").c_str());
    require_moab_success(ecode);
    mgard::MeshLevel mesh(mbcore);
    mgard::UniformMeshHierarchy hierarchy(mesh, 1);
    const mgard::MeshLevel &MESH = hierarchy.meshes.back();
    mgard::UniformRestriction R(MESH, mesh);
    std::vector<double> f(hierarchy.meshes.front().ndof());
    {
      std::vector<double> F = {2, 4, 0, -2, 1, -1, 1, -1, 3, -3};
      std::vector<double> expected = {2.5, 5.5, -2.5, -1.5};
      R(F.data(), f.data());
      require_vector_equality(f, expected);
    }
  }
}
