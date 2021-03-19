#include "catch2/catch_test_macros.hpp"

#include <array>
#include <set>

#include "moab/Core.hpp"

#include "UniformMeshRefiner.hpp"
#include "utilities.hpp"

#include "testing_utilities.hpp"

typedef std::array<double, 3> NodeCoordinates;
typedef std::set<NodeCoordinates> Element;

static moab::ErrorCode
check_elements(mgard::MeshLevel &MESH,
               const std::set<Element> &expected_elements) {
  moab::ErrorCode ecode;
  std::set<Element> elements;
  for (const moab::EntityHandle t : MESH.entities[MESH.element_type]) {
    moab::EntityHandle const *connectivity;
    int num_nodes;
    ecode = MESH.impl.get_connectivity(t, connectivity, num_nodes);
    MB_CHK_ERR(ecode);
    assert(num_nodes == 3 || num_nodes == 4);
    Element element;
    for (const moab::EntityHandle x :
         mgard::PseudoArray(connectivity, num_nodes)) {
      NodeCoordinates xyz;
      // Could look this up ahead of time.
      ecode = MESH.impl.get_coords(&x, 1, xyz.data());
      MB_CHK_ERR(ecode);
      element.insert(xyz);
    }
    elements.insert(element);
  }
  REQUIRE(elements == expected_elements);
  return moab::MB_SUCCESS;
}

TEST_CASE("refining multiple triangles", "[UniformMeshRefiner]") {
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path("seated.msh").c_str());
  require_moab_success(ecode);
  mgard::MeshLevel mesh(mbcore);

  mgard::UniformMeshRefiner refiner;
  mgard::MeshLevel MESH = refiner(mesh);

  REQUIRE(MESH.topological_dimension == 2);
  REQUIRE(MESH.ndof() == 12);
  REQUIRE(MESH.entities[moab::MBEDGE].size() == 23);
  REQUIRE(MESH.entities[moab::MBTRI].size() == 12);

  ecode =
      check_elements(MESH,
                     {// Elements given by the coordinates of their vertices.
                      {{0, 0, 0}, {4, 0, 0}, {0, 3, 0}},
                      {{4, 0, 0}, {4, 3, 0}, {0, 3, 0}},
                      {{4, 0, 0}, {8, 0, 0}, {4, 3, 0}},
                      {{0, 3, 0}, {4, 3, 0}, {0, 6, 0}},
                      {{0, 0, 0}, {4, 0, 0}, {2, -2, 0}},
                      {{4, 0, 0}, {8, 0, 0}, {6, -2, 0}},
                      {{2, -2, 0}, {4, 0, 0}, {6, -2, 0}},
                      {{2, -2, 0}, {4, -4, 0}, {6, -2, 0}},
                      {{4, -4, 0}, {6, -4, 0}, {6, -2, 0}},
                      {{6, -4, 0}, {6, -2, 0}, {8, -2, 0}},
                      {{6, -4, 0}, {8, -4, 0}, {8, -2, 0}},
                      {{6, -2, 0}, {8, -2, 0}, {8, 0, 0}}});
  require_moab_success(ecode);
}

TEST_CASE("refining multiple tetrahedra", "[UniformMeshRefiner]") {
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path("hexahedron.msh").c_str());
  require_moab_success(ecode);
  mgard::MeshLevel mesh(mbcore);

  mgard::UniformMeshRefiner refiner;
  mgard::MeshLevel MESH = refiner(mesh);

  REQUIRE(MESH.topological_dimension == 3);
  REQUIRE(MESH.ndof() == 14);
  // 44 accounts for repeats on the edges shared by the original tetrahedra, but
  // not for repeats on the (interior of the) face they share. Apparently there
  // are three there.
  REQUIRE(MESH.entities[moab::MBEDGE].size() <= 44);
  REQUIRE(MESH.entities[moab::MBTET].size() == 16);

  ecode = check_elements(
      MESH,
      {// Elements given by the coordinates of their vertices.
       {{0, 0, 0}, {2.5, 0.5, 0.0}, {-0.5, 1.5, 0.0}, {0.0, 0.0, 2.0}},
       {{5, 1, 0}, {2.5, 0.5, 0.0}, {2.0, 2.0, 0.0}, {2.5, 0.5, 2.0}},
       {{-1, 3, 0}, {-0.5, 1.5, 0.0}, {2.0, 2.0, 0.0}, {-0.5, 1.5, 2.0}},
       {{0, 0, 4}, {0.0, 0.0, 2.0}, {2.5, 0.5, 2.0}, {-0.5, 1.5, 2.0}},
       {{-0.5, 1.5, 0.0}, {0.0, 0.0, 2.0}, {-0.5, 1.5, 2.0}, {2.0, 2.0, 0.0}},
       {{0.0, 0.0, 2.0}, {-0.5, 1.5, 2.0}, {2.5, 0.5, 2.0}, {2.0, 2.0, 0.0}},
       {{-0.5, 1.5, 0.0}, {2.5, 0.5, 0.0}, {0.0, 0.0, 2.0}, {2.0, 2.0, 0.0}},
       {{2.5, 0.5, 0.0}, {0.0, 0.0, 2.0}, {2.0, 2.0, 0.0}, {2.5, 0.5, 2.0}},
       {{-1, 3, 0}, {2.0, 2.0, 0.0}, {-0.5, 1.5, 2.0}, {1.0, 2.5, 1.5}},
       {{5, 1, 0}, {2.0, 2.0, 0.0}, {2.5, 0.5, 2.0}, {4.0, 1.5, 1.5}},
       {{0, 0, 4}, {-0.5, 1.5, 2.0}, {2.5, 0.5, 2.0}, {1.5, 1.0, 3.5}},
       {{3, 2, 3}, {1.0, 2.5, 1.5}, {4.0, 1.5, 1.5}, {1.5, 1.0, 3.5}},
       {{-0.5, 1.5, 2.0}, {1.0, 2.5, 1.5}, {1.5, 1.0, 3.5}, {2.5, 0.5, 2.0}},
       {{1.0, 2.5, 1.5}, {1.5, 1.0, 3.5}, {4.0, 1.5, 1.5}, {2.5, 0.5, 2.0}},
       {{-0.5, 1.5, 2.0}, {2.0, 2.0, 0.0}, {1.0, 2.5, 1.5}, {2.5, 0.5, 2.0}},
       {{2.0, 2.0, 0.0}, {1.0, 2.5, 1.5}, {2.5, 0.5, 2.0}, {4.0, 1.5, 1.5}}});
  require_moab_success(ecode);
}

TEST_CASE("refining triangle multiply", "[UniformMeshRefiner]") {
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path("triangle.msh").c_str());
  require_moab_success(ecode);
  mgard::MeshLevel mesh(mbcore);

  mgard::UniformMeshRefiner refiner;
  mgard::MeshLevel MESH = refiner(refiner(mesh));

  REQUIRE(MESH.ndof() == 15);
  REQUIRE(MESH.entities[moab::MBEDGE].size() == 30);
  REQUIRE(MESH.entities[moab::MBTRI].size() == 16);

  double z;
  // All the nodes have the same `z` coordinate.
  {
    double const *_x;
    double const *_y;
    double const *_z;
    const moab::EntityHandle node = mesh.entities[moab::MBVERTEX].front();
    ecode = mbcore.get_coords(node, _x, _y, _z);
    require_moab_success(ecode);
    z = *_z;
  }
  ecode = check_elements(MESH, {
                                   {{0, 0, z}, {2.5, 0, z}, {2.5, 1.5, z}},
                                   {{2.5, 0, z}, {5, 0, z}, {5, 1.5, z}},
                                   {{5, 0, z}, {7.5, 0, z}, {7.5, 1.5, z}},
                                   {{7.5, 0, z}, {10, 0, z}, {10, 1.5, z}},
                                   {{2.5, 0, z}, {2.5, 1.5, z}, {5, 1.5, z}},
                                   {{5, 0, z}, {5, 1.5, z}, {7.5, 1.5, z}},
                                   {{7.5, 0, z}, {7.5, 1.5, z}, {10, 1.5, z}},
                                   {{2.5, 1.5, z}, {5, 1.5, z}, {5, 3, z}},
                                   {{5, 1.5, z}, {7.5, 1.5, z}, {7.5, 3, z}},
                                   {{7.5, 1.5, z}, {10, 1.5, z}, {10, 3, z}},
                                   {{5, 1.5, z}, {5, 3, z}, {7.5, 3, z}},
                                   {{7.5, 1.5, z}, {7.5, 3, z}, {10, 3, z}},
                                   {{5, 3, z}, {7.5, 3, z}, {7.5, 4.5, z}},
                                   {{7.5, 3, z}, {10, 3, z}, {10, 4.5, z}},
                                   {{7.5, 3, z}, {7.5, 4.5, z}, {10, 4.5, z}},
                                   {{7.5, 4.5, z}, {10, 4.5, z}, {10, 6, z}},
                               });
  require_moab_success(ecode);
}

TEST_CASE("refining tetrahedron multiply", "[UniformMeshRefiner]") {
  moab::ErrorCode ecode;
  moab::Core mbcore;
  ecode = mbcore.load_file(mesh_path("tetrahedron.msh").c_str());
  require_moab_success(ecode);
  mgard::MeshLevel mesh(mbcore);

  mgard::UniformMeshRefiner refiner;
  // Mostly just checking that we don't hit any errors here.
  mgard::MeshLevel MESH = refiner(refiner(refiner(mesh)));
  REQUIRE(MESH.entities[moab::MBTET].size() == 512);
}
