#include "catch2/catch.hpp"

#include "moab/Core.hpp"

#include "MeshLevel.hpp"
#include "UniformRestriction.hpp"
#include "UniformMeshHierarchy.hpp"

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
            bool all_close = true;
            for (std::size_t i = 0; i < f.size(); ++i) {
                all_close = all_close && f.at(i) == Approx(expected.at(i));
            }
            REQUIRE(all_close);
        }
        {
            std::vector<double> F = {2, 3, 5, 6, 0, -2, 4, -3, -4, -5, 1, -1};
            std::vector<double> expected = {0, 1.5, 3, 1.5, 0};
            R(F.data(), f.data());
            bool all_close = true;
            for (std::size_t i = 0; i < f.size(); ++i) {
                all_close = all_close && f.at(i) == Approx(expected.at(i));
            }
            REQUIRE(all_close);
        }
    }
}
