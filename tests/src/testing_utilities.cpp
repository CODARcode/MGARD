#include "testing_utilities.hpp"

#include <cassert>

#include <stdexcept>

#include "moab/EntityType.hpp"

std::filesystem::path mesh_path(const std::string &filename) {
    return std::filesystem::path("tests") / "meshes" / filename;
}

void require_moab_success(const moab::ErrorCode ecode) {
    if (ecode != moab::MB_SUCCESS) {
        throw std::runtime_error("MOAB error encountered");
    }
}
