#include <cstddef>

#include <experimental/filesystem>
#include <string>

#include "moab/Interface.hpp"

#include "MeshLevel.hpp"

std::experimental::filesystem::path mesh_path(const std::string &filename);

std::experimental::filesystem::path output_path(const std::string &filename);

void require_moab_success(const moab::ErrorCode ecode);
