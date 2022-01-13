#include "unstructured/testing_utilities.hpp"

#include <stdexcept>

#include "moab/EntityType.hpp"

#include "testing_paths.hpp"

void require_moab_success(const moab::ErrorCode ecode) {
  if (ecode != moab::MB_SUCCESS) {
    throw std::runtime_error("MOAB error encountered");
  }
}

std::string mesh_path(const std::string &filename) {
  std::string path;
  path += ROOT;
  path += "/";
  path += "meshes";
  path += "/";
  path += filename;
  return path;
}

std::string output_path(const std::string &filename) {
  std::string path;
  path += ROOT;
  path += "/";
  path += "outputs";
  path += "/";
  path += filename;
  return path;
}
