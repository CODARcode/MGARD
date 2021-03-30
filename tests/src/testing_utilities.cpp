#include "testing_utilities.hpp"

#include <stdexcept>

#include "moab/EntityType.hpp"

#include "testing_paths.hpp"

std::string mesh_path(const std::string &filename) {
  std::string path;
  path += ROOT;
  path += "/";
  path += "tests";
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
  path += "tests";
  path += "/";
  path += "outputs";
  path += "/";
  path += filename;
  return path;
}

void require_moab_success(const moab::ErrorCode ecode) {
  if (ecode != moab::MB_SUCCESS) {
    throw std::runtime_error("MOAB error encountered");
  }
}

TrialTracker::TrialTracker() : nsuccesses(0), nfailures(0), ntrials(0) {}

TrialTracker &TrialTracker::operator+=(const bool result) {
  if (result) {
    ++nsuccesses;
  } else {
    ++nfailures;
  }
  ++ntrials;
  return *this;
}

TrialTracker::operator bool() const {
  return nsuccesses == ntrials && !nfailures;
}

std::ostream &operator<<(std::ostream &os, const TrialTracker &tracker) {
  return os << tracker.nsuccesses << " successes and " << tracker.nfailures
            << " failures out of " << tracker.ntrials << " trials";
}
