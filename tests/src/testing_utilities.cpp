#include "testing_utilities.hpp"

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
