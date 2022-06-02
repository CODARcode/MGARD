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

PeriodicGenerator::PeriodicGenerator(const std::size_t period,
                                     const long int value)
    : period(period), value(value), ncalls(0) {}

long int PeriodicGenerator::operator()() {
  return value + static_cast<long int>(ncalls++ % period);
}
