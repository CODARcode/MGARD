namespace cli {

template <typename Real>
SmoothnessParameter<Real>::SmoothnessParameter(const Real s) : s(s) {}

template <typename Real> SmoothnessParameter<Real>::operator Real() const {
  return s;
}

} // namespace cli
