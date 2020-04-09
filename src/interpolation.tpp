namespace mgard {

template <typename Real>
Real interpolate(const Real q0, const Real q1, const Real x0, const Real x1,
                 const Real x) {
  return ((x1 - x) * q0 + (x - x0) * q1) / (x1 - x0);
}

template <typename Real>
Real interpolate(const Real q00, const Real q01, const Real q10, const Real q11,
                 const Real x0, const Real x1, const Real y0, const Real y1,
                 const Real x, const Real y) {
  return interpolate(interpolate(q00, q01, y0, y1, y),
                     interpolate(q10, q11, y0, y1, y), x0, x1, x);
}

template <typename Real>
Real interpolate(const Real q000, const Real q001, const Real q010,
                 const Real q011, const Real q100, const Real q101,
                 const Real q110, const Real q111, const Real x0, const Real x1,
                 const Real y0, const Real y1, const Real z0, const Real z1,
                 const Real x, const Real y, const Real z) {
  return interpolate(interpolate(q000, q001, q010, q011, y0, y1, z0, z1, y, z),
                     interpolate(q100, q101, q110, q111, y0, y1, z0, z1, y, z),
                     x0, x1, x);
}

} // namespace mgard
