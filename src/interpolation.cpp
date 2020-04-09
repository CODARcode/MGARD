#include "interpolation.hpp"
#include "interpolation.tpp"

namespace mgard {

template float interpolate<float>(const float q0, const float q1,
                                  const float x0, const float x1,
                                  const float x);

template double interpolate<double>(const double q0, const double q1,
                                    const double x0, const double x1,
                                    const double x);

template float interpolate<float>(const float q00, const float q01,
                                  const float q10, const float q11,
                                  const float x0, const float x1,
                                  const float y0, const float y1, const float x,
                                  const float y);

template double interpolate<double>(const double q00, const double q01,
                                    const double q10, const double q11,
                                    const double x0, const double x1,
                                    const double y0, const double y1,
                                    const double x, const double y);

template float interpolate<float>(const float q000, const float q001,
                                  const float q010, const float q011,
                                  const float q100, const float q101,
                                  const float q110, const float q111,
                                  const float x0, const float x1,
                                  const float y0, const float y1,
                                  const float z0, const float z1, const float x,
                                  const float y, const float z);

template double
interpolate<double>(const double q000, const double q001, const double q010,
                    const double q011, const double q100, const double q101,
                    const double q110, const double q111, const double x0,
                    const double x1, const double y0, const double y1,
                    const double z0, const double z1, const double x,
                    const double y, const double z);

} // namespace mgard
