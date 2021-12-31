#include "catch2/catch_test_macros.hpp"

#include <cstddef>

#include <algorithm>

#include "proto/mgard.pb.h"

#include "TensorMeshHierarchy.hpp"
#include "TensorMultilevelCoefficientQuantizer.hpp"
#include "format.hpp"
#include "quantize.hpp"

TEST_CASE("quantization", "[quantize]") {
  const mgard::TensorMeshHierarchy<2, float> hierarchy({9, 10});
  const std::size_t ndof = hierarchy.ndof();
  const float s = 0;
  const float tolerance = 0.01;
  float *const coefficients = new float[ndof];
  for (std::size_t i = 0; i < ndof; ++i) {
    coefficients[i] = 0.25f / (i + 1);
  }

  mgard::pb::Header header;
  mgard::populate_defaults(header);
  {
    mgard::pb::Quantization &q = *header.mutable_quantization();
    q.set_type(mgard::pb::Quantization::INT8_T);
  }

  std::int8_t *const quantized = new std::int8_t[ndof];
  mgard::quantize(hierarchy, s, tolerance, coefficients, quantized, header);

  using Qntzr =
      mgard::TensorMultilevelCoefficientQuantizer<2, float, std::int8_t>;
  const Qntzr quantizer(hierarchy, s, tolerance);
  using It = typename Qntzr::iterator;
  const mgard::RangeSlice<It> quantized_ = quantizer(coefficients);

  REQUIRE(std::equal(quantized, quantized + ndof, quantized_.begin()));
  delete[] quantized;
  delete[] coefficients;
}

TEST_CASE("dequantization", "[quantize]") {
  const mgard::TensorMeshHierarchy<1, double> hierarchy({148});
  const std::size_t ndof = hierarchy.ndof();
  const double s = 0.5;
  const double tolerance = 0.01;
  void const *const data = nullptr;
  const std::size_t size = 0;
  mgard::pb::Header header;
  mgard::populate_defaults(header);
  {
    mgard::pb::Quantization &q = *header.mutable_quantization();
    q.set_type(mgard::pb::Quantization::INT16_T);
  }
  const mgard::CompressedDataset<1, double> compressed(hierarchy, s, tolerance,
                                                       data, size, header);

  std::int16_t *const quantized = new std::int16_t[ndof];
  for (std::size_t i = 0; i < ndof; ++i) {
    const std::int16_t j = static_cast<std::int16_t>(i);
    quantized[i] = (2 * (j % 2) - 1) * (j / 2);
  }
  double *const dequantized = new double[ndof];
  mgard::dequantize(compressed, quantized, dequantized);

  using Dqntzr =
      mgard::TensorMultilevelCoefficientDequantizer<1, std::int16_t, double>;
  const Dqntzr dequantizer(compressed.hierarchy, compressed.s,
                           compressed.tolerance);
  using It = typename Dqntzr::template iterator<std::int16_t *>;
  const mgard::RangeSlice<It> dequantized_ =
      dequantizer(quantized, quantized + ndof);

  REQUIRE(std::equal(dequantized, dequantized + ndof, dequantized_.begin()));
  delete[] dequantized;
  delete[] quantized;
}

TEST_CASE("alignment and endianness", "[quantize]") {
  const mgard::TensorMeshHierarchy<3, float> hierarchy({5, 12, 13});
  const std::size_t ndof = hierarchy.ndof();
  const float s = 0;
  const float tolerance = 1;
  float const *const coefficients = nullptr;
  mgard::pb::Header header;
  mgard::populate_defaults(header);
  {
    mgard::pb::Quantization &q = *header.mutable_quantization();
    q.set_type(mgard::pb::Quantization::INT32_T);
  }

  void const *const data = nullptr;
  const std::size_t size = 0;
  mgard::CompressedDataset<3, float> compressed(hierarchy, s, tolerance, data,
                                                size, header);
  float *const dequantized = nullptr;

  std::uint32_t *const p_ = new std::uint32_t;
  unsigned char *const p = reinterpret_cast<unsigned char *>(p_);
  unsigned char *const q = p + sizeof(std::uint32_t) / 2;
  if (p != q) {
    void *quantized = static_cast<void *>(q);
    REQUIRE_THROWS(mgard::quantize(hierarchy, s, tolerance, coefficients,
                                   quantized, compressed.header));
    REQUIRE_THROWS(mgard::dequantize(compressed, quantized, dequantized));
  }
  delete p_;

  {
    mgard::pb::Quantization &q = *compressed.header.mutable_quantization();
    q.set_big_endian(true);
  }
  {
    std::int32_t *q_ = new std::int32_t[ndof];
    void *quantized = q_;
    REQUIRE_THROWS(mgard::quantize(hierarchy, s, tolerance, coefficients,
                                   quantized, compressed.header));
    REQUIRE_THROWS(mgard::dequantize(compressed, quantized, dequantized));
    delete[] q_;
  }
}
