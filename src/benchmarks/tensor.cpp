#include <benchmark/benchmark.h>

#include <numeric>
#include <random>
#include <string>

#include "proto/mgard.pb.h"

#include "TensorMassMatrix.hpp"
#include "TensorMultilevelCoefficientQuantizer.hpp"
#include "TensorProlongation.hpp"
#include "TensorRestriction.hpp"
#include "compress.hpp"
#include "decompose.hpp"
#include "format.hpp"
#include "shuffle.hpp"

#define LOG_RANGE_LO 10
#define LOG_RANGE_MD 20
#define LOG_RANGE_HI 25

template <std::size_t N>
static std::array<std::size_t, N> mesh_shape(const std::size_t M) {
  std::array<std::size_t, N> shape;
  std::size_t product = 1;
  for (std::size_t i = 0; i < N; ++i) {
    product *= shape.at(i) =
        std::pow(M / product, static_cast<float>(1) / (N - i));
  }
  return shape;
}

template <std::size_t N, typename Real,
          template <std::size_t, typename> typename TLO>
static void BM_TensorLinearOperator(benchmark::State &state) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(
      mesh_shape<N>(state.range(0)));
  TLO<N, Real> A(hierarchy, hierarchy.L);

  const std::size_t ndof = hierarchy.ndof();
  Real *const u = new Real[ndof];
  for (auto _ : state) {
    A(u);
  }
  delete[] u;

  state.SetComplexityN(ndof);
  state.SetBytesProcessed(state.iterations() * ndof * sizeof(Real));
}

#define TLO_BENCHMARK_OPTIONS                                                  \
  ->RangeMultiplier(2)                                                         \
      ->Range(1 << LOG_RANGE_LO, 1 << LOG_RANGE_HI)                            \
      ->Unit(benchmark::kMillisecond)                                          \
      ->Complexity()

#define TLO_BENCHMARK(N, Real, TLO)                                            \
  BENCHMARK_TEMPLATE(BM_TensorLinearOperator, N, Real, TLO)                    \
  TLO_BENCHMARK_OPTIONS

TLO_BENCHMARK(1, double, mgard::TensorMassMatrix);
TLO_BENCHMARK(2, double, mgard::TensorMassMatrix);
TLO_BENCHMARK(3, double, mgard::TensorMassMatrix);

TLO_BENCHMARK(1, double, mgard::TensorMassMatrixInverse);
TLO_BENCHMARK(2, double, mgard::TensorMassMatrixInverse);
TLO_BENCHMARK(3, double, mgard::TensorMassMatrixInverse);

TLO_BENCHMARK(1, double, mgard::TensorRestriction);
TLO_BENCHMARK(2, double, mgard::TensorRestriction);
TLO_BENCHMARK(3, double, mgard::TensorRestriction);

TLO_BENCHMARK(1, double, mgard::TensorProlongationAddition);
TLO_BENCHMARK(2, double, mgard::TensorProlongationAddition);
TLO_BENCHMARK(3, double, mgard::TensorProlongationAddition);

template <std::size_t N, typename Real>
static void BM_structured_shuffle(benchmark::State &state) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(
      mesh_shape<N>(state.range(0)));

  const std::size_t ndof = hierarchy.ndof();
  Real *const u = new Real[ndof];
  Real *const v = new Real[ndof];
  for (auto _ : state) {
    mgard::shuffle(hierarchy, u, v);
  }
  delete[] v;
  delete[] u;

  state.SetComplexityN(ndof);
  state.SetBytesProcessed(state.iterations() * ndof * sizeof(Real));
}

template <std::size_t N, typename Real>
static void BM_structured_unshuffle(benchmark::State &state) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(
      mesh_shape<N>(state.range(0)));

  const std::size_t ndof = hierarchy.ndof();
  Real *const u = new Real[ndof];
  Real *const v = new Real[ndof];
  for (auto _ : state) {
    mgard::unshuffle(hierarchy, u, v);
  }
  delete[] v;
  delete[] u;

  state.SetComplexityN(ndof);
  state.SetBytesProcessed(state.iterations() * ndof * sizeof(Real));
}

#define SHUFFLE_UNSHUFFLE_BENCHMARK_OPTIONS                                    \
  ->RangeMultiplier(2)                                                         \
      ->Range(1 << LOG_RANGE_LO, 1 << LOG_RANGE_HI)                            \
      ->Unit(benchmark::kMillisecond)                                          \
      ->Complexity()

#define SHUFFLE_UNSHUFFLE_BENCHMARK(N, Real)                                   \
  BENCHMARK_TEMPLATE(BM_structured_shuffle, N, Real)                           \
  SHUFFLE_UNSHUFFLE_BENCHMARK_OPTIONS;                                         \
  BENCHMARK_TEMPLATE(BM_structured_unshuffle, N, Real)                         \
  SHUFFLE_UNSHUFFLE_BENCHMARK_OPTIONS

SHUFFLE_UNSHUFFLE_BENCHMARK(1, double);
SHUFFLE_UNSHUFFLE_BENCHMARK(2, double);
SHUFFLE_UNSHUFFLE_BENCHMARK(3, double);

template <std::size_t N, typename Real>
static void BM_structured_decompose(benchmark::State &state) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(
      mesh_shape<N>(state.range(0)));
  mgard::pb::Header header;
  mgard::populate_defaults(header);
  hierarchy.populate(header);

  const std::size_t ndof = hierarchy.ndof();
  Real *const u = new Real[ndof];
  for (auto _ : state) {
    mgard::decompose(hierarchy, header, u);
  }
  delete[] u;

  state.SetComplexityN(ndof);
  state.SetBytesProcessed(state.iterations() * ndof * sizeof(Real));
}

template <std::size_t N, typename Real>
static void BM_structured_recompose(benchmark::State &state) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(
      mesh_shape<N>(state.range(0)));
  mgard::pb::Header header;
  mgard::populate_defaults(header);
  hierarchy.populate(header);

  const std::size_t ndof = hierarchy.ndof();
  Real *const u = new Real[ndof];
  for (auto _ : state) {
    mgard::recompose(hierarchy, header, u);
  }
  delete[] u;

  state.SetComplexityN(ndof);
  state.SetBytesProcessed(state.iterations() * ndof * sizeof(Real));
}

#define DECOMPOSE_RECOMPOSE_BENCHMARK_OPTIONS                                  \
  ->RangeMultiplier(2)                                                         \
      ->Range(1 << LOG_RANGE_LO, 1 << LOG_RANGE_HI)                            \
      ->Unit(benchmark::kMillisecond)                                          \
      ->Complexity()

#define DECOMPOSE_RECOMPOSE_BENCHMARK(N, Real)                                 \
  BENCHMARK_TEMPLATE(BM_structured_decompose, N, Real)                         \
  DECOMPOSE_RECOMPOSE_BENCHMARK_OPTIONS;                                       \
  BENCHMARK_TEMPLATE(BM_structured_recompose, N, Real)                         \
  DECOMPOSE_RECOMPOSE_BENCHMARK_OPTIONS

DECOMPOSE_RECOMPOSE_BENCHMARK(1, double);
DECOMPOSE_RECOMPOSE_BENCHMARK(2, double);
DECOMPOSE_RECOMPOSE_BENCHMARK(3, double);

template <std::size_t N, typename Real, typename Int>
static void BM_structured_quantize(benchmark::State &state) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(
      mesh_shape<N>(state.range(0)));

  const std::size_t ndof = hierarchy.ndof();
  Real *const u = new Real[ndof];
  std::fill(u, u + ndof, 3);
  const Real s = 1;
  const Real tolerance = 0.75;
  using Quantizer = mgard::TensorMultilevelCoefficientQuantizer<N, Real, Int>;
  using It = typename Quantizer::iterator;
  const Quantizer quantizer(hierarchy, s, tolerance);
  const mgard::RangeSlice<It> quantized = quantizer(u);
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::accumulate(quantized.begin(), quantized.end(),
                                             static_cast<Int>(0)));
  }
  delete[] u;

  state.SetComplexityN(ndof);
  state.SetBytesProcessed(state.iterations() * ndof * sizeof(Real));
}

template <std::size_t N, typename Int, typename Real>
static void BM_structured_dequantize(benchmark::State &state) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(
      mesh_shape<N>(state.range(0)));

  const std::size_t ndof = hierarchy.ndof();
  Int *const n = new Int[ndof];
  std::fill(n, n + ndof, -2);
  const Real s = -1;
  const Real tolerance = 0.25;
  using Dequantizer =
      mgard::TensorMultilevelCoefficientDequantizer<N, Int, Real>;
  using It = typename Dequantizer::template iterator<Int *>;
  const Dequantizer dequantizer(hierarchy, s, tolerance);
  const mgard::RangeSlice<It> dequantized = dequantizer(n, n + ndof);
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::accumulate(
        dequantized.begin(), dequantized.end(), static_cast<Real>(0)));
  }
  delete[] n;

  state.SetComplexityN(ndof);
  state.SetBytesProcessed(state.iterations() * ndof * sizeof(Int));
}

#define QUANTIZE_DEQUANTIZE_BENCHMARK_OPTIONS                                  \
  ->RangeMultiplier(2)                                                         \
      ->Range(1 << LOG_RANGE_LO, 1 << LOG_RANGE_HI)                            \
      ->Unit(benchmark::kMillisecond)                                          \
      ->Complexity()

#define QUANTIZE_DEQUANTIZE_BENCHMARK(N, Real, Int)                            \
  BENCHMARK_TEMPLATE(BM_structured_quantize, N, Real, Int)                     \
  QUANTIZE_DEQUANTIZE_BENCHMARK_OPTIONS;                                       \
  BENCHMARK_TEMPLATE(BM_structured_dequantize, N, Int, Real)                   \
  QUANTIZE_DEQUANTIZE_BENCHMARK_OPTIONS

QUANTIZE_DEQUANTIZE_BENCHMARK(1, double, int);
QUANTIZE_DEQUANTIZE_BENCHMARK(2, double, int);
QUANTIZE_DEQUANTIZE_BENCHMARK(3, double, int);

template <std::size_t N, typename Real>
static void BM_structured_compress(benchmark::State &state) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(
      mesh_shape<N>(state.range(0)));

  const std::size_t ndof = hierarchy.ndof();
  Real *const u = new Real[ndof];
  const Real s = 0;
  const Real tolerance = 0.5;
  for (auto _ : state) {
    // TODO: Think of a better way to do this.
    std::fill(u, u + ndof, 0);
    mgard::compress(hierarchy, u, s, tolerance);
  }
  delete[] u;

  state.SetComplexityN(ndof);
  state.SetBytesProcessed(state.iterations() * ndof * sizeof(Real));
}

template <std::size_t N, typename Real>
static void BM_structured_decompress(benchmark::State &state) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(
      mesh_shape<N>(state.range(0)));

  const std::size_t ndof = hierarchy.ndof();
  Real *const u = new Real[ndof];
  // TODO: Think of a better way to do this.
  std::fill(u, u + ndof, 0);
  const Real s = 0;
  const Real tolerance = 0.5;
  const mgard::CompressedDataset<N, Real> compressed =
      mgard::compress(hierarchy, u, s, tolerance);

  for (auto _ : state) {
    mgard::decompress(compressed);
  }
  delete[] u;

  state.SetComplexityN(ndof);
  state.SetBytesProcessed(state.iterations() * ndof * sizeof(Real));
}

#define COMPRESS_DECOMPRESS_BENCHMARK_OPTIONS                                  \
  ->RangeMultiplier(2)                                                         \
      ->Range(1 << LOG_RANGE_LO, 1 << LOG_RANGE_MD)                            \
      ->Unit(benchmark::kMillisecond)                                          \
      ->Complexity()

#define COMPRESS_DECOMPRESS_BENCHMARK(N, Real)                                 \
  BENCHMARK_TEMPLATE(BM_structured_compress, N, Real)                          \
  COMPRESS_DECOMPRESS_BENCHMARK_OPTIONS;                                       \
  BENCHMARK_TEMPLATE(BM_structured_decompress, N, Real)                        \
  COMPRESS_DECOMPRESS_BENCHMARK_OPTIONS

COMPRESS_DECOMPRESS_BENCHMARK(1, double);
COMPRESS_DECOMPRESS_BENCHMARK(2, double);
COMPRESS_DECOMPRESS_BENCHMARK(3, double);

BENCHMARK_MAIN();
