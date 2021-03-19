#include <benchmark/benchmark.h>

#include <random>
#include <string>

#include "blas.hpp"
#include "moab/Core.hpp"

#include "mgard_api.h"

#include "MassMatrix.hpp"
#include "MeshLevel.hpp"
#include "UniformMeshHierarchy.hpp"
#include "data.hpp"

#include "TensorMassMatrix.hpp"
#include "TensorMultilevelCoefficientQuantizer.hpp"
#include "TensorProlongation.hpp"
#include "TensorRestriction.hpp"
#include "mgard.hpp"
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
  Real *const u = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  for (auto _ : state) {
    A(u);
  }
  std::free(u);

  state.SetComplexityN(ndof);
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
  Real *const u = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  Real *const v = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  for (auto _ : state) {
    mgard::shuffle(hierarchy, u, v);
  }
  std::free(v);
  std::free(u);

  state.SetComplexityN(ndof);
}

template <std::size_t N, typename Real>
static void BM_structured_unshuffle(benchmark::State &state) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(
      mesh_shape<N>(state.range(0)));

  const std::size_t ndof = hierarchy.ndof();
  Real *const u = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  Real *const v = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  for (auto _ : state) {
    mgard::unshuffle(hierarchy, u, v);
  }
  std::free(v);
  std::free(u);

  state.SetComplexityN(ndof);
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

  const std::size_t ndof = hierarchy.ndof();
  Real *const u = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  for (auto _ : state) {
    mgard::decompose(hierarchy, u);
  }
  std::free(u);

  state.SetComplexityN(ndof);
}

template <std::size_t N, typename Real>
static void BM_structured_recompose(benchmark::State &state) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(
      mesh_shape<N>(state.range(0)));

  const std::size_t ndof = hierarchy.ndof();
  Real *const u = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  for (auto _ : state) {
    mgard::recompose(hierarchy, u);
  }
  std::free(u);

  state.SetComplexityN(ndof);
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
  Real *const u = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  std::fill(u, u + ndof, 3);
  const Real s = 1;
  const Real tolerance = 0.75;
  const mgard::TensorMultilevelCoefficientQuantizer<N, Real, Int> quantizer(
      hierarchy, s, tolerance);

  [[maybe_unused]] Int total = 0;
  for (auto _ : state) {
    for (const Int n : quantizer(u)) {
      total += n;
    }
  }
  std::free(u);

  state.SetComplexityN(ndof);
}

template <std::size_t N, typename Int, typename Real>
static void BM_structured_dequantize(benchmark::State &state) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(
      mesh_shape<N>(state.range(0)));

  const std::size_t ndof = hierarchy.ndof();
  Int *const n = static_cast<Int *>(std::malloc(ndof * sizeof(int)));
  std::fill(n, n + ndof, -2);
  const Real s = -1;
  const Real tolerance = 0.25;
  const mgard::TensorMultilevelCoefficientDequantizer<N, Int, Real> dequantizer(
      hierarchy, s, tolerance);

  [[maybe_unused]] Real total = 0;
  for (auto _ : state) {
    for (const Real x : dequantizer(n, n + ndof)) {
      total += x;
    }
  }
  std::free(n);

  state.SetComplexityN(ndof);
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
  Real *const u = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  const Real s = 0;
  const Real tolerance = 0.5;
  for (auto _ : state) {
    // TODO: Think of a better way to do this.
    std::fill(u, u + ndof, 0);
    mgard::compress(hierarchy, u, s, tolerance);
  }
  std::free(u);

  state.SetComplexityN(ndof);
}

template <std::size_t N, typename Real>
static void BM_structured_decompress(benchmark::State &state) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(
      mesh_shape<N>(state.range(0)));

  const std::size_t ndof = hierarchy.ndof();
  Real *const u = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  // TODO: Think of a better way to do this.
  std::fill(u, u + ndof, 0);
  const Real s = 0;
  const Real tolerance = 0.5;
  const mgard::CompressedDataset<N, Real> compressed =
      mgard::compress(hierarchy, u, s, tolerance);

  for (auto _ : state) {
    mgard::decompress(compressed);
  }
  std::free(u);

  state.SetComplexityN(ndof);
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

static mgard::UniformMeshHierarchy
read_mesh_and_refine(moab::Core &mbcore, const std::string &filename,
                     const std::size_t L) {
  const std::string filepath = "tests/meshes/" + filename;
  [[maybe_unused]] const moab::ErrorCode ecode =
      mbcore.load_file(filepath.c_str());
  assert(ecode == moab::MB_SUCCESS);
  return mgard::UniformMeshHierarchy(mgard::MeshLevel(mbcore), L);
}

static void normalize(std::vector<double> &u) {
  const std::size_t N = u.size();
  double *const p = u.data();
  const double norm = blas::nrm2(N, p);
  ;
  blas::scal(N, 1 / norm, p);
}

static void BM_unstructured_decompose(benchmark::State &state,
                                      const std::string &filename) {
  std::default_random_engine gen;
  gen.seed(2987);
  std::uniform_real_distribution<double> dis(-1, 1);

  const int64_t L = state.range(0);
  moab::Core mbcore;
  mgard::UniformMeshHierarchy hierarchy =
      read_mesh_and_refine(mbcore, filename, L);

  const std::size_t N = hierarchy.ndof();
  std::vector<double> u_(N);
  for (double &x : u_) {
    x = dis(gen);
  }
  mgard::NodalCoefficients<double> u(u_.data());

  // Could preallocate buffer needed for decomposition.
  for (auto _ : state) {
    benchmark::DoNotOptimize(hierarchy.decompose(u));

    // Normalize `u` to prevent blowup. We could turn off the timing for
    // this, but it's `O(N)` so it shouldn't affect the complexity.
    normalize(u_);
  }

  // Could alternatively count up all the entities in all the levels.
  state.SetComplexityN(N);
  state.SetBytesProcessed(state.iterations() * N * sizeof(double));
}
BENCHMARK_CAPTURE(BM_unstructured_decompose, circle, "circle.msh")
    ->DenseRange(1, 6, 1)
    ->Complexity()
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_unstructured_decompose, seated, "seated.msh")
    ->DenseRange(3, 9, 1)
    ->Complexity()
    ->Unit(benchmark::kMillisecond);

static void BM_unstructured_recompose(benchmark::State &state,
                                      const std::string &filename) {
  std::default_random_engine gen;
  gen.seed(2340);
  std::uniform_real_distribution<double> dis(-2, 0);

  const int64_t L = state.range(0);
  moab::Core mbcore;
  mgard::UniformMeshHierarchy hierarchy =
      read_mesh_and_refine(mbcore, filename, L);

  const std::size_t N = hierarchy.ndof();
  std::vector<double> u_(N);
  for (double &x : u_) {
    x = dis(gen);
  }
  mgard::MultilevelCoefficients<double> u(u_.data());

  // Could preallocate buffer needed for recomposition.
  for (auto _ : state) {
    benchmark::DoNotOptimize(hierarchy.recompose(u));

    // Normalize `u` to prevent blowup. We could turn off the timing for
    // this, but it's `O(N)` so it shouldn't affect the complexity.
    normalize(u_);
  }

  // Could alternatively count up all the entities in all the levels.
  state.SetComplexityN(N);
  state.SetBytesProcessed(state.iterations() * N * sizeof(double));
}
BENCHMARK_CAPTURE(BM_unstructured_recompose, circle, "circle.msh")
    ->DenseRange(1, 6, 1)
    ->Complexity()
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_unstructured_recompose, pyramid, "pyramid.msh")
    ->DenseRange(3, 8, 1)
    ->Complexity()
    ->Unit(benchmark::kMillisecond);

static void BM_unstructured_mass_matrix(benchmark::State &state,
                                        const std::string &filename) {
  std::default_random_engine gen;
  gen.seed(3319);
  std::uniform_real_distribution<double> dis(-5, 0);

  const int64_t L = state.range(0);
  moab::Core mbcore;
  mgard::UniformMeshHierarchy hierarchy =
      read_mesh_and_refine(mbcore, filename, L);

  const std::size_t N = hierarchy.ndof();
  std::vector<double> u_(N);
  std::vector<double> rhs_(N);
  for (double &x : u_) {
    x = dis(gen);
  }
  double *const u = u_.data();
  double *const rhs = rhs_.data();
  const mgard::MeshLevel &MESH = hierarchy.meshes.back();
  const mgard::MassMatrix M(MESH);

  for (auto _ : state) {
    M(u, rhs);
    benchmark::DoNotOptimize(rhs[0]);
  }

  state.SetComplexityN(N);
  state.SetBytesProcessed(state.iterations() * N * sizeof(double));
}
BENCHMARK_CAPTURE(BM_unstructured_mass_matrix, circle, "circle.msh")
    ->DenseRange(2, 7, 1)
    ->Complexity()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
