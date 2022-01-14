#include <benchmark/benchmark.h>

#include <cstddef>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "moab/Core.hpp"

#include "blas.hpp"

#include "unstructured/MassMatrix.hpp"
#include "unstructured/MeshLevel.hpp"
#include "unstructured/UniformMeshHierarchy.hpp"
#include "unstructured/data.hpp"

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
  std::generate(u_.begin(), u_.end(), [&]() -> double { return dis(gen); });
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
  std::generate(u_.begin(), u_.end(), [&]() -> double { return dis(gen); });
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
  std::generate(u_.begin(), u_.end(), [&]() -> double { return dis(gen); });
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
