#include <benchmark/benchmark.h>

#include <random>
#include <string>

#include "blas.hpp"
#include "moab/Core.hpp"

#include "mgard_api.h"

#include "MeshLevel.hpp"
#include "MassMatrix.hpp"
#include "UniformMeshHierarchy.hpp"

static void BM_MGARD(benchmark::State& state) {
  std::mt19937_64 gen(124124);
  std::normal_distribution<double> dis{0,1};

  std::vector<double> v(state.range(0));
  for (auto & x : v) {
    x = dis(gen);
  }

  int out_size;
  double tol = 1e-3;
  for (auto _ : state) {
    benchmark::DoNotOptimize(mgard_compress(/* double = 0 */ 0, v.data(), out_size,  4,  v.size()/4,  1, tol));
  }

  state.SetComplexityN(state.range(0));
  state.SetBytesProcessed(int64_t(state.range(0))*int64_t(state.iterations())*int64_t(sizeof(double)));
}
BENCHMARK(BM_MGARD)->RangeMultiplier(2)->Range(1<<4, 1<<22)->Complexity();

static mgard::UniformMeshHierarchy read_mesh_and_refine(
    moab::Core &mbcore, const std::string &filename, const std::size_t L
) {
    const std::string filepath = "tests/meshes/" + filename;
    moab::ErrorCode ecode = mbcore.load_file(filepath.c_str());
    assert(ecode == moab::MB_SUCCESS);
    return mgard::UniformMeshHierarchy(mgard::MeshLevel(mbcore), L);
}

static void normalize(std::vector<double> &u) {
    const std::size_t N = u.size();
    double * const p = u.data();
    const double norm = blas::nrm2(N, p);;
    blas::scal(N, 1 / norm, p);
}

static void BM_unstructured_decompose(
    benchmark::State &state, const std::string &filename
) {
    std::default_random_engine gen;
    gen.seed(2987);
    std::uniform_real_distribution<double> dis(-1, 1);

    const int64_t L = state.range(0);
    moab::Core mbcore;
    mgard::UniformMeshHierarchy hierarchy = read_mesh_and_refine(
        mbcore, filename, L
    );

    const std::size_t N = hierarchy.ndof();
    std::vector<double> u(N);
    for (double &x : u) {
        x = dis(gen);
    }

    //Could preallocate buffer needed for decomposition.
    for (auto _ : state) {
        moab::ErrorCode ecode;
        benchmark::DoNotOptimize(ecode = hierarchy.decompose(u.data()));
        assert(ecode == moab::MB_SUCCESS);

        //Normalize `u` to prevent blowup. We could turn off the timing for
        //this, but it's `O(N)` so it shouldn't affect the complexity.
        normalize(u);
    }

    //Could alternatively count up all the entities in all the levels.
    state.SetComplexityN(N);
    state.SetBytesProcessed(state.iterations() * N * sizeof(double));
}
BENCHMARK_CAPTURE(
    BM_unstructured_decompose, circle, "circle.msh"
)->DenseRange(1, 6, 1)->Complexity()->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(
    BM_unstructured_decompose, seated, "seated.msh"
)->DenseRange(3, 9, 1)->Complexity()->Unit(benchmark::kMillisecond);

static void BM_unstructured_recompose(
    benchmark::State &state, const std::string &filename
) {
    std::default_random_engine gen;
    gen.seed(2340);
    std::uniform_real_distribution<double> dis(-2, 0);

    const int64_t L = state.range(0);
    moab::Core mbcore;
    mgard::UniformMeshHierarchy hierarchy = read_mesh_and_refine(
        mbcore, filename, L
    );

    const std::size_t N = hierarchy.ndof();
    std::vector<double> u(N);
    for (double &x : u) {
        x = dis(gen);
    }

    //Could preallocate buffer needed for recomposition.
    for (auto _ : state) {
        moab::ErrorCode ecode;
        benchmark::DoNotOptimize(ecode = hierarchy.recompose(u.data()));
        assert(ecode == moab::MB_SUCCESS);

        //Normalize `u` to prevent blowup. We could turn off the timing for
        //this, but it's `O(N)` so it shouldn't affect the complexity.
        normalize(u);
    }

    //Could alternatively count up all the entities in all the levels.
    state.SetComplexityN(N);
    state.SetBytesProcessed(state.iterations() * N * sizeof(double));
}
BENCHMARK_CAPTURE(BM_unstructured_recompose, circle, "circle.msh")
    ->DenseRange(1, 6, 1)->Complexity()->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_unstructured_recompose, pyramid, "pyramid.msh")
    ->DenseRange(3, 8, 1)->Complexity()->Unit(benchmark::kMillisecond);

static void BM_unstructured_mass_matrix(
    benchmark::State &state, const std::string &filename
) {
    std::default_random_engine gen;
    gen.seed(3319);
    std::uniform_real_distribution<double> dis(-5, 0);

    const int64_t L = state.range(0);
    moab::Core mbcore;
    mgard::UniformMeshHierarchy hierarchy = read_mesh_and_refine(
        mbcore, filename, L
    );

    const std::size_t N = hierarchy.ndof();
    std::vector<double> u_(N);
    std::vector<double> rhs_(N);
    for (double &x : u_) {
        x = dis(gen);
    }
    double * const u = u_.data();
    double * const rhs = rhs_.data();
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
    ->DenseRange(2, 7, 1)->Complexity()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
